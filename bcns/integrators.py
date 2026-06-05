"""Pseudo-time integrators for BCNS vorticity evolution."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .config import FlowConfig
from .navier_stokes import (
    anisotropic_diffusion_operator,
    anisotropic_diffusivity,
    jacobian_upwind,
    ns_rhs,
)
from .operators import _validate_scalar_field, laplacian_5pt, masked_l2_norm, perpendicular_gradient


@dataclass
class FlowStepResult:
    """Single-step vorticity result and diagnostics."""

    vorticity: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _validate_step_inputs(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    mask_unknown: torch.Tensor,
) -> None:
    _validate_scalar_field(intensity, "intensity")
    if vorticity.shape != intensity.shape or mask_unknown.shape != intensity.shape:
        raise ValueError("intensity, vorticity, and mask_unknown must have identical shape.")


def _known_boundary_values(
    vorticity: torch.Tensor,
    mask_unknown: torch.Tensor,
    vorticity_boundary: Optional[torch.Tensor],
    config: FlowConfig,
) -> torch.Tensor:
    if config.vorticity_boundary_mode == "hard" and vorticity_boundary is not None:
        if vorticity_boundary.shape != vorticity.shape:
            raise ValueError("vorticity_boundary must match vorticity shape.")
        return vorticity_boundary
    return vorticity


def _apply_boundary(
    candidate: torch.Tensor,
    old_vorticity: torch.Tensor,
    mask_unknown: torch.Tensor,
    vorticity_boundary: Optional[torch.Tensor],
    config: FlowConfig,
) -> torch.Tensor:
    mask = mask_unknown.to(dtype=candidate.dtype, device=candidate.device)
    known_values = _known_boundary_values(old_vorticity, mask_unknown, vorticity_boundary, config)
    if config.vorticity_boundary_mode == "hard" and vorticity_boundary is not None:
        return candidate * mask + known_values * (1.0 - mask)
    return candidate * mask + old_vorticity * (1.0 - mask)


def compute_cfl_dt_bound(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    config: FlowConfig,
) -> float:
    """Return conservative FTCS advection plus explicit diffusion timestep bound."""

    h = config.poisson.h
    vx, vy = perpendicular_gradient(intensity, h)
    max_velocity = torch.sqrt(vx * vx + vy * vy).max()
    diffusivity = anisotropic_diffusivity(vorticity, config.kappa, config.smoothing_sigma, h)
    max_g = diffusivity.max()
    eps = torch.finfo(vorticity.dtype).eps
    adv_bound = h / (max_velocity + eps)
    if config.nu > 0:
        diff_bound = (h * h) / (4.0 * config.nu * max_g + eps)
        bound = torch.minimum(adv_bound, diff_bound)
    else:
        bound = adv_bound
    return float((config.cfl * bound).item())


def ftcs_step(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    mask_unknown: torch.Tensor,
    vorticity_boundary: Optional[torch.Tensor],
    config: FlowConfig,
) -> FlowStepResult:
    """Advance vorticity by explicit Euler/FTCS using upwind advection."""

    _validate_step_inputs(intensity, vorticity, mask_unknown)
    cfl_bound = compute_cfl_dt_bound(intensity, vorticity, config)
    if config.check_cfl and config.dt > cfl_bound:
        raise ValueError(
            f"FTCS dt={config.dt:g} violates conservative CFL bound {cfl_bound:g}."
        )
    rhs, diagnostics = ns_rhs(
        intensity,
        vorticity,
        nu=config.nu,
        kappa=config.kappa,
        smoothing_sigma=config.smoothing_sigma,
        h=config.poisson.h,
        use_upwind=True,
    )
    candidate = vorticity + config.dt * rhs
    updated = _apply_boundary(candidate, vorticity, mask_unknown, vorticity_boundary, config)
    diagnostics.update({"cfl_dt_bound": cfl_bound, "dt": config.dt, "integrator": "ftcs"})
    return FlowStepResult(updated, diagnostics)


def _cg_solve_masked(
    apply_a: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    mask: torch.Tensor,
    tol: float,
    max_iter: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x = torch.zeros_like(b)
    r = b - apply_a(x)
    p = r.clone()
    rs_old = (r * r * mask).sum()
    eps = torch.finfo(b.dtype).eps
    history = [float(torch.sqrt(rs_old / mask.sum().clamp_min(1.0)).item())]
    num_iter = 0
    for it in range(1, max_iter + 1):
        ap = apply_a(p)
        denom = (p * ap * mask).sum().clamp_min(eps)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r * mask).sum()
        num_iter = it
        history.append(float(torch.sqrt(rs_new / mask.sum().clamp_min(1.0)).item()))
        if history[-1] <= tol:
            break
        beta = rs_new / rs_old.clamp_min(eps)
        p = r + beta * p
        rs_old = rs_new
    return x * mask, {
        "linear_solver": "cg",
        "linear_num_iter": num_iter,
        "linear_residual": history[-1],
        "linear_converged": history[-1] <= tol,
    }


def _implicit_diffusion_solve(
    vorticity: torch.Tensor,
    rhs_base: torch.Tensor,
    mask_unknown: torch.Tensor,
    boundary_values: torch.Tensor,
    diffusivity: torch.Tensor,
    alpha: float,
    config: FlowConfig,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    mask = mask_unknown.to(dtype=vorticity.dtype, device=vorticity.device)
    known = 1.0 - mask
    boundary_full = boundary_values * known
    boundary_correction = alpha * anisotropic_diffusion_operator(
        boundary_full, diffusivity, config.poisson.h
    )
    b = (rhs_base + boundary_correction) * mask

    def apply_a(delta: torch.Tensor) -> torch.Tensor:
        delta_unknown = delta * mask
        return (delta_unknown - alpha * anisotropic_diffusion_operator(delta_unknown, diffusivity, config.poisson.h)) * mask

    x, diagnostics = _cg_solve_masked(
        apply_a,
        b,
        mask,
        tol=config.implicit_tol,
        max_iter=config.implicit_max_iter,
    )
    return x * mask + boundary_values * known, diagnostics


def imex_backward_euler_step(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    mask_unknown: torch.Tensor,
    vorticity_boundary: Optional[torch.Tensor],
    config: FlowConfig,
) -> FlowStepResult:
    """Advance with explicit advection and implicit backward-Euler diffusion."""

    _validate_step_inputs(intensity, vorticity, mask_unknown)
    h = config.poisson.h
    jacobian = jacobian_upwind(intensity, vorticity, h)
    diffusivity = anisotropic_diffusivity(vorticity, config.kappa, config.smoothing_sigma, h)
    boundary_values = _known_boundary_values(vorticity, mask_unknown, vorticity_boundary, config)
    rhs_base = vorticity - config.dt * jacobian
    updated, lin_diag = _implicit_diffusion_solve(
        vorticity,
        rhs_base,
        mask_unknown,
        boundary_values,
        diffusivity,
        alpha=config.dt * config.nu,
        config=config,
    )
    updated = _apply_boundary(updated, vorticity, mask_unknown, vorticity_boundary, config)
    rhs, rhs_diag = ns_rhs(intensity, updated, config.nu, config.kappa, config.smoothing_sigma, h)
    diagnostics = {
        "integrator": "imex_be",
        "dt": config.dt,
        "jacobian_norm": float(masked_l2_norm(jacobian, torch.ones_like(jacobian)).item()),
        "rhs_norm": rhs_diag["rhs_norm"],
    }
    diagnostics.update(lin_diag)
    return FlowStepResult(updated, diagnostics)


def imex_crank_nicolson_step(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    mask_unknown: torch.Tensor,
    vorticity_boundary: Optional[torch.Tensor],
    config: FlowConfig,
) -> FlowStepResult:
    """Advance with explicit advection and Crank-Nicolson implicit diffusion."""

    _validate_step_inputs(intensity, vorticity, mask_unknown)
    h = config.poisson.h
    jacobian = jacobian_upwind(intensity, vorticity, h)
    diffusivity = anisotropic_diffusivity(vorticity, config.kappa, config.smoothing_sigma, h)
    boundary_values = _known_boundary_values(vorticity, mask_unknown, vorticity_boundary, config)
    alpha = 0.5 * config.dt * config.nu
    rhs_base = (
        vorticity
        + alpha * anisotropic_diffusion_operator(vorticity, diffusivity, h)
        - config.dt * jacobian
    )
    updated, lin_diag = _implicit_diffusion_solve(
        vorticity,
        rhs_base,
        mask_unknown,
        boundary_values,
        diffusivity,
        alpha=alpha,
        config=config,
    )
    updated = _apply_boundary(updated, vorticity, mask_unknown, vorticity_boundary, config)
    _, rhs_diag = ns_rhs(intensity, updated, config.nu, config.kappa, config.smoothing_sigma, h)
    diagnostics = {
        "integrator": "imex_cn",
        "dt": config.dt,
        "jacobian_norm": float(masked_l2_norm(jacobian, torch.ones_like(jacobian)).item()),
        "rhs_norm": rhs_diag["rhs_norm"],
    }
    diagnostics.update(lin_diag)
    return FlowStepResult(updated, diagnostics)
