"""Finite BCNS vorticity-stream flow structure targets for DPS Step 3."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import FlowConfig, PoissonSolverConfig
from .dps_adapter import (
    dps_known_to_bcns_unknown,
    hard_project_clean,
    validate_image_tensor,
    validate_mask_tensor,
)
from .flow import evolve_bcns_flow
from .navier_stokes import ns_rhs
from .operators import laplacian_5pt, masked_l2_norm
from .poisson import poisson_residual
from .proximal import normalized_known_luminance_smooth, structure_image


@dataclass
class FlowStructureTargetResult:
    """Scalar terminal flow structure plus BCNS flow diagnostics."""

    structure: torch.Tensor
    initial_intensity: torch.Tensor
    final_vorticity: torch.Tensor
    boundary_intensity: torch.Tensor
    boundary_vorticity: Optional[torch.Tensor]
    mask_unknown: torch.Tensor
    history: List[dict]
    diagnostics: dict


def _validate_mu_measurement_mask(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
) -> None:
    validate_image_tensor(mu, "mu")
    if measurement.shape != mu.shape:
        raise ValueError("measurement must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")


def make_initial_intensity(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    mode: str = "projected_mu",
) -> torch.Tensor:
    """Return the scalar initial intensity for finite BCNS flow."""

    _validate_mu_measurement_mask(mu, measurement, mask_known)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if mode == "mu":
        return structure_image(mu, sigma)
    if mode == "projected_mu":
        return structure_image(hard_project_clean(mu, measurement, mask_known), sigma)
    if mode == "normalized_known_smooth":
        return normalized_known_luminance_smooth(measurement, mask_known, sigma)
    raise ValueError("initial mode must be one of 'mu', 'projected_mu', or 'normalized_known_smooth'.")


def make_boundary_intensity(
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    mode: str = "normalized_known_smooth",
) -> torch.Tensor:
    """Return scalar known-side boundary intensity from the measurement."""

    validate_image_tensor(measurement, "measurement")
    validate_mask_tensor(mask_known, measurement, "mask_known")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if mode == "normalized_known_smooth":
        return normalized_known_luminance_smooth(measurement, mask_known, sigma)
    if mode == "measurement_only_smooth":
        return structure_image(measurement, sigma)
    raise ValueError("boundary mode must be one of 'normalized_known_smooth' or 'measurement_only_smooth'.")


def make_boundary_vorticity(
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    mode: str = "none",
    h: float = 1.0,
) -> Optional[torch.Tensor]:
    """Return optional scalar boundary vorticity for hard boundary modes."""

    if h <= 0:
        raise ValueError("h must be positive.")
    if mode == "none":
        return None
    if mode == "fd_laplacian":
        boundary_intensity = make_boundary_intensity(
            measurement=measurement,
            mask_known=mask_known,
            sigma=sigma,
            mode="normalized_known_smooth",
        )
        return laplacian_5pt(boundary_intensity, h)
    raise ValueError("boundary_vorticity_mode must be one of 'none' or 'fd_laplacian'.")


def _last_history_value(history: List[dict], key: str) -> float:
    if not history:
        return float("nan")
    value = history[-1].get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_flow_structure_target(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    structure_sigma: float = 1.0,
    initial_mode: str = "projected_mu",
    boundary_mode: str = "normalized_known_smooth",
    boundary_vorticity_mode: str = "none",
    integrator: str = "imex_be",
    poisson_method: str = "sor_rb",
    poisson_max_iter: int = 100,
    poisson_tol: float = 1e-4,
    poisson_omega: float = 1.7,
    h: float = 1.0,
    dt: float = 1e-3,
    pseudo_time: float = 3e-3,
    nu: float = 0.1,
    kappa: float = 0.1,
    smoothing_sigma: float = 1.0,
    cfl: float = 0.25,
    check_cfl: bool = True,
    vorticity_boundary_mode: str = "none",
) -> FlowStructureTargetResult:
    """Build a detached scalar terminal flow structure target."""

    _validate_mu_measurement_mask(mu, measurement, mask_known)
    mask_unknown = dps_known_to_bcns_unknown(mask_known).to(dtype=mu.dtype, device=mu.device)
    initial_intensity = make_initial_intensity(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        sigma=structure_sigma,
        mode=initial_mode,
    )
    boundary_intensity = make_boundary_intensity(
        measurement=measurement,
        mask_known=mask_known,
        sigma=structure_sigma,
        mode=boundary_mode,
    )
    boundary_vorticity = make_boundary_vorticity(
        measurement=measurement,
        mask_known=mask_known,
        sigma=structure_sigma,
        mode=boundary_vorticity_mode,
        h=h,
    )
    poisson_config = PoissonSolverConfig(
        method=poisson_method,
        max_iter=poisson_max_iter,
        tol=poisson_tol,
        omega=poisson_omega,
        h=h,
        record_history=True,
    )
    flow_config = FlowConfig(
        integrator=integrator,
        poisson=poisson_config,
        dt=dt,
        pseudo_time=pseudo_time,
        nu=nu,
        kappa=kappa,
        smoothing_sigma=smoothing_sigma,
        cfl=cfl,
        check_cfl=check_cfl,
        vorticity_boundary_mode=vorticity_boundary_mode,
    )
    with torch.no_grad():
        try:
            flow_result = evolve_bcns_flow(
                initial_intensity=initial_intensity.detach(),
                boundary_intensity=boundary_intensity.detach(),
                boundary_vorticity=None if boundary_vorticity is None else boundary_vorticity.detach(),
                mask_unknown=mask_unknown.detach(),
                config=flow_config,
            )
        except ValueError as exc:
            if integrator == "ftcs" and "CFL" in str(exc):
                raise ValueError(f"FTCS CFL check failed while building flow target: {exc}") from exc
            raise

        structure = flow_result.intensity.detach()
        final_vorticity = flow_result.vorticity.detach()
        final_residual_tensor = poisson_residual(structure, final_vorticity, mask_unknown, h)
        final_poisson_residual = float(masked_l2_norm(final_residual_tensor, mask_unknown).item())
        if flow_result.history:
            final_rhs_norm = _last_history_value(flow_result.history, "rhs_norm")
        else:
            try:
                _, rhs_diag = ns_rhs(structure, final_vorticity, nu, kappa, smoothing_sigma, h)
                final_rhs_norm = float(rhs_diag.get("rhs_norm", float("nan")))
            except ValueError:
                final_rhs_norm = float("nan")
        has_nan = bool(
            (not torch.isfinite(structure).all().item())
            or (not torch.isfinite(final_vorticity).all().item())
        )
        initial_disp = torch.linalg.norm((structure - initial_intensity).reshape(-1))

    diagnostics: Dict[str, object] = {
        "integrator": integrator,
        "flow_integrator": integrator,
        "pseudo_time": float(pseudo_time),
        "dt": float(dt),
        "num_flow_steps": len(flow_result.history),
        "flow_num_steps": len(flow_result.history),
        "poisson_method": poisson_method,
        "total_runtime_sec": float(flow_result.runtime_sec),
        "flow_runtime_sec": float(flow_result.runtime_sec),
        "final_poisson_residual": final_poisson_residual,
        "flow_final_poisson_residual": final_poisson_residual,
        "final_rhs_norm": final_rhs_norm,
        "flow_final_rhs_norm": final_rhs_norm,
        "has_nan": has_nan,
        "flow_has_nan": has_nan,
        "boundary_vorticity_mode": boundary_vorticity_mode,
        "vorticity_boundary_mode": vorticity_boundary_mode,
        "initial_mode": initial_mode,
        "boundary_mode": boundary_mode,
        "nu": float(nu),
        "kappa": float(kappa),
        "smoothing_sigma": float(smoothing_sigma),
        "cfl": float(cfl),
        "check_cfl": bool(check_cfl),
        "flow_structure_disp": float(initial_disp.item()),
    }
    return FlowStructureTargetResult(
        structure=structure,
        initial_intensity=initial_intensity.detach(),
        final_vorticity=final_vorticity,
        boundary_intensity=boundary_intensity.detach(),
        boundary_vorticity=None if boundary_vorticity is None else boundary_vorticity.detach(),
        mask_unknown=mask_unknown.detach(),
        history=[dict(item) for item in flow_result.history],
        diagnostics=diagnostics,
    )
