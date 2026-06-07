"""Poisson and harmonic scalar structure targets for BCNS DPS Step 2."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import PoissonSolverConfig
from .dps_adapter import (
    dps_known_to_bcns_unknown,
    hard_project_clean,
    validate_image_tensor,
    validate_mask_tensor,
)
from .operators import laplacian_5pt
from .poisson import solve_poisson
from .proximal import normalized_known_luminance_smooth, structure_image


@dataclass
class PoissonStructureTargetResult:
    """Scalar structure target plus Poisson solver diagnostics."""

    structure: torch.Tensor
    boundary_values: torch.Tensor
    rhs_w: torch.Tensor
    mask_unknown: torch.Tensor
    residual_history: List[float]
    diagnostics: dict


def make_boundary_values(
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    mode: str = "normalized_known_smooth",
) -> torch.Tensor:
    """Build scalar known-side boundary values from a DPS measurement."""

    validate_image_tensor(measurement, "measurement")
    validate_mask_tensor(mask_known, measurement, "mask_known")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if mode == "normalized_known_smooth":
        return normalized_known_luminance_smooth(measurement, mask_known, sigma)
    if mode == "measurement_only_smooth":
        return structure_image(measurement, sigma)
    raise ValueError(
        "boundary mode must be one of 'normalized_known_smooth' or "
        "'measurement_only_smooth'."
    )


def make_poisson_rhs(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    rhs_mode: str,
    h: float,
) -> torch.Tensor:
    """Build ``w_mu`` for ``Delta_h I = w_mu`` as a scalar field."""

    validate_image_tensor(mu, "mu")
    if measurement.shape != mu.shape:
        raise ValueError("measurement must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if rhs_mode == "zero":
        return torch.zeros((mu.shape[0], 1, mu.shape[-2], mu.shape[-1]), dtype=mu.dtype, device=mu.device)
    if rhs_mode == "mu_laplacian":
        return laplacian_5pt(structure_image(mu, sigma), h)
    if rhs_mode == "projected_mu_laplacian":
        projected = hard_project_clean(mu, measurement, mask_known)
        return laplacian_5pt(structure_image(projected, sigma), h)
    raise ValueError(
        "rhs_mode must be one of 'zero', 'mu_laplacian', or "
        "'projected_mu_laplacian'."
    )


def _final_residual(history: List[float]) -> float:
    return float(history[-1]) if history else float("nan")


def _solver_dt_for_method(
    poisson_method: str,
    solver_dt: Optional[float],
    solver_dt_ftcs: float,
    solver_dt_be: float,
    solver_dt_cn: float,
) -> float:
    if solver_dt is not None:
        return float(solver_dt)
    if poisson_method == "ftcs":
        return float(solver_dt_ftcs)
    if poisson_method == "be":
        return float(solver_dt_be)
    if poisson_method == "cn":
        return float(solver_dt_cn)
    return 1.0


def build_poisson_structure_target(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float = 1.0,
    rhs_mode: str = "zero",
    boundary_mode: str = "normalized_known_smooth",
    poisson_method: str = "sor_rb",
    poisson_max_iter: int = 200,
    poisson_tol: float = 1e-4,
    poisson_omega: float = 1.7,
    poisson_h: float = 1.0,
    solver_dt: Optional[float] = None,
    solver_dt_ftcs: float = 0.2,
    solver_dt_be: float = 1.0,
    solver_dt_cn: float = 1.0,
    record_history: bool = True,
) -> PoissonStructureTargetResult:
    """Solve a masked harmonic/Poisson scalar structure target in the hole."""

    validate_image_tensor(mu, "mu")
    if measurement.shape != mu.shape:
        raise ValueError("measurement must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    mask_unknown = dps_known_to_bcns_unknown(mask_known).to(dtype=mu.dtype, device=mu.device)
    boundary_values = make_boundary_values(
        measurement=measurement,
        mask_known=mask_known,
        sigma=sigma,
        mode=boundary_mode,
    )
    rhs_w = make_poisson_rhs(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        sigma=sigma,
        rhs_mode=rhs_mode,
        h=poisson_h,
    )
    selected_solver_dt = _solver_dt_for_method(
        poisson_method=poisson_method,
        solver_dt=solver_dt,
        solver_dt_ftcs=solver_dt_ftcs,
        solver_dt_be=solver_dt_be,
        solver_dt_cn=solver_dt_cn,
    )
    config = PoissonSolverConfig(
        method=poisson_method,
        max_iter=poisson_max_iter,
        tol=poisson_tol,
        omega=poisson_omega,
        h=poisson_h,
        dt=selected_solver_dt,
        record_history=record_history,
    )
    with torch.no_grad():
        solver_result = solve_poisson(
            rhs_w=rhs_w.detach(),
            boundary_values=boundary_values.detach(),
            mask_unknown=mask_unknown.detach(),
            config=config,
        )
        known = 1.0 - mask_unknown
        structure = solver_result.solution * mask_unknown + boundary_values * known
        initial_structure = structure_image(mu, sigma)
        target_disp = torch.linalg.norm((structure - initial_structure).reshape(-1))
        unknown_fraction = mask_unknown.mean()

    diagnostics: Dict[str, object] = {
        "poisson_method": poisson_method,
        "elliptic_solver": poisson_method,
        "solver_tol": float(poisson_tol),
        "solver_max_iter": int(poisson_max_iter),
        "solver_dt": float(selected_solver_dt),
        "solver_num_iter": int(solver_result.num_iter),
        "poisson_num_iter": int(solver_result.num_iter),
        "poisson_converged": bool(solver_result.converged),
        "poisson_final_residual": _final_residual(solver_result.residual_history),
        "poisson_runtime_sec": float(solver_result.runtime_sec),
        "rhs_mode": rhs_mode,
        "boundary_mode": boundary_mode,
        "mask_unknown_fraction": float(unknown_fraction.item()),
        "target_disp": float(target_disp.item()),
    }
    diagnostics.update(solver_result.diagnostics)
    diagnostics["poisson_method"] = poisson_method
    diagnostics["elliptic_solver"] = poisson_method
    diagnostics["solver_tol"] = float(poisson_tol)
    diagnostics["solver_max_iter"] = int(poisson_max_iter)
    diagnostics["solver_dt"] = float(selected_solver_dt)
    diagnostics["solver_num_iter"] = int(solver_result.num_iter)
    return PoissonStructureTargetResult(
        structure=structure.detach(),
        boundary_values=boundary_values.detach(),
        rhs_w=rhs_w.detach(),
        mask_unknown=mask_unknown.detach(),
        residual_history=list(solver_result.residual_history),
        diagnostics=diagnostics,
    )
