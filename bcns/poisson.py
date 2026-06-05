"""Masked Poisson reconstruction solvers for ``Delta_h I = w``.

Jacobi, red-black Gauss-Seidel, red-black SOR, and the dense reference solve
work directly with the sign convention ``Delta_h I = w``. The matrix-free CG
solver uses the SPD equivalent ``-Delta_h I = -w`` restricted to unknown pixels.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .config import PoissonSolverConfig
from .operators import _validate_scalar_field, laplacian_5pt, masked_l2_norm


@dataclass
class SolverResult:
    """Result returned by all Poisson solvers."""

    solution: torch.Tensor
    residual_history: List[float]
    num_iter: int
    converged: bool
    runtime_sec: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _validate_poisson_inputs(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
) -> None:
    _validate_scalar_field(rhs_w, "rhs_w")
    if boundary_values.shape != rhs_w.shape or mask_unknown.shape != rhs_w.shape:
        raise ValueError("rhs_w, boundary_values, and mask_unknown must have identical shape.")
    if not torch.all((mask_unknown.detach() == 0) | (mask_unknown.detach() == 1)):
        raise ValueError("mask_unknown must be binary.")


def _mask_float(mask_unknown: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask_unknown.to(device=like.device, dtype=like.dtype)


def _initial_solution(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    initial: Optional[torch.Tensor],
) -> torch.Tensor:
    mask = _mask_float(mask_unknown, rhs_w)
    known = 1.0 - mask
    if initial is None:
        solution = torch.zeros_like(rhs_w)
    else:
        if initial.shape != rhs_w.shape:
            raise ValueError("initial must match rhs_w shape.")
        solution = initial.clone()
    return solution * mask + boundary_values * known


def poisson_residual(
    solution: torch.Tensor,
    rhs_w: torch.Tensor,
    mask_unknown: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Return residual ``Delta_h solution - rhs_w`` inside Omega."""

    _validate_poisson_inputs(rhs_w, solution, mask_unknown)
    mask = _mask_float(mask_unknown, solution)
    return (laplacian_5pt(solution, h) - rhs_w) * mask


def _residual_norm(solution: torch.Tensor, rhs_w: torch.Tensor, mask_unknown: torch.Tensor, h: float) -> float:
    return float(masked_l2_norm(poisson_residual(solution, rhs_w, mask_unknown, h), mask_unknown).item())


def _neighbor_sum(x: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    return (
        padded[..., :-2, 1:-1]
        + padded[..., 2:, 1:-1]
        + padded[..., 1:-1, :-2]
        + padded[..., 1:-1, 2:]
    )


def _finish(
    solution: torch.Tensor,
    history: List[float],
    start: float,
    tol: float,
    num_iter: int,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> SolverResult:
    converged = bool(history and history[-1] <= tol)
    return SolverResult(
        solution=solution,
        residual_history=history,
        num_iter=num_iter,
        converged=converged,
        runtime_sec=time.time() - start,
        diagnostics=diagnostics or {},
    )


def solve_poisson_jacobi(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: PoissonSolverConfig,
    initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Solve masked Poisson reconstruction with Jacobi iteration."""

    _validate_poisson_inputs(rhs_w, boundary_values, mask_unknown)
    start = time.time()
    mask = _mask_float(mask_unknown, rhs_w)
    known = 1.0 - mask
    solution = _initial_solution(rhs_w, boundary_values, mask_unknown, initial)
    history: List[float] = []
    num_iter = 0
    for it in range(1, config.max_iter + 1):
        new_unknown = (_neighbor_sum(solution) - (config.h * config.h) * rhs_w) / 4.0
        solution = new_unknown * mask + boundary_values * known
        num_iter = it
        if config.record_history or it == config.max_iter:
            history.append(_residual_norm(solution, rhs_w, mask_unknown, config.h))
        if history and history[-1] <= config.tol:
            break
    return _finish(solution, history, start, config.tol, num_iter)


def _red_black_masks(mask_unknown: torch.Tensor, like: torch.Tensor) -> Dict[str, torch.Tensor]:
    height, width = mask_unknown.shape[-2:]
    y = torch.arange(height, device=like.device).view(height, 1)
    x = torch.arange(width, device=like.device).view(1, width)
    parity = ((x + y) % 2).view(1, 1, height, width)
    mask = mask_unknown.bool()
    return {
        "red": (mask & (parity == 0)).to(dtype=like.dtype),
        "black": (mask & (parity == 1)).to(dtype=like.dtype),
    }


def _solve_poisson_red_black(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: PoissonSolverConfig,
    initial: Optional[torch.Tensor],
    omega: float,
    method_name: str,
) -> SolverResult:
    _validate_poisson_inputs(rhs_w, boundary_values, mask_unknown)
    start = time.time()
    mask = _mask_float(mask_unknown, rhs_w)
    known = 1.0 - mask
    colors = _red_black_masks(mask_unknown, rhs_w)
    solution = _initial_solution(rhs_w, boundary_values, mask_unknown, initial)
    history: List[float] = []
    num_iter = 0
    for it in range(1, config.max_iter + 1):
        for color in ("red", "black"):
            color_mask = colors[color]
            base = (_neighbor_sum(solution) - (config.h * config.h) * rhs_w) / 4.0
            relaxed = (1.0 - omega) * solution + omega * base
            solution = relaxed * color_mask + solution * (1.0 - color_mask)
            solution = solution * mask + boundary_values * known
        num_iter = it
        if config.record_history or it == config.max_iter:
            history.append(_residual_norm(solution, rhs_w, mask_unknown, config.h))
        if history and history[-1] <= config.tol:
            break
    return _finish(solution, history, start, config.tol, num_iter, {"method": method_name, "omega": omega})


def solve_poisson_gauss_seidel_red_black(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: PoissonSolverConfig,
    initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Solve masked Poisson reconstruction with red-black Gauss-Seidel."""

    return _solve_poisson_red_black(rhs_w, boundary_values, mask_unknown, config, initial, 1.0, "gs_rb")


def solve_poisson_sor_red_black(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: PoissonSolverConfig,
    initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Solve masked Poisson reconstruction with vectorized red-black SOR."""

    return _solve_poisson_red_black(
        rhs_w, boundary_values, mask_unknown, config, initial, config.omega, "sor_rb"
    )


def _dot_masked(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=a.dtype, device=a.device)
    return (a * b * mask_f).sum()


def solve_poisson_cg(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: PoissonSolverConfig,
    initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Matrix-free CG on ``-Delta_h I = -w`` restricted to Omega."""

    _validate_poisson_inputs(rhs_w, boundary_values, mask_unknown)
    start = time.time()
    mask = _mask_float(mask_unknown, rhs_w)
    known = 1.0 - mask
    boundary_full = boundary_values * known
    b = (-rhs_w + laplacian_5pt(boundary_full, config.h)) * mask

    def apply_a(delta: torch.Tensor) -> torch.Tensor:
        return (-laplacian_5pt(delta * mask, config.h)) * mask

    if initial is None:
        x = torch.zeros_like(rhs_w)
    else:
        x = initial * mask
    r = b - apply_a(x)
    p = r.clone()
    rs_old = _dot_masked(r, r, mask)
    history: List[float] = [float(torch.sqrt(rs_old / mask.sum().clamp_min(1.0)).item())]
    num_iter = 0
    eps = torch.finfo(rhs_w.dtype).eps
    for it in range(1, config.max_iter + 1):
        ap = apply_a(p)
        alpha = rs_old / _dot_masked(p, ap, mask).clamp_min(eps)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = _dot_masked(r, r, mask)
        num_iter = it
        if config.record_history or it == config.max_iter:
            history.append(float(torch.sqrt(rs_new / mask.sum().clamp_min(1.0)).item()))
        if history[-1] <= config.tol:
            rs_old = rs_new
            break
        beta = rs_new / rs_old.clamp_min(eps)
        p = r + beta * p
        rs_old = rs_new
    solution = x * mask + boundary_values * known
    final_residual = _residual_norm(solution, rhs_w, mask_unknown, config.h)
    if not history or history[-1] != final_residual:
        history.append(final_residual)
    return _finish(solution, history, start, config.tol, num_iter, {"system": "-Delta I = -w"})


def solve_poisson_dense_reference(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    h: float,
) -> SolverResult:
    """Dense float64 CPU reference solve for small grids only."""

    _validate_poisson_inputs(rhs_w, boundary_values, mask_unknown)
    start = time.time()
    rhs_cpu = rhs_w.detach().to(device="cpu", dtype=torch.float64)
    boundary_cpu = boundary_values.detach().to(device="cpu", dtype=torch.float64)
    mask_cpu = mask_unknown.detach().to(device="cpu").bool()
    batch, _, height, width = rhs_cpu.shape
    solution = boundary_cpu.clone()
    unknown_counts: List[int] = []
    for b in range(batch):
        coords = torch.nonzero(mask_cpu[b, 0], as_tuple=False)
        n = int(coords.shape[0])
        unknown_counts.append(n)
        if n == 0:
            continue
        index = {(int(y.item()), int(x.item())): i for i, (y, x) in enumerate(coords)}
        a = torch.zeros((n, n), dtype=torch.float64)
        bb = torch.zeros((n,), dtype=torch.float64)
        for row, coord in enumerate(coords):
            y = int(coord[0].item())
            x = int(coord[1].item())
            a[row, row] += -4.0 / (h * h)
            bb[row] = rhs_cpu[b, 0, y, x]
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    a[row, row] += 1.0 / (h * h)
                elif mask_cpu[b, 0, yy, xx]:
                    a[row, index[(yy, xx)]] += 1.0 / (h * h)
                else:
                    bb[row] -= boundary_cpu[b, 0, yy, xx] / (h * h)
        vals = torch.linalg.solve(a, bb)
        for row, coord in enumerate(coords):
            solution[b, 0, int(coord[0].item()), int(coord[1].item())] = vals[row]
    solution = solution.to(device=rhs_w.device, dtype=rhs_w.dtype)
    residual = _residual_norm(solution, rhs_w, mask_unknown, h)
    return SolverResult(
        solution=solution,
        residual_history=[residual],
        num_iter=1,
        converged=True,
        runtime_sec=time.time() - start,
        diagnostics={"reference": "dense_cpu_float64", "unknown_counts": unknown_counts},
    )


def solve_poisson(
    rhs_w: torch.Tensor,
    boundary_values: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: PoissonSolverConfig,
    initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Dispatch masked Poisson reconstruction by ``config.method``."""

    if config.method == "jacobi":
        return solve_poisson_jacobi(rhs_w, boundary_values, mask_unknown, config, initial)
    if config.method == "gs_rb":
        return solve_poisson_gauss_seidel_red_black(rhs_w, boundary_values, mask_unknown, config, initial)
    if config.method == "sor_rb":
        return solve_poisson_sor_red_black(rhs_w, boundary_values, mask_unknown, config, initial)
    if config.method == "cg":
        return solve_poisson_cg(rhs_w, boundary_values, mask_unknown, config, initial)
    if config.method == "dense_reference":
        return solve_poisson_dense_reference(rhs_w, boundary_values, mask_unknown, config.h)
    raise ValueError(f"Unsupported Poisson method {config.method!r}.")
