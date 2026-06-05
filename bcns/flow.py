"""Coupled vorticity-stream BCNS pseudo-time evolution."""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .config import FlowConfig
from .integrators import (
    ftcs_step,
    imex_backward_euler_step,
    imex_crank_nicolson_step,
)
from .operators import laplacian_5pt
from .poisson import solve_poisson


@dataclass
class FlowResult:
    """Final fields and diagnostic history for BCNS flow evolution."""

    intensity: torch.Tensor
    vorticity: torch.Tensor
    history: List[Dict[str, Any]] = field(default_factory=list)
    runtime_sec: float = 0.0


def evolve_bcns_flow(
    initial_intensity: torch.Tensor,
    boundary_intensity: torch.Tensor,
    boundary_vorticity: Optional[torch.Tensor],
    mask_unknown: torch.Tensor,
    config: FlowConfig,
) -> FlowResult:
    """Evolve vorticity, reconstructing intensity by Poisson at each step."""

    start = time.time()
    if boundary_intensity.shape != initial_intensity.shape or mask_unknown.shape != initial_intensity.shape:
        raise ValueError("initial_intensity, boundary_intensity, and mask_unknown must match.")
    h = config.poisson.h
    mask = mask_unknown.to(dtype=initial_intensity.dtype, device=initial_intensity.device)
    known = 1.0 - mask
    intensity = initial_intensity * mask + boundary_intensity * known
    vorticity = laplacian_5pt(intensity, h)
    if boundary_vorticity is not None and config.vorticity_boundary_mode == "hard":
        vorticity = vorticity * mask + boundary_vorticity * known

    if config.pseudo_time == 0:
        return FlowResult(intensity=intensity, vorticity=vorticity, history=[], runtime_sec=time.time() - start)

    steps = max(1, int(math.ceil(config.pseudo_time / config.dt)))
    history: List[Dict[str, Any]] = []
    for step_idx in range(steps):
        poisson_result = solve_poisson(vorticity, boundary_intensity, mask_unknown, config.poisson, intensity)
        intensity = poisson_result.solution
        if config.integrator == "ftcs":
            step = ftcs_step(intensity, vorticity, mask_unknown, boundary_vorticity, config)
        elif config.integrator == "imex_be":
            step = imex_backward_euler_step(intensity, vorticity, mask_unknown, boundary_vorticity, config)
        elif config.integrator == "imex_cn":
            step = imex_crank_nicolson_step(intensity, vorticity, mask_unknown, boundary_vorticity, config)
        else:
            raise ValueError(f"Unsupported integrator {config.integrator!r}.")
        vorticity = step.vorticity
        record: Dict[str, Any] = {
            "step": step_idx + 1,
            "pseudo_time": min((step_idx + 1) * config.dt, config.pseudo_time),
            "poisson_iter": poisson_result.num_iter,
            "poisson_residual": poisson_result.residual_history[-1] if poisson_result.residual_history else float("nan"),
            "poisson_converged": poisson_result.converged,
        }
        record.update(step.diagnostics)
        history.append(record)

    final_poisson = solve_poisson(vorticity, boundary_intensity, mask_unknown, config.poisson, intensity)
    return FlowResult(
        intensity=final_poisson.solution,
        vorticity=vorticity,
        history=history,
        runtime_sec=time.time() - start,
    )
