"""Reusable metrics and output helpers for BCNS validation."""

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask.to(device=like.device, dtype=like.dtype)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Return relative L2 error over masked entries."""

    mask_f = _mask(mask, pred)
    numerator = torch.sqrt((((pred - target) * mask_f) ** 2).sum())
    denominator = torch.sqrt(((target * mask_f) ** 2).sum()).clamp_min(torch.finfo(pred.dtype).eps)
    return float((numerator / denominator).item())


def max_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Return maximum absolute error over masked entries."""

    mask_bool = mask.to(device=pred.device).bool()
    if not bool(mask_bool.any().item()):
        return 0.0
    return float((pred - target).abs()[mask_bool].max().item())


def poisson_residual_norm(solution: torch.Tensor, rhs_w: torch.Tensor, mask_unknown: torch.Tensor, h: float) -> float:
    """Return RMS masked Poisson residual norm."""

    from .operators import masked_l2_norm
    from .poisson import poisson_residual

    return float(masked_l2_norm(poisson_residual(solution, rhs_w, mask_unknown, h), mask_unknown).item())


def ns_rhs_norm(rhs: torch.Tensor, mask: torch.Tensor) -> float:
    """Return RMS norm of a Navier-Stokes RHS tensor."""

    from .operators import masked_l2_norm

    return float(masked_l2_norm(rhs, mask).item())


def save_history_csv(history: List[Dict], path: str) -> None:
    """Save a list of flat diagnostic dictionaries to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in history for key in row.keys()})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_convergence_curves(
    curves: Dict[str, List[float]],
    output_path: str,
    title: str,
    ylabel: str,
) -> None:
    """Save convergence curves to disk without opening an interactive window."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in curves.items():
        ax.plot(range(len(values)), values, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output), dpi=150)
    plt.close(fig)
