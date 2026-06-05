#!/usr/bin/env python
"""Validate BCNS masked Poisson solvers on manufactured fields."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.config import PoissonSolverConfig
from bcns.diagnostics import max_abs_error, plot_convergence_curves, relative_l2_error
from bcns.manufactured import make_poisson_manufactured_solution
from bcns.masks import make_center_box_mask, make_disconnected_box_mask, make_thin_scratch_mask
from bcns.poisson import solve_poisson


def _dtype(name):
    return torch.float64 if name == "float64" else torch.float32


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _run_solver(method, rhs, boundary, mask, h, omega=1.7, max_iter=2000):
    config = PoissonSolverConfig(method=method, h=h, tol=1e-8, max_iter=max_iter, omega=omega)
    return solve_poisson(rhs, boundary, mask, config)


def experiment_center(device, dtype, output_dir):
    rows = []
    curves = {}
    example = None
    for size in (32, 64, 128):
        h = 1.0 / (size - 1)
        intensity, rhs = make_poisson_manufactured_solution(size, h, device=device, dtype=dtype)
        box = max(4, size // 4)
        mask = make_center_box_mask(size, size, box, box, device=device, dtype=dtype)
        methods = ["jacobi", "gs_rb", "sor_rb", "cg"]
        if size == 32:
            methods = ["dense_reference"] + methods
        for method in methods:
            max_iter = 20000 if method == "jacobi" else (10000 if method == "gs_rb" else 1500)
            result = _run_solver(method, rhs, intensity, mask, h, max_iter=max_iter)
            rows.append(
                {
                    "experiment": "center_box",
                    "size": size,
                    "method": method,
                    "iterations": result.num_iter,
                    "final_residual": result.residual_history[-1],
                    "relative_l2_error": relative_l2_error(result.solution, intensity, mask),
                    "max_abs_error": max_abs_error(result.solution, intensity, mask),
                    "runtime_sec": result.runtime_sec,
                    "converged": result.converged,
                }
            )
            if size in (32, 64) and method in {"jacobi", "gs_rb", "sor_rb", "cg"}:
                curves.setdefault(size, {})[method] = result.residual_history
            if size == 64 and method == "cg":
                example = (intensity.detach().cpu(), result.solution.detach().cpu(), mask.detach().cpu())
    for size, curve in curves.items():
        plot_convergence_curves(
            curve,
            str(output_dir / f"poisson_convergence_center_{size}.png"),
            f"Poisson convergence, center mask {size}",
            "RMS residual",
        )
    return rows, example


def experiment_sor_sweep(device, dtype, output_dir):
    size = 64
    h = 1.0 / (size - 1)
    intensity, rhs = make_poisson_manufactured_solution(size, h, device=device, dtype=dtype)
    mask = make_center_box_mask(size, size, 16, 16, device=device, dtype=dtype)
    rows = []
    for omega in (1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9, 1.95):
        result = _run_solver("sor_rb", rhs, intensity, mask, h, omega=omega, max_iter=1500)
        rows.append(
            {
                "omega": omega,
                "iterations": result.num_iter,
                "final_residual": result.residual_history[-1],
                "relative_l2_error": relative_l2_error(result.solution, intensity, mask),
                "runtime_sec": result.runtime_sec,
                "converged": result.converged,
            }
        )
    _write_csv(rows, output_dir / "sor_omega_sweep.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["omega"] for r in rows], [r["iterations"] for r in rows], marker="o")
    ax.set_xlabel("omega")
    ax.set_ylabel("iterations")
    ax.set_title("Red-black SOR omega sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sor_omega_sweep.png", dpi=150)
    plt.close(fig)


def experiment_irregular(device, dtype):
    rows = []
    size = 64
    h = 1.0 / (size - 1)
    intensity, rhs = make_poisson_manufactured_solution(size, h, device=device, dtype=dtype)
    masks = {
        "thin_scratch": make_thin_scratch_mask(size, size, thickness=2, device=device, dtype=dtype),
        "disconnected_boxes": make_disconnected_box_mask(size, size, device=device, dtype=dtype),
    }
    for mask_name, mask in masks.items():
        for method in ("sor_rb", "cg"):
            result = _run_solver(method, rhs, intensity, mask, h, max_iter=1500)
            rows.append(
                {
                    "mask": mask_name,
                    "method": method,
                    "iterations": result.num_iter,
                    "final_residual": result.residual_history[-1],
                    "relative_l2_error": relative_l2_error(result.solution, intensity, mask),
                    "runtime_sec": result.runtime_sec,
                    "converged": result.converged,
                }
            )
    return rows


def save_examples(example, output_dir):
    if example is None:
        return
    target, recon, mask = example
    err = (recon - target).abs() * mask
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, image, title in (
        (axes[0], target[0, 0], "target"),
        (axes[1], recon[0, 0], "reconstruction"),
        (axes[2], mask[0, 0], "mask"),
        (axes[3], err[0, 0], "abs error"),
    ):
        ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "visual_reconstruction_examples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-dir", default="results/bcns_step1/poisson")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center_rows, example = experiment_center(device, dtype, output_dir)
    _write_csv(center_rows, output_dir / "poisson_summary.csv")
    experiment_sor_sweep(device, dtype, output_dir)
    irregular_rows = experiment_irregular(device, dtype)
    _write_csv(irregular_rows, output_dir / "irregular_mask_summary.csv")
    save_examples(example, output_dir)
    print(f"Wrote Poisson validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
