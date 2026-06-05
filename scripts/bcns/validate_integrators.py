#!/usr/bin/env python
"""Validate BCNS pseudo-time integrators."""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.config import FlowConfig, PoissonSolverConfig
from bcns.flow import evolve_bcns_flow
from bcns.integrators import ftcs_step, imex_backward_euler_step, imex_crank_nicolson_step
from bcns.manufactured import (
    make_crossing_line_structure,
    make_curved_edge_structure,
    make_text_like_structure,
)
from bcns.masks import make_center_box_mask, make_thin_scratch_mask
from bcns.operators import laplacian_5pt


def _dtype(name):
    return torch.float64 if name == "float64" else torch.float32


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _flow_config(name, dt, h, check_cfl=True, poisson_method="cg"):
    return FlowConfig(
        integrator=name,
        poisson=PoissonSolverConfig(method=poisson_method, h=h, tol=1e-7, max_iter=500, omega=1.7),
        dt=dt,
        pseudo_time=0.01,
        nu=0.1,
        kappa=1e12,
        smoothing_sigma=0.0,
        check_cfl=check_cfl,
        implicit_max_iter=300,
        implicit_tol=1e-9,
    )


def pure_diffusion(device, dtype, output_dir):
    size = 40
    h = 1.0
    y = torch.arange(size, device=device, dtype=dtype).view(size, 1) + 0.5
    x = torch.arange(size, device=device, dtype=dtype).view(1, size) + 0.5
    w0 = (torch.cos(math.pi * x / size) * torch.cos(math.pi * y / size)).view(1, 1, size, size)
    intensity = torch.zeros_like(w0)
    mask = torch.ones_like(w0)
    nu = 0.1
    tau = 0.01
    lam = (8.0 / (h * h)) * (math.sin(math.pi / (2.0 * size)) ** 2)
    exact = math.exp(-nu * lam * tau) * w0
    rows = []
    for name, fn in (
        ("ftcs", ftcs_step),
        ("imex_be", imex_backward_euler_step),
        ("imex_cn", imex_crank_nicolson_step),
    ):
        for dt in (0.005, 0.0025, 0.00125):
            cfg = _flow_config(name, dt, h, check_cfl=True)
            cfg.pseudo_time = tau
            cfg.nu = nu
            w = w0
            steps = int(round(tau / dt))
            for _ in range(steps):
                w = fn(intensity, w, mask, None, cfg).vorticity
            err = float(torch.linalg.norm((w - exact).reshape(-1)) / torch.linalg.norm(exact.reshape(-1)))
            rows.append({"integrator": name, "dt": dt, "relative_l2_error": err, "runtime_sec": 0.0})
    _write_csv(rows, output_dir / "pure_diffusion_accuracy.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("ftcs", "imex_be", "imex_cn"):
        xs = [r["dt"] for r in rows if r["integrator"] == name]
        ys = [r["relative_l2_error"] for r in rows if r["integrator"] == name]
        ax.loglog(xs, ys, marker="o", label=name)
    ax.invert_xaxis()
    ax.set_xlabel("dt")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Pure diffusion temporal accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pure_diffusion_accuracy.png", dpi=150)
    plt.close(fig)


def ftcs_stability(device, dtype, output_dir):
    size = 32
    h = 1.0
    intensity = torch.zeros((1, 1, size, size), device=device, dtype=dtype)
    vorticity = torch.randn_like(intensity)
    mask = torch.ones_like(intensity)
    rows = []
    stable_cfg = _flow_config("ftcs", 0.05, h, check_cfl=True)
    unstable_cfg = _flow_config("ftcs", 10.0, h, check_cfl=True)
    ftcs_step(intensity, vorticity, mask, None, stable_cfg)
    raised = False
    try:
        ftcs_step(intensity, vorticity, mask, None, unstable_cfg)
    except ValueError:
        raised = True
    rows.append({"case": "cfl_error_check", "raised": raised})
    no_check = _flow_config("ftcs", 1.0, h, check_cfl=False)
    norms = []
    w = vorticity
    for step in range(10):
        w = ftcs_step(intensity, w, mask, None, no_check).vorticity
        norm = float(torch.linalg.norm(w.reshape(-1)).item())
        norms.append(norm)
        rows.append({"case": "no_check_norm_growth", "step": step + 1, "norm": norm})
    _write_csv(rows, output_dir / "ftcs_stability.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(norms) + 1), norms, marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("||w||")
    ax.set_title("FTCS no-check diagnostic norm growth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ftcs_stability.png", dpi=150)
    plt.close(fig)


def full_flow_smoke(device, dtype, output_dir):
    size = 48
    h = 1.0
    patterns = {
        "crossing_lines": make_crossing_line_structure(size, device=device, dtype=dtype),
        "curved_edge": make_curved_edge_structure(size, device=device, dtype=dtype),
        "text_like": make_text_like_structure(size, device=device, dtype=dtype),
    }
    masks = {
        "crossing_lines": make_center_box_mask(size, size, 14, 14, device=device, dtype=dtype),
        "curved_edge": make_center_box_mask(size, size, 14, 14, device=device, dtype=dtype),
        "text_like": make_thin_scratch_mask(size, size, thickness=2, device=device, dtype=dtype),
    }
    rows = []
    examples = []
    for pattern_name, image in patterns.items():
        mask = masks[pattern_name]
        boundary_intensity = image * (1.0 - mask)
        boundary_vorticity = laplacian_5pt(image, h) * (1.0 - mask)
        initial = boundary_intensity.clone()
        for poisson_method in ("cg", "sor_rb"):
            for integrator in ("ftcs", "imex_be", "imex_cn"):
                cfg = _flow_config(integrator, 0.001, h, check_cfl=False, poisson_method=poisson_method)
                cfg.pseudo_time = 0.003
                cfg.kappa = 0.5
                cfg.smoothing_sigma = 1.0
                result = evolve_bcns_flow(initial, boundary_intensity, boundary_vorticity, mask, cfg)
                finite = bool(torch.isfinite(result.intensity).all().item() and torch.isfinite(result.vorticity).all().item())
                rows.append(
                    {
                        "pattern": pattern_name,
                        "poisson_solver": poisson_method,
                        "integrator": integrator,
                        "runtime_sec": result.runtime_sec,
                        "finite": finite,
                        "final_poisson_residual": result.history[-1]["poisson_residual"] if result.history else float("nan"),
                        "final_rhs_norm": result.history[-1].get("rhs_norm", float("nan")) if result.history else float("nan"),
                    }
                )
                if poisson_method == "cg" and integrator == "imex_cn":
                    examples.append((pattern_name, result.intensity.detach().cpu(), mask.detach().cpu()))
    _write_csv(rows, output_dir / "full_flow_summary.csv")
    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 3 * max(1, len(examples))))
    if len(examples) == 1:
        axes = axes.reshape(1, 2)
    for row, (name, image, mask) in enumerate(examples):
        axes[row, 0].imshow(image[0, 0], cmap="gray")
        axes[row, 0].set_title(name)
        axes[row, 1].imshow(mask[0, 0], cmap="gray")
        axes[row, 1].set_title("mask")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "full_flow_examples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-dir", default="results/bcns_step1/integrators")
    args = parser.parse_args()
    torch.manual_seed(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    pure_diffusion(device, dtype, output_dir)
    ftcs_stability(device, dtype, output_dir)
    full_flow_smoke(device, dtype, output_dir)
    print(f"Wrote integrator validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
