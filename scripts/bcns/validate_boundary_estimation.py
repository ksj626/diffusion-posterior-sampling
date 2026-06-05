#!/usr/bin/env python
"""Validate BCNS boundary intensity and vorticity estimation."""

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

from bcns.boundary import estimate_boundary_data, estimate_boundary_vorticity_fd
from bcns.config import BoundaryConfig
from bcns.masks import make_center_box_mask
from bcns.structure import gaussian_smooth, masked_normalized_gaussian_smooth


def _dtype(name):
    return torch.float64 if name == "float64" else torch.float32


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def smoothing_bias(device, dtype, output_dir):
    size = 64
    y = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype).view(size, 1)
    x = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype).view(1, size)
    field = (0.2 + 0.5 * x + 0.3 * y).view(1, 1, size, size)
    mask = make_center_box_mask(size, size, 18, 18, device=device, dtype=dtype)
    known = 1.0 - mask
    zero_masked = field * known
    direct = gaussian_smooth(zero_masked, sigma=2.0)
    normalized = masked_normalized_gaussian_smooth(zero_masked, known, sigma=2.0)
    bias = (normalized - direct).abs() * mask
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, image, title in (
        (axes[0], field.detach().cpu()[0, 0], "field"),
        (axes[1], direct.detach().cpu()[0, 0], "zero-mask conv"),
        (axes[2], normalized.detach().cpu()[0, 0], "normalized"),
        (axes[3], bias.detach().cpu()[0, 0], "boundary bias"),
    ):
        ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "smoothing_bias_comparison.png", dpi=150)
    plt.close(fig)


def quadratic_vorticity(device, dtype, output_dir):
    size = 48
    y = torch.arange(size, device=device, dtype=dtype).view(size, 1)
    x = torch.arange(size, device=device, dtype=dtype).view(1, size)
    a3, a5 = 0.45, -0.15
    field = 0.3 + 0.2 * x - 0.1 * y + a3 * x * x + 0.05 * x * y + a5 * y * y
    image = field.view(1, 1, size, size)
    mask = make_center_box_mask(size, size, 8, 8, device=device, dtype=dtype)
    observed = image * (1.0 - mask)
    expected = 2.0 * a3 + 2.0 * a5
    poly_cfg = BoundaryConfig(
        gaussian_sigma=0.0,
        collar_width=5,
        polynomial_radius=5,
        vorticity_estimator="polynomial",
    )
    fd_cfg = BoundaryConfig(gaussian_sigma=0.0, collar_width=5, vorticity_estimator="fd")
    poly = estimate_boundary_data(observed, mask, poly_cfg)
    fd = estimate_boundary_vorticity_fd(observed, mask, fd_cfg)
    rows = []
    for name, estimate, extra in (
        ("finite_difference", fd, {}),
        ("polynomial", poly.vorticity_trace, poly.diagnostics),
    ):
        err = (estimate - expected).abs()[mask.bool()]
        rows.append(
            {
                "estimator": name,
                "mean_abs_error": float(err.mean().item()),
                "max_abs_error": float(err.max().item()),
                "fallback_count": extra.get("fallback_count", 0),
                "fallback_ratio": extra.get("fallback_ratio", 0.0),
            }
        )
    _write_csv(rows, output_dir / "boundary_vorticity_summary.csv")
    error_map = (poly.vorticity_trace - expected).abs() * mask
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(error_map.detach().cpu()[0, 0], cmap="magma")
    ax.set_title("polynomial vorticity abs error")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_vorticity_error_map.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-dir", default="results/bcns_step1/boundary")
    args = parser.parse_args()
    torch.manual_seed(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoothing_bias(torch.device(args.device), _dtype(args.dtype), output_dir)
    quadratic_vorticity(torch.device(args.device), _dtype(args.dtype), output_dir)
    print(f"Wrote boundary validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
