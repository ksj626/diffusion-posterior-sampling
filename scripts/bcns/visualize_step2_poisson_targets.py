#!/usr/bin/env python
"""Visualize BCNS Step 2 harmonic and Poisson structure targets."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.manufactured import make_crossing_line_structure, make_curved_edge_structure
from bcns.masks import make_center_box_mask, make_thin_scratch_mask
from bcns.poisson_targets import build_poisson_structure_target
from bcns.proximal import structure_image, structure_proximal_target_from_structure
from bcns.visualization import make_contact_sheet, save_heatmap_uint8, save_mask_image, save_tensor_image


def _rgb_from_scalar(scalar: torch.Tensor) -> torch.Tensor:
    return scalar.repeat(1, 3, 1, 1)


def _smooth_gradient(size: int, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    scalar = 0.55 * xx + 0.35 * yy + 0.10 * torch.sin(3.0 * xx)
    return scalar.clamp(-1.0, 1.0).view(1, 1, size, size)


def _structures(size: int, dtype: torch.dtype):
    return [
        ("smooth_gradient", _smooth_gradient(size, dtype)),
        ("crossing_line", 2.0 * make_crossing_line_structure(size=size, dtype=dtype) - 1.0),
        ("curved_edge", 2.0 * make_curved_edge_structure(size=size, dtype=dtype) - 1.0),
    ]


def _masks(size: int, dtype: torch.dtype):
    return [
        ("center_box", 1.0 - make_center_box_mask(size, size, size // 3, size // 3, dtype=dtype)),
        ("thin_scratch", 1.0 - make_thin_scratch_mask(size, size, thickness=max(1, size // 24), dtype=dtype)),
    ]


def _save_case(output_dir: Path, case_name: str, mu: torch.Tensor, mask_known: torch.Tensor) -> None:
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    measurement = mu * mask_known
    size = (int(mu.shape[-1]), int(mu.shape[-2]))

    harmonic = build_poisson_structure_target(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        sigma=1.0,
        rhs_mode="zero",
        poisson_method="sor_rb",
        poisson_max_iter=500,
        poisson_tol=1e-5,
    )
    poisson = build_poisson_structure_target(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        sigma=1.0,
        rhs_mode="mu_laplacian",
        poisson_method="sor_rb",
        poisson_max_iter=500,
        poisson_tol=1e-5,
    )
    harmonic_rgb = structure_proximal_target_from_structure(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        target_structure=harmonic.structure.to(dtype=mu.dtype),
        tau2=1.0,
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=1,
        prox_step_size=0.05,
    ).target
    poisson_rgb = structure_proximal_target_from_structure(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        target_structure=poisson.structure.to(dtype=mu.dtype),
        tau2=1.0,
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=1,
        prox_step_size=0.05,
    ).target
    mu_structure = structure_image(mu, sigma=1.0)

    paths = {
        "mu": case_dir / "mu.png",
        "measurement": case_dir / "measurement.png",
        "mask": case_dir / "mask.png",
        "boundary": case_dir / "boundary_values.png",
        "structure_mu": case_dir / "structure_mu.png",
        "harmonic": case_dir / "harmonic_structure.png",
        "poisson": case_dir / "poisson_mu_laplacian_structure.png",
        "harmonic_diff": case_dir / "harmonic_abs_structure_diff.png",
        "poisson_diff": case_dir / "poisson_abs_structure_diff.png",
        "harmonic_rgb": case_dir / "harmonic_rgb_proximal_target.png",
        "poisson_rgb": case_dir / "poisson_rgb_proximal_target.png",
    }
    save_tensor_image(mu, paths["mu"])
    save_tensor_image(measurement, paths["measurement"])
    save_mask_image(mask_known, paths["mask"])
    save_tensor_image(harmonic.boundary_values, paths["boundary"])
    save_tensor_image(mu_structure, paths["structure_mu"])
    save_tensor_image(harmonic.structure, paths["harmonic"])
    save_tensor_image(poisson.structure, paths["poisson"])
    save_heatmap_uint8((harmonic.structure - mu_structure).abs(), paths["harmonic_diff"], target_size=size)
    save_heatmap_uint8((poisson.structure - mu_structure).abs(), paths["poisson_diff"], target_size=size)
    save_tensor_image(harmonic_rgb, paths["harmonic_rgb"])
    save_tensor_image(poisson_rgb, paths["poisson_rgb"])

    ordered = [
        paths["mu"],
        paths["measurement"],
        paths["mask"],
        paths["boundary"],
        paths["structure_mu"],
        paths["harmonic"],
        paths["poisson"],
        paths["harmonic_diff"],
        paths["poisson_diff"],
        paths["harmonic_rgb"],
        paths["poisson_rgb"],
    ]
    labels = [
        "mu",
        "measurement",
        "mask",
        "boundary",
        "S(mu)",
        "harmonic",
        "poisson",
        "|harm-S|",
        "|pois-S|",
        "harm RGB",
        "poisson RGB",
    ]
    make_contact_sheet(ordered, labels, case_dir / "contact_sheet.png", cols=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/bcns_step2/poisson_target_visualization")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_paths = []
    all_labels = []
    for structure_name, scalar in _structures(args.size, dtype):
        mu = _rgb_from_scalar(scalar)
        for mask_name, mask_known in _masks(args.size, dtype):
            case_name = f"{structure_name}_{mask_name}"
            _save_case(output_dir, case_name, mu, mask_known)
            all_paths.append(output_dir / case_name / "contact_sheet.png")
            all_labels.append(case_name)
    make_contact_sheet(all_paths, all_labels, output_dir / "summary_contact_sheet.png", cols=2)
    print(f"Wrote Step 2 target visualizations to {output_dir}")


if __name__ == "__main__":
    main()
