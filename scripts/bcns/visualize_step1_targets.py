#!/usr/bin/env python
"""Visualize BCNS Step 1 target builders without diffusion checkpoints."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.dps_adapter import hard_project_clean
from bcns.masks import make_center_box_mask, make_thin_scratch_mask
from bcns.proximal import structure_image
from bcns.target_builders import get_target_builder
from bcns.visualization import make_contact_sheet, save_heatmap, save_mask_image, save_tensor_image


def _synthetic_rgb(size: int = 96) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    red = xx
    green = yy
    blue = torch.sin(3.0 * xx) * torch.cos(4.0 * yy)
    return torch.stack([red, green, blue], dim=0).unsqueeze(0).clamp(-1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/bcns_step1/target_visualization")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mu = _synthetic_rgb()
    center_unknown = make_center_box_mask(96, 96, 30, 30, dtype=mu.dtype)
    scratch_unknown = make_thin_scratch_mask(96, 96, thickness=3, dtype=mu.dtype)
    mask_known = 1.0 - center_unknown
    measurement = mu * mask_known

    saved_paths = []
    labels = []
    for name, tensor in (("mu", mu), ("measurement", measurement)):
        path = output_dir / f"{name}.png"
        save_tensor_image(tensor, path)
        saved_paths.append(path)
        labels.append(name)
    mask_path = output_dir / "mask.png"
    save_mask_image(mask_known, mask_path)
    saved_paths.append(mask_path)
    labels.append("mask")
    save_mask_image(1.0 - scratch_unknown, output_dir / "mask_scratch.png")

    save_heatmap(structure_image(mu, sigma=1.0), output_dir / "structure_mu.png")
    builders = {
        "identity": get_target_builder("identity"),
        "hard_projection": get_target_builder("hard_projection"),
        "simple_hole_shift": get_target_builder("simple_hole_shift", shift=0.12),
        "structure_prox": get_target_builder(
            "structure_prox",
            lambda_structure=0.05,
            structure_sigma=1.0,
            prox_steps=3,
            prox_step_size=0.1,
            target_mode="measurement_only_smooth",
        ),
    }
    for name, builder in builders.items():
        result = builder(mu=mu, measurement=measurement, mask_known=mask_known)
        target = result.target
        target_path = output_dir / f"target_{name}.png"
        diff_path = output_dir / f"abs_diff_{name}.png"
        structure_path = output_dir / f"structure_target_{name}.png"
        save_tensor_image(target, target_path)
        save_heatmap((target - mu).abs().mean(dim=1, keepdim=True), diff_path)
        save_heatmap(structure_image(target, sigma=1.0), structure_path)
        saved_paths.extend([target_path, diff_path])
        labels.extend([f"target {name}", f"diff {name}"])

    make_contact_sheet(saved_paths, labels, output_dir / "contact_sheet.png", cols=4)
    print(f"Wrote target visualizations to {output_dir}")


if __name__ == "__main__":
    main()
