#!/usr/bin/env python
"""Visualize BCNS Step 4.5 target displacement and structure guidance signals."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.masks import make_center_box_mask, make_thin_scratch_mask
from bcns.proximal import structure_image
from bcns.target_builders import get_target_builder
from bcns.visualization import make_contact_sheet, save_heatmap_uint8, save_mask_image, save_tensor_image


def synthetic_image(size: int, device, dtype):
    coords = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    red = xx
    green = yy
    blue = torch.sin(3.0 * xx) * torch.cos(4.0 * yy)
    image = torch.stack((red, green, blue), dim=0).unsqueeze(0)
    return image.clamp(-1.0, 1.0)


def make_known_mask(mask_mode: str, size: int, device, dtype):
    if mask_mode == "center_box":
        unknown = make_center_box_mask(size, size, size // 3, size // 3, device=device, dtype=dtype)
    elif mask_mode == "thin_scratch":
        unknown = make_thin_scratch_mask(size, size, thickness=max(2, size // 32), device=device, dtype=dtype)
    elif mask_mode == "random":
        generator = torch.Generator()
        generator.manual_seed(123)
        unknown = (torch.rand((1, 1, size, size), generator=generator, device=device, dtype=dtype) < 0.35).to(dtype)
    else:
        raise ValueError(f"Unsupported mask_mode {mask_mode!r}.")
    return 1.0 - unknown


def builder_specs():
    return [
        (
            "structure_prox",
            get_target_builder(
                "structure_prox",
                target_mode="normalized_known_smooth",
                lambda_structure=0.5,
                prox_steps=3,
                prox_step_size=0.1,
            ),
        ),
        (
            "harmonic_structure",
            get_target_builder(
                "harmonic_structure",
                poisson_method="sor_rb",
                poisson_max_iter=80,
                poisson_tol=1e-4,
            ),
        ),
        (
            "poisson_structure",
            get_target_builder(
                "poisson_structure",
                rhs_mode="projected_mu_laplacian",
                poisson_method="sor_rb",
                poisson_max_iter=80,
                poisson_tol=1e-4,
            ),
        ),
        (
            "flow_structure",
            get_target_builder(
                "flow_structure",
                integrator="imex_be",
                pseudo_time=1e-3,
                dt=1e-3,
                poisson_method="sor_rb",
                poisson_max_iter=50,
            ),
        ),
    ]


def save_builder_visuals(output_dir, name, result, mu, measurement, mask_known, sigma):
    method_dir = output_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    target_rgb = result.target.detach()
    s_mu = structure_image(mu, sigma).detach()
    target_structure = result.target_structure
    if target_structure is None:
        target_structure = torch.zeros_like(s_mu)
    else:
        target_structure = target_structure.detach()
    hole = 1.0 - mask_known
    rgb_delta = (target_rgb - mu).abs().mean(dim=1, keepdim=True)
    struct_delta = (target_structure - s_mu).abs()
    size = (int(mu.shape[-1]), int(mu.shape[-2]))

    paths = [
        method_dir / "mu.png",
        method_dir / "measurement.png",
        method_dir / "mask.png",
        method_dir / "target_rgb.png",
        method_dir / "target_structure.png",
        method_dir / "structure_mu.png",
        method_dir / "rgb_delta.png",
        method_dir / "hole_rgb_delta.png",
        method_dir / "structure_delta.png",
        method_dir / "hole_structure_delta.png",
    ]
    save_tensor_image(mu, paths[0])
    save_tensor_image(measurement, paths[1])
    save_mask_image(mask_known, paths[2])
    save_tensor_image(target_rgb, paths[3])
    save_heatmap_uint8(target_structure, paths[4], target_size=size)
    save_heatmap_uint8(s_mu, paths[5], target_size=size)
    save_heatmap_uint8(rgb_delta, paths[6], target_size=size)
    save_heatmap_uint8(rgb_delta * hole, paths[7], target_size=size)
    save_heatmap_uint8(struct_delta, paths[8], target_size=size)
    save_heatmap_uint8(struct_delta * hole, paths[9], target_size=size)
    labels = [
        "mu",
        "measurement",
        "mask",
        "target rgb",
        "target structure",
        "S(mu)",
        "|target-mu|",
        "hole |target-mu|",
        "|I-S(mu)|",
        "hole |I-S(mu)|",
    ]
    make_contact_sheet(paths, labels, method_dir / "contact_sheet.png", cols=5)
    return method_dir / "contact_sheet.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask_mode", choices=("random", "center_box", "thin_scratch"), default="thin_scratch")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--structure_sigma", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cpu")
    dtype = torch.float64
    torch.manual_seed(123)
    label = synthetic_image(args.size, device=device, dtype=dtype)
    mask_known = make_known_mask(args.mask_mode, args.size, device=device, dtype=dtype)
    measurement = label * mask_known
    mu = measurement + (1.0 - mask_known) * torch.randn_like(label).mul(0.35)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    labels = []
    for name, builder in builder_specs():
        result = builder(mu=mu, measurement=measurement, mask_known=mask_known, tau2=1.0)
        sheets.append(save_builder_visuals(output_dir, name, result, mu, measurement, mask_known, args.structure_sigma))
        labels.append(name)
    make_contact_sheet(sheets, labels, output_dir / "guidance_activation_contact_sheet.png", cols=2)
    print(f"Wrote guidance activation visualization to {output_dir}")


if __name__ == "__main__":
    main()
