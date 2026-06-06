#!/usr/bin/env python
"""Visualize Step 4 structural diagnostics for one saved inpainting result."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.navier_stokes import ns_rhs
from bcns.operators import gradient_central, laplacian_5pt
from bcns.structural_metrics import boundary_band, luminance_structure
from bcns.visualization import make_contact_sheet, save_heatmap_uint8, save_mask_image, save_tensor_image, to_uint8_image


def _load_dps_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _load_mask(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    arr = (np.asarray(image).astype("float32") / 255.0 > 0.5).astype("float32")
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _resolve_method_dir(result_dir: Path, method: str) -> Path:
    if (result_dir / "recon_raw.png").exists():
        return result_dir
    if method:
        candidate = result_dir / method
        if not candidate.exists():
            raise FileNotFoundError(f"Method directory not found: {candidate}")
        return candidate
    for preferred in ("bcns_flow_be_projected", "bcns_flow_cn_projected", "bcns_poisson_sor_projected"):
        candidate = result_dir / preferred
        if (candidate / "recon_raw.png").exists():
            return candidate
    for candidate in sorted(result_dir.iterdir()):
        if candidate.is_dir() and (candidate / "recon_raw.png").exists():
            return candidate
    raise FileNotFoundError(f"No saved method result found under {result_dir}")


def _gradient_magnitude(x: torch.Tensor, sigma: float, h: float) -> torch.Tensor:
    structure = luminance_structure(x, sigma=sigma)
    gx, gy = gradient_central(structure, h)
    return torch.sqrt(gx * gx + gy * gy)


def _ns_residual_map(x: torch.Tensor, mask_known: torch.Tensor, sigma: float, h: float) -> torch.Tensor:
    structure = luminance_structure(x, sigma=sigma)
    vorticity = laplacian_5pt(structure, h)
    rhs, _ = ns_rhs(structure, vorticity, nu=0.1, kappa=0.1, smoothing_sigma=1.0, h=h)
    return rhs.abs() * (1.0 - mask_known)


def _save_isophote_overlay(x: torch.Tensor, mask_known: torch.Tensor, path: Path, sigma: float, h: float) -> None:
    structure = luminance_structure(x, sigma=sigma)
    gx, gy = gradient_central(structure, h)
    vx, vy = -gy[0, 0], gx[0, 0]
    band = boundary_band(mask_known, width=3)["band"][0, 0] > 0.5
    image = Image.fromarray(to_uint8_image(x))
    draw = ImageDraw.Draw(image)
    step = max(8, int(min(image.size) // 24))
    length = max(4.0, float(step) * 0.45)
    height, width = band.shape
    for y in range(step // 2, height, step):
        for x_idx in range(step // 2, width, step):
            if not bool(band[y, x_idx].item()):
                continue
            vx_value = float(vx[y, x_idx].item())
            vy_value = float(vy[y, x_idx].item())
            norm = (vx_value * vx_value + vy_value * vy_value) ** 0.5
            if norm <= 1e-8:
                continue
            dx = length * vx_value / norm
            dy = length * vy_value / norm
            draw.line(
                [(x_idx - dx, y - dy), (x_idx + dx, y + dy)],
                fill=(0, 255, 0),
                width=1,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    args = parser.parse_args()

    method_dir = _resolve_method_dir(Path(args.result_dir), args.method)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = _load_dps_image(method_dir / "label.png")
    recon = _load_dps_image(method_dir / "recon_composite.png")
    raw = _load_dps_image(method_dir / "recon_raw.png")
    mask = _load_mask(method_dir / "mask.png")
    size = (int(label.shape[-1]), int(label.shape[-2]))

    band = boundary_band(mask, width=3)["band"]
    label_grad = _gradient_magnitude(label, args.sigma, args.h)
    recon_grad = _gradient_magnitude(recon, args.sigma, args.h)
    lap_map = laplacian_5pt(luminance_structure(recon, args.sigma), args.h).abs()
    residual = _ns_residual_map(recon, mask, args.sigma, args.h)

    outputs = {
        "label": output_dir / "label.png",
        "recon_raw": output_dir / "recon_raw.png",
        "recon_composite": output_dir / "recon_composite.png",
        "boundary_band": output_dir / "boundary_band.png",
        "label_grad": output_dir / "label_gradient_magnitude.png",
        "recon_grad": output_dir / "recon_gradient_magnitude.png",
        "laplacian": output_dir / "recon_laplacian_abs.png",
        "ns_residual": output_dir / "recon_ns_residual.png",
        "isophote": output_dir / "isophote_overlay.png",
    }
    save_tensor_image(label, outputs["label"])
    save_tensor_image(raw, outputs["recon_raw"])
    save_tensor_image(recon, outputs["recon_composite"])
    save_mask_image(band, outputs["boundary_band"])
    save_heatmap_uint8(label_grad, outputs["label_grad"], target_size=size)
    save_heatmap_uint8(recon_grad, outputs["recon_grad"], target_size=size)
    save_heatmap_uint8(lap_map, outputs["laplacian"], target_size=size)
    save_heatmap_uint8(residual, outputs["ns_residual"], target_size=size)
    _save_isophote_overlay(recon, mask, outputs["isophote"], args.sigma, args.h)

    make_contact_sheet(
        [
            outputs["label"],
            outputs["recon_raw"],
            outputs["recon_composite"],
            outputs["boundary_band"],
            outputs["label_grad"],
            outputs["recon_grad"],
            outputs["laplacian"],
            outputs["ns_residual"],
            outputs["isophote"],
        ],
        [
            "label",
            "raw",
            "composite",
            "boundary band",
            "label grad",
            "recon grad",
            "laplacian",
            "NS residual",
            "isophote",
        ],
        output_dir / "structural_metrics_contact_sheet.png",
        cols=3,
    )
    print(f"Wrote structural metric visualization for {method_dir} to {output_dir}")


if __name__ == "__main__":
    main()
