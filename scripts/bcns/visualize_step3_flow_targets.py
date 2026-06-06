#!/usr/bin/env python
"""Visualize BCNS Step 3 finite flow structure targets."""

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.flow_targets import build_flow_structure_target
from bcns.manufactured import make_crossing_line_structure, make_curved_edge_structure
from bcns.masks import make_center_box_mask, make_thin_scratch_mask
from bcns.poisson_targets import build_poisson_structure_target
from bcns.proximal import normalized_known_luminance_smooth, structure_image
from bcns.visualization import make_contact_sheet, save_heatmap_uint8, save_mask_image, save_tensor_image


def _rgb_from_scalar(scalar):
    return scalar.repeat(1, 3, 1, 1)


def _smooth_gradient(size, dtype):
    coords = torch.linspace(-1.0, 1.0, size, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    scalar = 0.55 * xx + 0.35 * yy + 0.10 * torch.sin(3.0 * xx)
    return scalar.clamp(-1.0, 1.0).view(1, 1, size, size)


def _structures(size, dtype):
    return [
        ("smooth_gradient", _smooth_gradient(size, dtype)),
        ("crossing_line", 2.0 * make_crossing_line_structure(size=size, dtype=dtype) - 1.0),
        ("curved_edge", 2.0 * make_curved_edge_structure(size=size, dtype=dtype) - 1.0),
    ]


def _masks(size, dtype):
    return [
        ("center_box", 1.0 - make_center_box_mask(size, size, size // 3, size // 3, dtype=dtype)),
        ("thin_scratch", 1.0 - make_thin_scratch_mask(size, size, thickness=max(1, size // 24), dtype=dtype)),
    ]


def _write_diagnostics(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_case(output_dir, case_name, mu, mask_known):
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    measurement = mu * mask_known
    size = (int(mu.shape[-1]), int(mu.shape[-2]))
    mu_structure = structure_image(mu, sigma=1.0)
    normalized = normalized_known_luminance_smooth(measurement, mask_known, sigma=1.0)
    harmonic = build_poisson_structure_target(
        mu,
        measurement,
        mask_known,
        sigma=1.0,
        rhs_mode="zero",
        poisson_method="sor_rb",
        poisson_max_iter=300,
        poisson_tol=1e-5,
    )
    poisson = build_poisson_structure_target(
        mu,
        measurement,
        mask_known,
        sigma=1.0,
        rhs_mode="projected_mu_laplacian",
        poisson_method="sor_rb",
        poisson_max_iter=300,
        poisson_tol=1e-5,
    )
    flows = {
        "ftcs": build_flow_structure_target(
            mu,
            measurement,
            mask_known,
            integrator="ftcs",
            poisson_method="sor_rb",
            poisson_max_iter=100,
            poisson_tol=1e-4,
            dt=1e-4,
            pseudo_time=5e-4,
            check_cfl=True,
        ),
        "be": build_flow_structure_target(
            mu,
            measurement,
            mask_known,
            integrator="imex_be",
            poisson_method="sor_rb",
            poisson_max_iter=100,
            poisson_tol=1e-4,
            dt=1e-3,
            pseudo_time=3e-3,
        ),
        "cn": build_flow_structure_target(
            mu,
            measurement,
            mask_known,
            integrator="imex_cn",
            poisson_method="sor_rb",
            poisson_max_iter=100,
            poisson_tol=1e-4,
            dt=1e-3,
            pseudo_time=3e-3,
        ),
    }

    paths = {
        "mu": case_dir / "mu.png",
        "measurement": case_dir / "measurement.png",
        "mask": case_dir / "mask.png",
        "initial": case_dir / "initial_intensity.png",
        "boundary": case_dir / "boundary_intensity.png",
        "normalized": case_dir / "normalized_known_smooth.png",
        "harmonic": case_dir / "harmonic_structure.png",
        "poisson": case_dir / "poisson_structure.png",
        "flow_ftcs": case_dir / "flow_structure_ftcs.png",
        "flow_be": case_dir / "flow_structure_be.png",
        "flow_cn": case_dir / "flow_structure_cn.png",
        "diff_harmonic": case_dir / "harmonic_abs_diff.png",
        "diff_poisson": case_dir / "poisson_abs_diff.png",
        "diff_ftcs": case_dir / "flow_ftcs_abs_diff.png",
        "diff_be": case_dir / "flow_be_abs_diff.png",
        "diff_cn": case_dir / "flow_cn_abs_diff.png",
    }
    save_tensor_image(mu, paths["mu"])
    save_tensor_image(measurement, paths["measurement"])
    save_mask_image(mask_known, paths["mask"])
    save_tensor_image(flows["be"].initial_intensity, paths["initial"])
    save_tensor_image(flows["be"].boundary_intensity, paths["boundary"])
    save_tensor_image(normalized, paths["normalized"])
    save_tensor_image(harmonic.structure, paths["harmonic"])
    save_tensor_image(poisson.structure, paths["poisson"])
    save_tensor_image(flows["ftcs"].structure, paths["flow_ftcs"])
    save_tensor_image(flows["be"].structure, paths["flow_be"])
    save_tensor_image(flows["cn"].structure, paths["flow_cn"])
    save_heatmap_uint8((harmonic.structure - mu_structure).abs(), paths["diff_harmonic"], target_size=size)
    save_heatmap_uint8((poisson.structure - mu_structure).abs(), paths["diff_poisson"], target_size=size)
    save_heatmap_uint8((flows["ftcs"].structure - mu_structure).abs(), paths["diff_ftcs"], target_size=size)
    save_heatmap_uint8((flows["be"].structure - mu_structure).abs(), paths["diff_be"], target_size=size)
    save_heatmap_uint8((flows["cn"].structure - mu_structure).abs(), paths["diff_cn"], target_size=size)

    ordered_keys = [
        "mu",
        "measurement",
        "mask",
        "initial",
        "boundary",
        "normalized",
        "harmonic",
        "poisson",
        "flow_ftcs",
        "flow_be",
        "flow_cn",
        "diff_harmonic",
        "diff_poisson",
        "diff_ftcs",
        "diff_be",
        "diff_cn",
    ]
    make_contact_sheet(
        [paths[key] for key in ordered_keys],
        ordered_keys,
        case_dir / "contact_sheet.png",
        cols=8,
    )

    rows = [
        {"target": "harmonic", **harmonic.diagnostics},
        {"target": "poisson", **poisson.diagnostics},
    ]
    for name, result in flows.items():
        rows.append({"target": f"flow_{name}", **result.diagnostics})
    _write_diagnostics(case_dir / "diagnostics.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/bcns_step3/flow_target_visualization")
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
    print(f"Wrote Step 3 flow target visualizations to {output_dir}")


if __name__ == "__main__":
    main()
