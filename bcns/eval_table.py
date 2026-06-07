"""Evaluation table helpers for BCNS-DPS inpainting experiments."""

import csv
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from .eval_metrics import (
    hole_focused_lpips,
    lpips_metric,
    lpips_optional,
    mae_regions,
    mse_regions,
    psnr,
    ssim_simple,
)
from .structural_metrics import (
    gradient_mismatch,
    isophote_angle_error,
    laplacian_mismatch,
    ns_residual_metric,
    seam_mse,
)


def _prefixed(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _copy_diagnostics(row: OrderedDict, diagnostics: Optional[dict]) -> None:
    if not diagnostics:
        return
    for key, value in diagnostics.items():
        if key == "method":
            key = "diagnostic_method"
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        row[str(key)] = value


def evaluate_inpainting_result(
    recon_raw: torch.Tensor,
    recon_composite: torch.Tensor,
    label: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    diagnostics: Optional[dict] = None,
    structural_sigma: float = 1.0,
    boundary_width: int = 3,
    require_lpips: bool = False,
    lpips_loss_fn=None,
) -> dict:
    """Return a flat metric dict for one inpainting reconstruction."""

    row = OrderedDict()
    hole = 1.0 - mask_known
    for prefix, recon in (("raw", recon_raw), ("composite", recon_composite)):
        row[f"{prefix}_psnr"] = psnr(recon, label)
        row[f"{prefix}_known_psnr"] = psnr(recon, label, mask_known)
        row[f"{prefix}_hole_psnr"] = psnr(recon, label, hole)
        row[f"{prefix}_ssim"] = ssim_simple(recon, label)
        row[f"{prefix}_hole_ssim"] = ssim_simple(recon, label, hole)
        row.update(_prefixed(prefix, mse_regions(recon, label, mask_known)))
        row.update(_prefixed(prefix, mae_regions(recon, label, mask_known)))
        if require_lpips:
            lpips_value = lpips_metric(recon, label, loss_fn=lpips_loss_fn)
            hole_lpips_value = hole_focused_lpips(recon, label, mask_known, loss_fn=lpips_loss_fn)
        else:
            lpips_value = lpips_optional(recon, label, loss_fn=lpips_loss_fn)
            hole_lpips_value = (
                None
                if lpips_value is None
                else hole_focused_lpips(recon, label, mask_known, loss_fn=lpips_loss_fn)
            )
        row[f"{prefix}_lpips"] = "" if lpips_value is None else lpips_value
        row[f"{prefix}_hole_lpips"] = "" if hole_lpips_value is None else hole_lpips_value

    for prefix, recon in (("raw", recon_raw), ("composite", recon_composite)):
        row[f"{prefix}_seam_mse"] = seam_mse(recon, label, mask_known, width=boundary_width)
        row[f"{prefix}_gradient_mismatch"] = gradient_mismatch(
            recon,
            label,
            mask_known,
            width=boundary_width,
            sigma=structural_sigma,
        )
        row[f"{prefix}_isophote_error"] = isophote_angle_error(
            recon,
            label,
            mask_known,
            width=boundary_width,
            sigma=structural_sigma,
        )
        row[f"{prefix}_laplacian_mismatch"] = laplacian_mismatch(
            recon,
            label,
            mask_known,
            width=boundary_width,
            sigma=structural_sigma,
        )
        row[f"{prefix}_ns_residual"] = ns_residual_metric(
            recon,
            mask_known,
            nu=0.1,
            kappa=0.1,
            smoothing_sigma=1.0,
            sigma=structural_sigma,
        )

    row["measurement_full_mse"] = mse_regions(measurement, label, mask_known)["full_mse"]
    _copy_diagnostics(row, diagnostics)
    return dict(row)


def _fieldnames(rows: Iterable[dict]) -> List[str]:
    names = OrderedDict()
    for row in rows:
        for key in row.keys():
            names.setdefault(key, None)
    return list(names.keys())


def write_summary_csv(rows: List[dict], path) -> None:
    """Write rows to CSV using the union of row keys as fieldnames."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def summarize_by_method(rows: List[dict]) -> List[dict]:
    """Group rows by method and compute mean/std for numeric fields."""

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("method", "")].append(row)

    summaries = []
    for method in sorted(grouped.keys()):
        method_rows = grouped[method]
        summary = OrderedDict()
        summary["method"] = method
        summary["count"] = len(method_rows)
        keys = sorted({key for row in method_rows for key in row.keys()})
        for key in keys:
            if key == "method":
                continue
            values = [float(row[key]) for row in method_rows if key in row and _numeric(row[key])]
            if not values:
                continue
            mean = sum(values) / float(len(values))
            summary[f"{key}_mean"] = mean
            if len(values) > 1:
                var = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
                summary[f"{key}_std"] = math.sqrt(var)
            else:
                summary[f"{key}_std"] = 0.0
        summaries.append(dict(summary))
    return summaries


def write_summary_markdown(summary: List[dict], path) -> None:
    """Write a simple human-readable Markdown table."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        output.write_text("| method | count |\n| --- | --- |\n", encoding="utf-8")
        return
    preferred = [
        "method",
        "count",
        "composite_psnr_mean",
        "composite_hole_mse_mean",
        "composite_seam_mse_mean",
        "composite_gradient_mismatch_mean",
        "composite_isophote_error_mean",
        "composite_lpips_mean",
        "composite_hole_lpips_mean",
        "flow_runtime_sec_mean",
        "sample_runtime_sec_mean",
    ]
    existing = _fieldnames(summary)
    columns = [key for key in preferred if key in existing]
    for key in existing:
        if key not in columns:
            columns.append(key)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary:
        cells = []
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
