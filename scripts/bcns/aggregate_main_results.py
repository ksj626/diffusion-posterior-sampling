#!/usr/bin/env python
"""Aggregate BCNS main-result sweep outputs."""

import argparse
import csv
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is available in the DPS env.
    yaml = None


MAIN_MASKS = (
    "thick_scratch_24",
    "text_mask",
    "freeform_medium",
    "center_box_96",
    "center_box_128",
    "center_keep_96",
    "center_keep_128",
    "global_random_50_60",
)
MAIN_STEPS = (20, 50, 100, 250)
REQUIRED_FIELDS = (
    "mask_mode",
    "sampling_steps",
    "method",
    "count",
    "composite_hole_mse",
    "composite_hole_psnr",
    "composite_hole_ssim",
    "composite_seam_mse",
    "composite_gradient_mismatch",
    "composite_isophote_error",
    "composite_laplacian_mismatch",
    "composite_lpips",
    "composite_hole_lpips",
    "sample_runtime_sec",
    "elliptic_solver",
    "poisson_method",
    "solver_num_iter",
    "poisson_final_residual",
    "poisson_runtime_sec",
)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: Iterable[dict]) -> List[str]:
    names = OrderedDict()
    for row in rows:
        for key in row.keys():
            names.setdefault(key, None)
    return list(names.keys())


def _write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    if not fieldnames:
        fieldnames = list(REQUIRED_FIELDS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field for field in REQUIRED_FIELDS if any(field in row for row in rows)]
    if not columns:
        columns = ["mask_mode", "sampling_steps", "method", "count"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _numeric(value) -> bool:
    if isinstance(value, bool) or value in ("", None):
        return False
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(converted)


def _mean_std(values: List[float]):
    mean = sum(values) / float(len(values))
    if len(values) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return mean, math.sqrt(var)


def _load_config_metadata(run_dir: Path) -> Dict[str, object]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists() or yaml is None:
        return {}
    with config_path.open() as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    return {
        "ablation_set": args.get("ablation_set", ""),
        "mask_mode": args.get("mask_mode", ""),
        "sampling_steps": args.get("sampling_steps", ""),
        "num_images_requested": args.get("num_images", ""),
        "seed": args.get("seed", ""),
    }


def _first_nonempty(rows: List[dict], key: str):
    for row in rows:
        value = row.get(key, "")
        if value != "":
            return value
    return ""


def _summarize_metrics_file(metrics_path: Path) -> List[dict]:
    rows = _read_csv(metrics_path)
    if not rows:
        return []
    run_dir = metrics_path.parent
    config_meta = _load_config_metadata(run_dir)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("method", "")].append(row)

    metadata_keys = {
        "method",
        "conditioning_method",
        "ablation_set",
        "mask_mode",
        "sampling_steps",
        "actual_num_reverse_updates",
        "elliptic_solver",
        "poisson_method",
        "method_family",
        "target_source",
        "transfer_mode",
        "projected",
        "sweep_family",
    }
    summaries = []
    for method in sorted(grouped.keys()):
        method_rows = grouped[method]
        summary = OrderedDict()
        summary["source_dir"] = str(run_dir)
        summary.update(config_meta)
        for key in metadata_keys:
            value = _first_nonempty(method_rows, key)
            if value != "":
                summary[key] = value
        summary["method"] = method
        summary["count"] = len(method_rows)
        keys = sorted({key for row in method_rows for key in row.keys()})
        for key in keys:
            if key in metadata_keys or key in ("image_id", "image_index"):
                continue
            values = [float(row[key]) for row in method_rows if _numeric(row.get(key, ""))]
            if not values:
                continue
            mean, std = _mean_std(values)
            summary[key] = mean
            summary[f"{key}_std"] = std
        summaries.append(dict(summary))
    return summaries


def _summary_fallback(summary_path: Path) -> List[dict]:
    run_dir = summary_path.parent
    config_meta = _load_config_metadata(run_dir)
    rows = []
    for source in _read_csv(summary_path):
        row = OrderedDict()
        row["source_dir"] = str(run_dir)
        row.update(config_meta)
        row["method"] = source.get("method", "")
        row["count"] = source.get("count", "")
        for key, value in source.items():
            if key.endswith("_mean"):
                row[key[:-5]] = value
            elif key not in row:
                row[key] = value
        rows.append(dict(row))
    return rows


def collect_rows(input_roots: List[Path]) -> List[dict]:
    rows = []
    seen_dirs = set()
    for root in input_roots:
        for summary_path in sorted(root.rglob("summary_by_method.csv")):
            run_dir = summary_path.parent
            if run_dir in seen_dirs:
                continue
            seen_dirs.add(run_dir)
            metrics_path = run_dir / "metrics.csv"
            metrics_rows = _summarize_metrics_file(metrics_path)
            rows.extend(metrics_rows if metrics_rows else _summary_fallback(summary_path))
    return rows


def _filter(rows: List[dict], ablation_set: str) -> List[dict]:
    return [row for row in rows if row.get("ablation_set") == ablation_set]


def _sort_rows(rows: List[dict]) -> List[dict]:
    def key(row):
        try:
            steps = int(float(row.get("sampling_steps", 0)))
        except (TypeError, ValueError):
            steps = 0
        return (str(row.get("mask_mode", "")), steps, str(row.get("method", "")))

    return sorted(rows, key=key)


def _write_group(rows: List[dict], output_dir: Path, stem: str) -> None:
    rows = _sort_rows(rows)
    _write_csv(rows, output_dir / f"{stem}.csv")
    _write_md(rows, output_dir / f"{stem}.md")


def _write_best_report(output_dir: Path) -> None:
    text = f"""# BCNS Main Result Configuration Report

## Harmonic Best

- method: `mcg_bcns_harmonic_best`
- conditioning: `mcg_bcns`
- MCG scale: `1.0`
- target builder: `luminance_lift_harmonic`
- ratio control: `true`
- target update ratio: `0.3`
- ratio clip max: `10000.0`
- apply every: `2`
- lift scale: `1.0`
- known projection: `true`

## Poisson Best

- method: `mcg_bcns_poisson_best`
- conditioning: `mcg_bcns`
- MCG scale: `1.0`
- target builder: `luminance_lift_poisson`
- ratio control: `true`
- target update ratio: `1.0`
- ratio clip max: `1000.0`
- apply every: `2`
- lift scale: `4.0`
- RHS mode: `projected_mu_laplacian`
- known projection: `true`

## Main Masks

{", ".join(f"`{mask}`" for mask in MAIN_MASKS)}

## Main Steps

{", ".join(f"`{step}`" for step in MAIN_STEPS)}

## Ablation Definitions

- `bcns_main_methods`: PS, fixed projection, fixed MCG, harmonic best, Poisson best.
- `bcns_main_solver_ablation_harmonic`: harmonic best with elliptic solver varied over `sor`, `cg`, `ftcs`, `be`, `cn`.
- `bcns_main_solver_ablation_poisson`: Poisson best with elliptic solver varied over `sor`, `cg`, `ftcs`, `be`, `cn`.
- `bcns_main_frequency_ablation`: fixed MCG, harmonic best, and `harmonic_r0.3_clip1000_every1`.

## Metric Definitions

- Composite metrics evaluate the final clean composite: known pixels from the measurement and hole pixels from the raw reconstruction.
- Hole LPIPS uses a hole-focused composite, with prediction in unknown pixels and ground truth in known pixels, compared against the full ground truth.
- Structural metrics report boundary seam MSE, gradient mismatch, isophote `1-cos` error, and Laplacian mismatch.
- Solver diagnostics report the harmonic/Poisson elliptic backend, not the Navier-Stokes flow integrator.
"""
    (output_dir / "best_config_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_roots", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows([Path(path) for path in args.input_roots])

    _write_csv(_sort_rows(rows), output_dir / "summary_all_long.csv")
    _write_group(_filter(rows, "bcns_main_methods"), output_dir, "summary_main_methods")
    _write_group(
        _filter(rows, "bcns_main_solver_ablation_harmonic"),
        output_dir,
        "summary_solver_harmonic",
    )
    _write_group(
        _filter(rows, "bcns_main_solver_ablation_poisson"),
        output_dir,
        "summary_solver_poisson",
    )
    _write_group(_filter(rows, "bcns_main_frequency_ablation"), output_dir, "summary_frequency")
    _write_best_report(output_dir)
    print(f"Wrote aggregated BCNS main results to {output_dir}")


if __name__ == "__main__":
    main()
