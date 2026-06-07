#!/usr/bin/env python
"""Run compact BCNS comparisons across structured hole variants."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.eval_table import write_summary_csv, write_summary_markdown
from scripts.bcns.run_bcns_step4_ablation import (
    ALLOWED_SAMPLING_STEPS,
    DEFAULT_HOLE_VARIANTS,
    summarize_by_keys,
)


def parse_hole_variants(value: str):
    if value is None or not value.strip():
        return list(DEFAULT_HOLE_VARIANTS)
    variants = [item.strip() for item in value.split(",") if item.strip()]
    if not variants:
        raise ValueError("hole_variants must contain at least one mask mode.")
    unsupported = [item for item in variants if item not in DEFAULT_HOLE_VARIANTS]
    if unsupported:
        raise ValueError(
            f"Unsupported hole variants {unsupported}; expected values from {list(DEFAULT_HOLE_VARIANTS)}."
        )
    return variants


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--diffusion_config", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--record_every", type=int, default=100)
    parser.add_argument("--sampling_steps", type=int, choices=ALLOWED_SAMPLING_STEPS, default=100)
    parser.add_argument("--ablation_set", default="hole_variant_sweep")
    parser.add_argument("--hole_variants", default=",".join(DEFAULT_HOLE_VARIANTS))
    parser.add_argument("--structural_sigma", type=float, default=1.0)
    parser.add_argument("--boundary_width", type=int, default=3)
    args = parser.parse_args()

    if args.ablation_set != "hole_variant_sweep":
        raise ValueError("run_hole_variant_sweep.py expects --ablation_set hole_variant_sweep.")

    variants = parse_hole_variants(args.hole_variants)
    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    runner = ROOT / "scripts" / "bcns" / "run_bcns_step4_ablation.py"
    for variant in variants:
        variant_dir = root / variant
        cmd = [
            sys.executable,
            str(runner),
            "--model_config",
            args.model_config,
            "--diffusion_config",
            args.diffusion_config,
            "--task_config",
            args.task_config,
            "--save_dir",
            str(variant_dir),
            "--gpu",
            str(args.gpu),
            "--num_images",
            str(args.num_images),
            "--seed",
            str(args.seed),
            "--mask_mode",
            variant,
            "--record_every",
            str(args.record_every),
            "--sampling_steps",
            str(args.sampling_steps),
            "--ablation_set",
            "hole_variant_sweep",
            "--structural_sigma",
            str(args.structural_sigma),
            "--boundary_width",
            str(args.boundary_width),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        for row in _read_csv(variant_dir / "metrics.csv"):
            row["hole_variant"] = variant
            all_rows.append(row)

    write_summary_csv(all_rows, root / "metrics_hole_variants.csv")
    summary = summarize_by_keys(all_rows, ("mask_mode", "method"))
    write_summary_csv(summary, root / "summary_hole_variants.csv")
    write_summary_markdown(summary, root / "summary_hole_variants.md")
    print(f"Wrote hole variant sweep outputs to {root}")


if __name__ == "__main__":
    main()
