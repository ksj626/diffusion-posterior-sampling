#!/usr/bin/env python
"""Visualize the BCNS Step 4.7 structured mask suite."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.eval_table import write_summary_csv
from bcns.masks import (
    make_center_box_unknown_mask,
    make_freeform_medium_mask,
    make_text_like_mask,
    make_thick_scratch_mask,
    make_thin_scratch_mask,
    mask_metadata,
)
from bcns.visualization import make_contact_sheet, save_mask_image


def _mask_specs(height: int, width: int):
    return [
        ("thin_scratch_5", make_thin_scratch_mask(height, width, thickness=5)),
        ("thick_scratch_12", make_thick_scratch_mask(height, width, thickness=12)),
        ("thick_scratch_24", make_thick_scratch_mask(height, width, thickness=24)),
        ("text_mask", make_text_like_mask(height, width, text="TEXT", thickness=12)),
        ("freeform_medium", make_freeform_medium_mask(height, width, seed=123)),
        ("center_box_96", make_center_box_unknown_mask(height, width, box_size=min(96, height, width))),
        ("center_box_128", make_center_box_unknown_mask(height, width, box_size=min(128, height, width))),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    root = Path(args.output_dir)
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = []
    labels = []
    for name, mask_unknown in _mask_specs(args.height, args.width):
        mask_known = 1.0 - mask_unknown.to(dtype=torch.float32)
        path = mask_dir / f"{name}.png"
        save_mask_image(mask_known, path)
        row = {"mask_name": name}
        row.update(mask_metadata(mask_unknown))
        rows.append(row)
        paths.append(path)
        labels.append(name)

    write_summary_csv(rows, root / "metadata.csv")
    make_contact_sheet(paths, labels, root / "contact_sheet.png", cols=4)
    print(f"Wrote mask suite visualization to {root}")


if __name__ == "__main__":
    main()
