#!/usr/bin/env python3
import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/dmsdmswns/diffusion-posterior-sampling/data/samples",
    )
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="student/FFHQ",
        help="Hugging Face FFHQ mirror.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=60000,
        help="Use 60000 to mimic FFHQ validation split when available.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name, split="train", streaming=True)

    saved = 0
    seen = 0

    for ex in ds:
        if seen < args.start_idx:
            seen += 1
            continue

        img = None
        for key in ["image", "img", "jpg", "png"]:
            if key in ex:
                img = ex[key]
                break

        if img is None:
            raise KeyError(f"Cannot find image column. Available keys: {list(ex.keys())}")

        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img = img.convert("RGB")
        img = img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)

        out_path = out_dir / f"{saved:05d}.png"
        img.save(out_path)

        saved += 1
        seen += 1

        if saved % 10 == 0:
            print(f"[INFO] Saved {saved}/{args.num_images}")

        if saved >= args.num_images:
            break

    print(f"[DONE] Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()