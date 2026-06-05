#!/usr/bin/env python
"""Controlled real DPS smoke run for BCNS Step 1.5 structure-prox guidance."""

import argparse
import csv
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.dps_adapter import clean_composite, split_known_hole_mse
from bcns.visualization import make_contact_sheet, save_heatmap_uint8, save_mask_image, save_tensor_image
from data.dataloader import get_dataloader, get_dataset
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from util.img_utils import mask_generator


def load_yaml(path: str) -> dict:
    with open(path) as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_checkpoint_exists(model_config: dict) -> None:
    model_path = Path(model_config.get("model_path", ""))
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")


def _condition(result, x_t, default_loss):
    if isinstance(result, tuple):
        return result
    return result, default_loss


def run_loop(
    sampler,
    model,
    x_start,
    measurement_clean,
    measurement_noisy,
    cond_fn,
    record_every,
    progress_dir,
):
    img = x_start
    last_loss = torch.zeros((), device=img.device)
    pbar = tqdm(list(range(sampler.num_timesteps))[::-1])
    for idx in pbar:
        time = torch.tensor([idx] * img.shape[0], device=img.device)
        img = img.requires_grad_()
        out = sampler.p_sample(x=img, t=time, model=model)
        noisy_measurement_t = sampler.q_sample(measurement_noisy, t=time)
        conditioned = cond_fn(
            x_t=out["sample"],
            measurement=measurement_clean,
            noisy_measurement=noisy_measurement_t,
            x_prev=img,
            x_0_hat=out["pred_xstart"],
            t_index=int(idx),
            num_steps=sampler.num_timesteps,
            time=time,
        )
        img, last_loss = _condition(conditioned, out["sample"], last_loss)
        img = img.detach_()
        pbar.set_postfix({"loss": float(last_loss.item())}, refresh=False)
        if record_every > 0 and int(idx) % record_every == 0:
            save_tensor_image(img, progress_dir / f"x_{str(int(idx)).zfill(4)}.png")
    return img, last_loss


def _write_outputs(root, stem, measurement_noisy, mask, label, recon_raw, recon_composite):
    size = (int(label.shape[-1]), int(label.shape[-2]))
    paths = {
        "input": root / "input" / f"{stem}.png",
        "mask": root / "mask" / f"{stem}.png",
        "label": root / "label" / f"{stem}.png",
        "raw": root / "recon_raw" / f"{stem}.png",
        "composite": root / "recon_composite" / f"{stem}.png",
        "raw_error": root / "raw_abs_error" / f"{stem}.png",
        "composite_error": root / "composite_abs_error" / f"{stem}.png",
    }
    save_tensor_image(measurement_noisy, paths["input"])
    save_mask_image(mask, paths["mask"])
    save_tensor_image(label, paths["label"])
    save_tensor_image(recon_raw, paths["raw"])
    save_tensor_image(recon_composite, paths["composite"])
    save_heatmap_uint8((recon_raw - label).abs().mean(dim=1, keepdim=True), paths["raw_error"], target_size=size)
    save_heatmap_uint8(
        (recon_composite - label).abs().mean(dim=1, keepdim=True),
        paths["composite_error"],
        target_size=size,
    )
    return paths


def _metric_values(recon, label, mask):
    return {key: float(value.detach().item()) for key, value in split_known_hole_mse(recon, label, mask).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--diffusion_config", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--method_name", default="bcns_step1_5_structure_prox_norm")
    parser.add_argument("--conditioning_method", default=None)
    parser.add_argument("--conditioning_params_json", default=None)
    parser.add_argument("--record_every", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    if args.conditioning_method:
        task_config["conditioning"]["method"] = args.conditioning_method
    if args.conditioning_params_json:
        task_config["conditioning"].setdefault("params", {}).update(json.loads(args.conditioning_params_json))

    assert_checkpoint_exists(model_config)
    model = create_model(**model_config).to(device)
    model.eval()
    operator = get_operator(device=device, **task_config["measurement"]["operator"])
    noiser = get_noise(**task_config["measurement"]["noise"])
    cond_config = task_config["conditioning"]
    cond_method = get_conditioning_method(
        cond_config["method"], operator, noiser, **cond_config["params"]
    )
    sampler = create_sampler(**diffusion_config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = get_dataset(**task_config["data"], transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    mask_gen = mask_generator(**task_config["measurement"]["mask_opt"])

    root = Path(args.save_dir)
    for folder in (
        "input",
        "mask",
        "label",
        "recon_raw",
        "recon_composite",
        "raw_abs_error",
        "composite_abs_error",
        "progress",
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    contact_paths = []
    contact_labels = []

    for image_index, ref_img in enumerate(loader):
        if image_index >= args.num_images:
            break
        ref_img = ref_img.to(device)
        mask = mask_gen(ref_img)[:, 0:1, :, :]
        measurement_clean = operator.forward(ref_img, mask=mask)
        measurement_noisy = noiser(measurement_clean)
        cond_fn = partial(cond_method.conditioning, mask=mask)
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        progress_dir = root / "progress" / str(image_index).zfill(5)
        progress_dir.mkdir(parents=True, exist_ok=True)
        recon_raw, final_loss = run_loop(
            sampler,
            model,
            x_start,
            measurement_clean,
            measurement_noisy,
            cond_fn,
            args.record_every,
            progress_dir,
        )
        recon_composite = clean_composite(recon_raw, measurement_clean, mask)

        stem = str(image_index).zfill(5)
        paths = _write_outputs(root, stem, measurement_noisy, mask, ref_img, recon_raw, recon_composite)
        ordered = [
            paths["input"],
            paths["mask"],
            paths["label"],
            paths["raw"],
            paths["composite"],
            paths["raw_error"],
            paths["composite_error"],
        ]
        labels = ["input", "mask", "label", "raw", "composite", "raw error", "composite error"]
        make_contact_sheet(ordered, labels, root / f"contact_sheet_{stem}.png", cols=7)
        contact_paths.extend(ordered)
        contact_labels.extend([f"{stem} {label}" for label in labels])

        raw_metrics = _metric_values(recon_raw, ref_img, mask)
        composite_metrics = _metric_values(recon_composite, ref_img, mask)
        rows.append(
            {
                "image_index": image_index,
                "method": args.method_name,
                "seed": args.seed,
                "final_loss": float(final_loss.item()),
                "raw_known_mse": raw_metrics["known_mse"],
                "raw_hole_mse": raw_metrics["hole_mse"],
                "raw_full_mse": raw_metrics["full_mse"],
                "composite_known_mse": composite_metrics["known_mse"],
                "composite_hole_mse": composite_metrics["hole_mse"],
                "composite_full_mse": composite_metrics["full_mse"],
                "recon_raw_min": float(recon_raw.min().item()),
                "recon_raw_max": float(recon_raw.max().item()),
                "recon_raw_mean": float(recon_raw.mean().item()),
                "has_nan": bool(torch.isnan(recon_raw).any().item() or torch.isnan(recon_composite).any().item()),
            }
        )

    make_contact_sheet(contact_paths, contact_labels, root / "contact_sheet.png", cols=7)
    fieldnames = [
        "image_index",
        "method",
        "seed",
        "final_loss",
        "raw_known_mse",
        "raw_hole_mse",
        "raw_full_mse",
        "composite_known_mse",
        "composite_hole_mse",
        "composite_full_mse",
        "recon_raw_min",
        "recon_raw_max",
        "recon_raw_mean",
        "has_nan",
    ]
    with (root / "diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote smoke outputs to {root}")


if __name__ == "__main__":
    main()
