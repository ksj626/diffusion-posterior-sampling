#!/usr/bin/env python
"""Controlled real DPS smoke run for BCNS Step 1 structure-prox guidance."""

import argparse
import csv
import json
import os
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

from bcns.dps_adapter import masked_mean_square
from bcns.visualization import make_contact_sheet, save_mask_image, save_tensor_image
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


def run_loop(sampler, model, x_start, measurement, cond_fn, record_every, progress_dir):
    img = x_start
    last_loss = torch.zeros((), device=img.device)
    pbar = tqdm(list(range(sampler.num_timesteps))[::-1])
    for idx in pbar:
        time = torch.tensor([idx] * img.shape[0], device=img.device)
        img = img.requires_grad_()
        out = sampler.p_sample(x=img, t=time, model=model)
        noisy_measurement = sampler.q_sample(measurement, t=time)
        img, last_loss = cond_fn(
            x_t=out["sample"],
            measurement=measurement,
            noisy_measurement=noisy_measurement,
            x_prev=img,
            x_0_hat=out["pred_xstart"],
            t_index=int(idx),
            num_steps=sampler.num_timesteps,
            time=time,
        )
        img = img.detach_()
        pbar.set_postfix({"loss": float(last_loss.item())}, refresh=False)
        if record_every > 0 and int(idx) % record_every == 0:
            save_tensor_image(img, progress_dir / f"x_{str(int(idx)).zfill(4)}.png")
    return img, last_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--diffusion_config", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--method_name", default="bcns_step1_structure_prox")
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
        task_config["conditioning"]["params"].update(json.loads(args.conditioning_params_json))

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
    for folder in ("input", "mask", "label", "recon", "progress"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    contact_paths = []
    contact_labels = []

    for image_index, ref_img in enumerate(loader):
        if image_index >= args.num_images:
            break
        ref_img = ref_img.to(device)
        mask = mask_gen(ref_img)[:, 0:1, :, :]
        measurement = operator.forward(ref_img, mask=mask)
        noisy_measurement = noiser(measurement)
        cond_fn = partial(cond_method.conditioning, mask=mask)
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        progress_dir = root / "progress" / str(image_index).zfill(5)
        progress_dir.mkdir(parents=True, exist_ok=True)
        recon, final_loss = run_loop(
            sampler,
            model,
            x_start,
            noisy_measurement,
            cond_fn,
            args.record_every,
            progress_dir,
        )

        stem = str(image_index).zfill(5)
        input_path = root / "input" / f"{stem}.png"
        mask_path = root / "mask" / f"{stem}.png"
        label_path = root / "label" / f"{stem}.png"
        recon_path = root / "recon" / f"{stem}.png"
        save_tensor_image(noisy_measurement, input_path)
        save_mask_image(mask, mask_path)
        save_tensor_image(ref_img, label_path)
        save_tensor_image(recon, recon_path)
        make_contact_sheet(
            [input_path, mask_path, label_path, recon_path],
            ["input", "mask", "label", "recon"],
            root / f"contact_sheet_{stem}.png",
            cols=4,
        )
        contact_paths.extend([input_path, mask_path, label_path, recon_path])
        contact_labels.extend([f"{stem} input", f"{stem} mask", f"{stem} label", f"{stem} recon"])

        known_mse = masked_mean_square(recon - noisy_measurement, mask).item()
        rows.append(
            {
                "image_index": image_index,
                "method": args.method_name,
                "seed": args.seed,
                "final_loss": float(final_loss.item()),
                "known_mse": float(known_mse),
                "recon_min": float(recon.min().item()),
                "recon_max": float(recon.max().item()),
                "recon_mean": float(recon.mean().item()),
                "has_nan": bool(torch.isnan(recon).any().item()),
            }
        )

    make_contact_sheet(contact_paths, contact_labels, root / "contact_sheet.png", cols=4)
    with (root / "diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_index",
                "method",
                "seed",
                "final_loss",
                "known_mse",
                "recon_min",
                "recon_max",
                "recon_mean",
                "has_nan",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote smoke outputs to {root}")


if __name__ == "__main__":
    main()
