#!/usr/bin/env python
"""BCNS Step 4 evaluation and ablation runner."""

import argparse
import copy
import csv
import random
import sys
import time
from collections import OrderedDict
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

from bcns.dps_adapter import clean_composite
from bcns.eval_table import (
    evaluate_inpainting_result,
    summarize_by_method,
    write_summary_csv,
    write_summary_markdown,
)
from bcns.masks import make_center_box_mask, make_thin_scratch_mask
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


def _structure_params(projected: bool = True):
    return {
        "scale": 1.0,
        "target_builder": "structure_prox",
        "target_builder_params": {
            "tau2": 1.0,
            "lambda_structure": 0.05,
            "structure_sigma": 1.0,
            "prox_steps": 1,
            "prox_step_size": 0.05,
            "target_mode": "normalized_known_smooth",
        },
        "gamma_max": 0.02,
        "tau2": 1.0,
        "pde_start_frac": 0.7,
        "ramp_power": 2.0,
        "apply_noisy_known_projection": bool(projected),
    }


def _poisson_params(
    target_builder: str,
    poisson_method: str = "sor_rb",
    rhs_mode: str = None,
    projected: bool = True,
    poisson_max_iter: int = 200,
):
    params = {
        "scale": 1.0,
        "target_builder": target_builder,
        "target_builder_params": {
            "tau2": 1.0,
            "lambda_structure": 0.05,
            "structure_sigma": 1.0,
            "prox_steps": 1,
            "prox_step_size": 0.05,
            "boundary_mode": "normalized_known_smooth",
            "poisson_method": poisson_method,
            "poisson_max_iter": int(poisson_max_iter),
            "poisson_tol": 1e-4,
            "poisson_omega": 1.7,
            "poisson_h": 1.0,
        },
        "gamma_max": 0.02,
        "tau2": 1.0,
        "pde_start_frac": 0.7,
        "ramp_power": 2.0,
        "apply_noisy_known_projection": bool(projected),
    }
    if rhs_mode is not None:
        params["target_builder_params"]["rhs_mode"] = rhs_mode
    return params


def _flow_params(
    integrator: str,
    dt: float = 1e-3,
    pseudo_time: float = 3e-3,
    nu: float = 0.1,
    projected: bool = True,
    apply_every_n_steps: int = 10,
    reuse_last_target: bool = False,
):
    return {
        "scale": 1.0,
        "target_builder": "flow_structure",
        "target_builder_params": {
            "tau2": 1.0,
            "lambda_structure": 0.05,
            "structure_sigma": 1.0,
            "prox_steps": 1,
            "prox_step_size": 0.05,
            "initial_mode": "projected_mu",
            "boundary_mode": "normalized_known_smooth",
            "boundary_vorticity_mode": "none",
            "integrator": integrator,
            "poisson_method": "sor_rb",
            "poisson_max_iter": 100,
            "poisson_tol": 1e-4,
            "poisson_omega": 1.7,
            "h": 1.0,
            "dt": float(dt),
            "pseudo_time": float(pseudo_time),
            "nu": float(nu),
            "kappa": 0.1,
            "smoothing_sigma": 1.0,
            "cfl": 0.25,
            "check_cfl": True,
            "vorticity_boundary_mode": "none",
        },
        "gamma_max": 0.02,
        "tau2": 1.0,
        "pde_start_frac": 0.7,
        "ramp_power": 2.0,
        "apply_noisy_known_projection": bool(projected),
        "apply_every_n_steps": int(apply_every_n_steps),
        "reuse_last_target": bool(reuse_last_target),
    }


def _float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p").replace("e", "e")


def _method(method, conditioning, params, **meta):
    return (method, conditioning, params, meta)


def method_configs(ablation_set: str, scale_default: float):
    if ablation_set == "smoke":
        return [
            _method("ps", "ps", {"scale": scale_default}),
            _method("projection_fixed", "projection_fixed", {}),
            _method("mcg_fixed", "mcg_fixed", {"scale": scale_default}),
            _method("bcns_structure_prox_norm_projected", "bcns_target", _structure_params(True)),
            _method("bcns_harmonic_sor_projected", "bcns_target", _poisson_params("harmonic_structure", "sor_rb")),
            _method(
                "bcns_poisson_sor_projected",
                "bcns_target",
                _poisson_params("poisson_structure", "sor_rb", "projected_mu_laplacian"),
            ),
            _method("bcns_flow_be_projected", "bcns_target", _flow_params("imex_be")),
        ]

    if ablation_set == "flow_integrator":
        return [
            _method("bcns_flow_be_projected", "bcns_target", _flow_params("imex_be"), flow_integrator="imex_be"),
            _method("bcns_flow_cn_projected", "bcns_target", _flow_params("imex_cn"), flow_integrator="imex_cn"),
            _method(
                "bcns_flow_ftcs_projected",
                "bcns_target",
                _flow_params("ftcs", dt=1e-4, pseudo_time=5e-4),
                flow_integrator="ftcs",
            ),
        ]

    if ablation_set == "flow_strength":
        configs = []
        for pseudo_time in (0.0, 1e-3, 3e-3, 1e-2):
            for nu in (0.0, 0.05, 0.1, 0.2):
                name = f"bcns_flow_be_t{_float_tag(pseudo_time)}_nu{_float_tag(nu)}_projected"
                configs.append(
                    _method(
                        name,
                        "bcns_target",
                        _flow_params("imex_be", pseudo_time=pseudo_time, nu=nu),
                        ablation_pseudo_time=float(pseudo_time),
                        ablation_nu=float(nu),
                    )
                )
        return configs

    if ablation_set == "frequency":
        configs = []
        for every in (1, 5, 10, 20, 50):
            for reuse in (False, True):
                name = f"bcns_flow_be_every{every}_reuse{int(reuse)}_projected"
                configs.append(
                    _method(
                        name,
                        "bcns_target",
                        _flow_params("imex_be", apply_every_n_steps=every, reuse_last_target=reuse),
                        ablation_apply_every_n_steps=int(every),
                        ablation_reuse_last_target=bool(reuse),
                    )
                )
        return configs

    if ablation_set == "poisson_solver":
        configs = []
        for method in ("sor_rb", "cg"):
            for max_iter in (50, 100, 200):
                name = f"bcns_poisson_{method}_iter{max_iter}_projected"
                configs.append(
                    _method(
                        name,
                        "bcns_target",
                        _poisson_params(
                            "poisson_structure",
                            poisson_method=method,
                            rhs_mode="projected_mu_laplacian",
                            poisson_max_iter=max_iter,
                        ),
                        ablation_poisson_method=method,
                        ablation_poisson_max_iter=int(max_iter),
                    )
                )
        return configs

    if ablation_set == "projection_effect":
        return [
            _method("ps", "ps", {"scale": scale_default}),
            _method("bcns_flow_be", "bcns_target", _flow_params("imex_be", projected=False)),
            _method("bcns_flow_be_projected", "bcns_target", _flow_params("imex_be", projected=True)),
            _method(
                "bcns_poisson_sor",
                "bcns_target",
                _poisson_params("poisson_structure", "sor_rb", "projected_mu_laplacian", projected=False),
            ),
            _method(
                "bcns_poisson_sor_projected",
                "bcns_target",
                _poisson_params("poisson_structure", "sor_rb", "projected_mu_laplacian", projected=True),
            ),
        ]

    raise ValueError(f"Unsupported ablation_set {ablation_set!r}.")


def _condition(result, default_loss):
    if isinstance(result, tuple):
        return result
    return result, default_loss


def make_mask(args, mask_gen, ref_img):
    if args.mask_mode == "random":
        return mask_gen(ref_img)[:, 0:1, :, :]
    height, width = int(ref_img.shape[-2]), int(ref_img.shape[-1])
    if args.mask_mode == "center_box":
        unknown = make_center_box_mask(
            height,
            width,
            args.box_size,
            args.box_size,
            device=ref_img.device,
            dtype=ref_img.dtype,
        )
    elif args.mask_mode == "thin_scratch":
        unknown = make_thin_scratch_mask(
            height,
            width,
            thickness=args.scratch_thickness,
            device=ref_img.device,
            dtype=ref_img.dtype,
        )
    else:
        raise ValueError(f"Unsupported mask_mode {args.mask_mode!r}.")
    return (1.0 - unknown).repeat(ref_img.shape[0], 1, 1, 1)


def run_loop(
    sampler,
    model,
    x_start,
    measurement_clean,
    measurement_noisy,
    cond_fn,
    record_every,
    progress_dir,
    desc,
):
    img = x_start
    last_loss = torch.zeros((), device=img.device)
    pbar = tqdm(list(range(sampler.num_timesteps))[::-1], desc=desc, leave=False)
    for idx in pbar:
        time_tensor = torch.tensor([idx] * img.shape[0], device=img.device)
        img = img.requires_grad_()
        out = sampler.p_sample(x=img, t=time_tensor, model=model)
        noisy_measurement_t = sampler.q_sample(measurement_noisy, t=time_tensor)
        conditioned = cond_fn(
            x_t=out["sample"],
            measurement=measurement_clean,
            noisy_measurement=noisy_measurement_t,
            x_prev=img,
            x_0_hat=out["pred_xstart"],
            t_index=int(idx),
            num_steps=sampler.num_timesteps,
            time=time_tensor,
        )
        img, last_loss = _condition(conditioned, last_loss)
        img = img.detach_()
        pbar.set_postfix({"loss": float(last_loss.item())}, refresh=False)
        if record_every > 0 and int(idx) % record_every == 0:
            save_tensor_image(img, progress_dir / f"x_{str(int(idx)).zfill(4)}.png")
    return img, last_loss


def write_method_diagnostics(method_dir, diagnostics):
    if not diagnostics:
        return
    path = method_dir / "diagnostics.csv"
    keys = sorted(diagnostics.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerow({key: diagnostics[key] for key in keys})


def write_method_outputs(method_dir, measurement_noisy, mask, label, recon_raw, recon_composite):
    method_dir.mkdir(parents=True, exist_ok=True)
    size = (int(label.shape[-1]), int(label.shape[-2]))
    paths = {
        "input": method_dir / "input.png",
        "mask": method_dir / "mask.png",
        "label": method_dir / "label.png",
        "raw": method_dir / "recon_raw.png",
        "composite": method_dir / "recon_composite.png",
        "raw_error": method_dir / "raw_abs_error.png",
        "composite_error": method_dir / "composite_abs_error.png",
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
    make_contact_sheet(ordered, labels, method_dir / "contact_sheet.png", cols=7)
    return ordered, labels


def _method_specs_for_config(configs):
    specs = []
    for method_name, conditioning_name, params, meta in configs:
        specs.append(
            {
                "method": method_name,
                "conditioning_method": conditioning_name,
                "params": copy.deepcopy(params),
                "meta": copy.deepcopy(meta),
            }
        )
    return specs


def write_config_used(path, args, model_config, diffusion_config, task_config, configs):
    payload = {
        "args": vars(args),
        "model_config": model_config,
        "diffusion_config": diffusion_config,
        "task_config": task_config,
        "methods": _method_specs_for_config(configs),
    }
    with Path(path).open("w") as handle:
        yaml.dump(payload, handle, sort_keys=False)


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
    parser.add_argument("--mask_mode", choices=("random", "center_box", "thin_scratch"), default="thin_scratch")
    parser.add_argument("--box_size", type=int, default=96)
    parser.add_argument("--scratch_thickness", type=int, default=5)
    parser.add_argument(
        "--ablation_set",
        choices=("smoke", "flow_integrator", "flow_strength", "frequency", "poisson_solver", "projection_effect"),
        default="smoke",
    )
    parser.add_argument("--structural_sigma", type=float, default=1.0)
    parser.add_argument("--boundary_width", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    assert_checkpoint_exists(model_config)
    model = create_model(**model_config).to(device)
    model.eval()
    operator = get_operator(device=device, **task_config["measurement"]["operator"])
    noiser = get_noise(**task_config["measurement"]["noise"])
    sampler = create_sampler(**diffusion_config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = get_dataset(**task_config["data"], transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    mask_gen = mask_generator(**task_config["measurement"]["mask_opt"])

    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    base_params = task_config.get("conditioning", {}).get("params", {})
    scale_default = float(base_params.get("scale", 1.0))
    configs = method_configs(args.ablation_set, scale_default)
    write_config_used(root / "config_used.yaml", args, model_config, diffusion_config, task_config, configs)

    for image_index, ref_img in enumerate(loader):
        if image_index >= args.num_images:
            break
        image_id = str(image_index).zfill(5)
        image_dir = root / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        ref_img = ref_img.to(device)
        set_seed(args.seed + image_index)
        mask = make_mask(args, mask_gen, ref_img)
        measurement_clean = operator.forward(ref_img, mask=mask)
        measurement_noisy = noiser(measurement_clean)
        x_start_base = torch.randn(ref_img.shape, device=device)
        comparison_paths = []
        comparison_labels = []

        for method_name, conditioning_name, params, meta in configs:
            method_dir = image_dir / method_name
            progress_dir = method_dir / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            cond_method = get_conditioning_method(conditioning_name, operator, noiser, **params)
            cond_fn = partial(cond_method.conditioning, mask=mask)
            set_seed(args.seed + image_index * 10007 + 777)
            x_start = x_start_base.clone().detach().requires_grad_()
            start_time = time.time()
            recon_raw, final_loss = run_loop(
                sampler,
                model,
                x_start,
                measurement_clean,
                measurement_noisy,
                cond_fn,
                args.record_every,
                progress_dir,
                desc=f"{image_id} {method_name}",
            )
            sample_runtime_sec = time.time() - start_time
            diagnostics = getattr(cond_method, "last_diagnostics", {}) or {}
            write_method_diagnostics(method_dir, diagnostics)
            recon_composite = clean_composite(recon_raw, measurement_clean, mask)
            ordered, labels = write_method_outputs(
                method_dir, measurement_noisy, mask, ref_img, recon_raw, recon_composite
            )
            comparison_paths.extend(ordered)
            comparison_labels.extend([f"{method_name} {label}" for label in labels])

            row = OrderedDict()
            row["image_index"] = image_index
            row["image_id"] = image_id
            row["method"] = method_name
            row["conditioning_method"] = conditioning_name
            row["ablation_set"] = args.ablation_set
            row["seed"] = args.seed
            row["mask_mode"] = args.mask_mode
            row["final_loss"] = float(final_loss.item())
            row["sample_runtime_sec"] = float(sample_runtime_sec)
            row.update(meta)
            row.update(
                evaluate_inpainting_result(
                    recon_raw=recon_raw,
                    recon_composite=recon_composite,
                    label=ref_img,
                    measurement=measurement_clean,
                    mask_known=mask,
                    diagnostics=diagnostics,
                    structural_sigma=args.structural_sigma,
                    boundary_width=args.boundary_width,
                )
            )
            row["recon_raw_min"] = float(recon_raw.min().item())
            row["recon_raw_max"] = float(recon_raw.max().item())
            row["recon_raw_mean"] = float(recon_raw.mean().item())
            row["has_nan"] = bool(torch.isnan(recon_raw).any().item() or torch.isnan(recon_composite).any().item())
            rows.append(dict(row))

        make_contact_sheet(
            comparison_paths,
            comparison_labels,
            image_dir / "comparison_contact_sheet.png",
            cols=7,
        )

    write_summary_csv(rows, root / "metrics.csv")
    summary = summarize_by_method(rows)
    write_summary_csv(summary, root / "summary_by_method.csv")
    write_summary_markdown(summary, root / "summary_by_method.md")
    print(f"Wrote Step 4 ablation outputs to {root}")


if __name__ == "__main__":
    main()
