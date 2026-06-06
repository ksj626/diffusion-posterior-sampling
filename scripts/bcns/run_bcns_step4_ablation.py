#!/usr/bin/env python
"""BCNS Step 4 evaluation and ablation runner."""

import argparse
import copy
import csv
import math
import random
import sys
import time
from collections import OrderedDict, defaultdict
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

ALLOWED_SAMPLING_STEPS = (25, 50, 100, 250, 1000)


def load_yaml(path: str) -> dict:
    with open(path) as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def validate_sampling_steps(sampling_steps: int) -> int:
    sampling_steps = int(sampling_steps)
    if sampling_steps not in ALLOWED_SAMPLING_STEPS:
        raise ValueError(f"sampling_steps must be one of {ALLOWED_SAMPLING_STEPS}, got {sampling_steps}.")
    return sampling_steps


def diffusion_config_for_sampling_steps(diffusion_config: dict, sampling_steps: int) -> dict:
    sampling_steps = validate_sampling_steps(sampling_steps)
    config = copy.deepcopy(diffusion_config)
    config["timestep_respacing"] = str(sampling_steps)
    return config


def create_sampler_for_sampling_steps(diffusion_config: dict, sampling_steps: int):
    return create_sampler(**diffusion_config_for_sampling_steps(diffusion_config, sampling_steps))


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


def _add_guidance_weights(
    params,
    rgb_loss_weight: float = 1.0,
    structure_loss_weight: float = 0.0,
    structure_loss_sigma: float = 1.0,
):
    params = copy.deepcopy(params)
    params["rgb_loss_weight"] = float(rgb_loss_weight)
    params["structure_loss_weight"] = float(structure_loss_weight)
    params["structure_loss_sigma"] = float(structure_loss_sigma)
    return params


def _structure_params(
    projected: bool = True,
    rgb_loss_weight: float = 1.0,
    structure_loss_weight: float = 0.0,
):
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
        "rgb_loss_weight": float(rgb_loss_weight),
        "structure_loss_weight": float(structure_loss_weight),
        "structure_loss_sigma": 1.0,
    }


def _poisson_params(
    target_builder: str,
    poisson_method: str = "sor_rb",
    rhs_mode: str = None,
    projected: bool = True,
    poisson_max_iter: int = 200,
    lambda_structure: float = 0.05,
    prox_steps: int = 1,
    prox_step_size: float = 0.05,
    rgb_loss_weight: float = 1.0,
    structure_loss_weight: float = 0.0,
    gamma_max: float = 0.02,
):
    params = {
        "scale": 1.0,
        "target_builder": target_builder,
        "target_builder_params": {
            "tau2": 1.0,
            "lambda_structure": float(lambda_structure),
            "structure_sigma": 1.0,
            "prox_steps": int(prox_steps),
            "prox_step_size": float(prox_step_size),
            "boundary_mode": "normalized_known_smooth",
            "poisson_method": poisson_method,
            "poisson_max_iter": int(poisson_max_iter),
            "poisson_tol": 1e-4,
            "poisson_omega": 1.7,
            "poisson_h": 1.0,
        },
        "gamma_max": float(gamma_max),
        "tau2": 1.0,
        "pde_start_frac": 0.7,
        "ramp_power": 2.0,
        "apply_noisy_known_projection": bool(projected),
        "rgb_loss_weight": float(rgb_loss_weight),
        "structure_loss_weight": float(structure_loss_weight),
        "structure_loss_sigma": 1.0,
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
    lambda_structure: float = 0.05,
    prox_steps: int = 1,
    prox_step_size: float = 0.05,
    rgb_loss_weight: float = 1.0,
    structure_loss_weight: float = 0.0,
    gamma_max: float = 0.02,
):
    return {
        "scale": 1.0,
        "target_builder": "flow_structure",
        "target_builder_params": {
            "tau2": 1.0,
            "lambda_structure": float(lambda_structure),
            "structure_sigma": 1.0,
            "prox_steps": int(prox_steps),
            "prox_step_size": float(prox_step_size),
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
        "gamma_max": float(gamma_max),
        "tau2": 1.0,
        "pde_start_frac": 0.7,
        "ramp_power": 2.0,
        "apply_noisy_known_projection": bool(projected),
        "apply_every_n_steps": int(apply_every_n_steps),
        "reuse_last_target": bool(reuse_last_target),
        "rgb_loss_weight": float(rgb_loss_weight),
        "structure_loss_weight": float(structure_loss_weight),
        "structure_loss_sigma": 1.0,
    }


def _float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p").replace("e", "e")


def _weight_suffix(weight: float) -> str:
    if abs(float(weight) - 0.1) < 1e-12:
        return ""
    return f"_w{_float_tag(weight)}"


def _gamma_label(value: float) -> str:
    if abs(float(value) - 1.0) < 1e-12:
        return "1.0"
    return f"{float(value):g}"


def _strength_weight_label(value: float) -> str:
    return f"{float(value):g}"


def _luminance_lift_params(
    source: str,
    lift_scale: float,
    projected: bool = True,
    gamma_max: float = 0.5,
    mode: str = "equal_rgb",
):
    if source == "flow":
        params = _flow_params(
            "imex_be",
            projected=projected,
            apply_every_n_steps=5,
            rgb_loss_weight=1.0,
            structure_loss_weight=0.0,
            gamma_max=gamma_max,
        )
        params["target_builder"] = "luminance_lift_flow"
    elif source == "poisson":
        params = _poisson_params(
            "poisson_structure",
            "sor_rb",
            "projected_mu_laplacian",
            projected=projected,
            rgb_loss_weight=1.0,
            structure_loss_weight=0.0,
            gamma_max=gamma_max,
        )
        params["target_builder"] = "luminance_lift_poisson"
    elif source == "harmonic":
        params = _poisson_params(
            "harmonic_structure",
            "sor_rb",
            "zero",
            projected=projected,
            rgb_loss_weight=1.0,
            structure_loss_weight=0.0,
            gamma_max=gamma_max,
        )
        params["target_builder"] = "luminance_lift_harmonic"
    else:
        raise ValueError("source must be one of 'flow', 'poisson', or 'harmonic'.")
    params["target_builder_params"]["lift_scale"] = float(lift_scale)
    params["target_builder_params"]["mode"] = mode
    return params


def _method(method, conditioning, params, **meta):
    return (method, conditioning, params, meta)


def method_configs(ablation_set: str, scale_default: float, full_strength_grid: bool = False):
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

    if ablation_set == "guidance_activation":
        configs = [
            _method("projection_fixed", "projection_fixed", {}),
            _method("ps", "ps", {"scale": scale_default}),
            _method(
                "bcns_flow_rgb_only_projected",
                "bcns_target",
                _flow_params("imex_be", rgb_loss_weight=1.0, structure_loss_weight=0.0),
                ablation_rgb_loss_weight=1.0,
                ablation_structure_loss_weight=0.0,
            ),
        ]
        for weight in (0.03, 0.1, 0.3):
            suffix = _weight_suffix(weight)
            configs.extend(
                [
                    _method(
                        f"bcns_flow_struct_only_projected{suffix}",
                        "bcns_target",
                        _flow_params("imex_be", rgb_loss_weight=0.0, structure_loss_weight=weight),
                        ablation_rgb_loss_weight=0.0,
                        ablation_structure_loss_weight=float(weight),
                    ),
                    _method(
                        f"bcns_flow_rgb_struct_projected{suffix}",
                        "bcns_target",
                        _flow_params("imex_be", rgb_loss_weight=1.0, structure_loss_weight=weight),
                        ablation_rgb_loss_weight=1.0,
                        ablation_structure_loss_weight=float(weight),
                    ),
                    _method(
                        f"bcns_poisson_struct_only_projected{suffix}",
                        "bcns_target",
                        _poisson_params(
                            "poisson_structure",
                            "sor_rb",
                            "projected_mu_laplacian",
                            rgb_loss_weight=0.0,
                            structure_loss_weight=weight,
                        ),
                        ablation_rgb_loss_weight=0.0,
                        ablation_structure_loss_weight=float(weight),
                    ),
                    _method(
                        f"bcns_harmonic_struct_only_projected{suffix}",
                        "bcns_target",
                        _poisson_params(
                            "harmonic_structure",
                            "sor_rb",
                            "zero",
                            rgb_loss_weight=0.0,
                            structure_loss_weight=weight,
                        ),
                        ablation_rgb_loss_weight=0.0,
                        ablation_structure_loss_weight=float(weight),
                    ),
                ]
            )
        return configs

    if ablation_set == "proximal_strength":
        configs = []
        settings = [
            ("weak", 0.05, 1, 0.05),
            ("mid", 0.5, 3, 0.1),
            ("strong", 2.0, 5, 0.1),
        ]
        for label, lambda_structure, prox_steps, prox_step_size in settings:
            for structure_weight in (0.0, 0.1):
                suffix = "" if structure_weight == 0.0 else "_struct"
                configs.append(
                    _method(
                        f"bcns_flow_{label}{suffix}",
                        "bcns_target",
                        _flow_params(
                            "imex_be",
                            lambda_structure=lambda_structure,
                            prox_steps=prox_steps,
                            prox_step_size=prox_step_size,
                            rgb_loss_weight=1.0,
                            structure_loss_weight=structure_weight,
                        ),
                        ablation_lambda_structure=float(lambda_structure),
                        ablation_prox_steps=int(prox_steps),
                        ablation_prox_step_size=float(prox_step_size),
                        ablation_rgb_loss_weight=1.0,
                        ablation_structure_loss_weight=float(structure_weight),
                    )
                )
        return configs

    if ablation_set == "fewstep":
        configs = []
        for sampling_steps in ALLOWED_SAMPLING_STEPS:
            for method_name, conditioning_name, params in (
                ("ps", "ps", {"scale": scale_default}),
                ("projection_fixed", "projection_fixed", {}),
                ("mcg_fixed", "mcg_fixed", {"scale": scale_default}),
                (
                    "bcns_flow_struct_projected",
                    "bcns_target",
                    _flow_params("imex_be", rgb_loss_weight=0.0, structure_loss_weight=0.1),
                ),
                (
                    "bcns_poisson_struct_projected",
                    "bcns_target",
                    _poisson_params(
                        "poisson_structure",
                        "sor_rb",
                        "projected_mu_laplacian",
                        rgb_loss_weight=0.0,
                        structure_loss_weight=0.1,
                    ),
                ),
            ):
                configs.append(
                    _method(
                        method_name,
                        conditioning_name,
                        params,
                        sampling_steps=int(sampling_steps),
                        ablation_sampling_steps=int(sampling_steps),
                    )
                )
        return configs

    if ablation_set == "guidance_strength":
        gamma_values = (0.02, 0.1, 0.5, 1.0)
        weight_values = (0.1, 1.0, 10.0, 50.0)
        if full_strength_grid:
            configs = []
            for projected in (False, True):
                for every in (5, 10):
                    for gamma in gamma_values:
                        for weight in weight_values:
                            projected_tag = "_projected" if projected else ""
                            name = (
                                f"flow_struct{projected_tag}_g{_gamma_label(gamma)}"
                                f"_w{_strength_weight_label(weight)}_every{every}"
                            )
                            configs.append(
                                _method(
                                    name,
                                    "bcns_target",
                                    _flow_params(
                                        "imex_be",
                                        projected=projected,
                                        apply_every_n_steps=every,
                                        rgb_loss_weight=0.0,
                                        structure_loss_weight=weight,
                                        gamma_max=gamma,
                                    ),
                                    ablation_gamma_max=float(gamma),
                                    ablation_structure_loss_weight=float(weight),
                                    ablation_apply_every_n_steps=int(every),
                                    ablation_projected=bool(projected),
                                )
                            )
            return configs

        pairs = ((0.02, 0.1), (0.1, 1.0), (0.5, 10.0), (1.0, 50.0))
        configs = []
        for projected in (False, True):
            for gamma, weight in pairs:
                projected_tag = "_projected" if projected else ""
                name = f"flow_struct{projected_tag}_g{_gamma_label(gamma)}_w{_strength_weight_label(weight)}"
                configs.append(
                    _method(
                        name,
                        "bcns_target",
                        _flow_params(
                            "imex_be",
                            projected=projected,
                            apply_every_n_steps=5,
                            rgb_loss_weight=0.0,
                            structure_loss_weight=weight,
                            gamma_max=gamma,
                        ),
                        ablation_gamma_max=float(gamma),
                        ablation_structure_loss_weight=float(weight),
                        ablation_apply_every_n_steps=5,
                        ablation_projected=bool(projected),
                    )
                )
        return configs

    if ablation_set == "flow_strength_extended":
        settings = [
            ("flow_weak", 0.003, 0.001, 0.1),
            ("flow_mid", 0.01, 0.001, 0.1),
            ("flow_strong", 0.03, 0.001, 0.2),
            ("flow_very_strong", 0.1, 0.001, 0.2),
        ]
        return [
            _method(
                name,
                "bcns_target",
                _flow_params(
                    "imex_be",
                    dt=dt,
                    pseudo_time=pseudo_time,
                    nu=nu,
                    projected=True,
                    apply_every_n_steps=5,
                    rgb_loss_weight=0.0,
                    structure_loss_weight=10.0,
                    gamma_max=0.5,
                ),
                sampling_steps=100,
                ablation_pseudo_time=float(pseudo_time),
                ablation_dt=float(dt),
                ablation_nu=float(nu),
                ablation_gamma_max=0.5,
                ablation_structure_loss_weight=10.0,
                ablation_apply_every_n_steps=5,
            )
            for name, pseudo_time, dt, nu in settings
        ]

    if ablation_set == "luminance_lift":
        return [
            _method("projection_fixed", "projection_fixed", {}, sampling_steps=100),
            _method("ps", "ps", {"scale": scale_default}, sampling_steps=100),
            _method(
                "bcns_flow_struct_projected",
                "bcns_target",
                _flow_params(
                    "imex_be",
                    projected=True,
                    apply_every_n_steps=5,
                    rgb_loss_weight=0.0,
                    structure_loss_weight=10.0,
                    gamma_max=0.5,
                ),
                sampling_steps=100,
            ),
            _method(
                "bcns_lift_flow_s0.5_projected",
                "bcns_target",
                _luminance_lift_params("flow", 0.5, projected=True),
                sampling_steps=100,
                ablation_lift_scale=0.5,
            ),
            _method(
                "bcns_lift_flow_s1.0_projected",
                "bcns_target",
                _luminance_lift_params("flow", 1.0, projected=True),
                sampling_steps=100,
                ablation_lift_scale=1.0,
            ),
            _method(
                "bcns_lift_flow_s2.0_projected",
                "bcns_target",
                _luminance_lift_params("flow", 2.0, projected=True),
                sampling_steps=100,
                ablation_lift_scale=2.0,
            ),
            _method(
                "bcns_lift_poisson_s1.0_projected",
                "bcns_target",
                _luminance_lift_params("poisson", 1.0, projected=True),
                sampling_steps=100,
                ablation_lift_scale=1.0,
            ),
            _method(
                "bcns_lift_harmonic_s1.0_projected",
                "bcns_target",
                _luminance_lift_params("harmonic", 1.0, projected=True),
                sampling_steps=100,
                ablation_lift_scale=1.0,
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


def write_guidance_trace(method_dir, trace):
    if not trace:
        return
    write_summary_csv([dict(row) for row in trace], method_dir / "guidance_trace.csv")


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


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def summarize_by_keys(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    summaries = []
    for group in sorted(grouped.keys()):
        group_rows = grouped[group]
        summary = OrderedDict()
        for key, value in zip(group_keys, group):
            summary[key] = value
        summary["count"] = len(group_rows)
        keys = sorted({key for row in group_rows for key in row.keys()})
        for key in keys:
            if key in group_keys:
                continue
            values = [float(row[key]) for row in group_rows if key in row and _numeric(row[key])]
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
        choices=(
            "smoke",
            "flow_integrator",
            "flow_strength",
            "frequency",
            "poisson_solver",
            "projection_effect",
            "guidance_activation",
            "proximal_strength",
            "fewstep",
            "guidance_strength",
            "flow_strength_extended",
            "luminance_lift",
        ),
        default="smoke",
    )
    parser.add_argument("--sampling_steps", type=int, choices=ALLOWED_SAMPLING_STEPS, default=1000)
    parser.add_argument("--full_strength_grid", action="store_true")
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
    sampler_cache = {}

    def sampler_for_steps(sampling_steps: int):
        sampling_steps = validate_sampling_steps(sampling_steps)
        if sampling_steps not in sampler_cache:
            sampler_cache[sampling_steps] = create_sampler_for_sampling_steps(diffusion_config, sampling_steps)
        return sampler_cache[sampling_steps]

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
    configs = method_configs(args.ablation_set, scale_default, full_strength_grid=args.full_strength_grid)
    write_config_used(root / "config_used.yaml", args, model_config, diffusion_config, task_config, configs)

    for image_index, ref_img in enumerate(loader):
        if image_index >= args.num_images:
            break
        image_id = str(image_index).zfill(5)
        if args.ablation_set != "fewstep":
            image_dir = root / image_id
            image_dir.mkdir(parents=True, exist_ok=True)
        ref_img = ref_img.to(device)
        set_seed(args.seed + image_index)
        mask = make_mask(args, mask_gen, ref_img)
        measurement_clean = operator.forward(ref_img, mask=mask)
        measurement_noisy = noiser(measurement_clean)
        x_start_base = torch.randn(ref_img.shape, device=device)
        comparison_groups = defaultdict(lambda: {"paths": [], "labels": []})

        for method_name, conditioning_name, params, meta in configs:
            sampling_steps = int(meta.get("sampling_steps", args.sampling_steps))
            sampler = sampler_for_steps(sampling_steps)
            run_root = root / f"steps_{sampling_steps:04d}" if args.ablation_set == "fewstep" else root
            image_dir = run_root / image_id
            image_dir.mkdir(parents=True, exist_ok=True)
            method_dir = image_dir / method_name
            progress_dir = method_dir / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            cond_method = get_conditioning_method(conditioning_name, operator, noiser, **params)
            if hasattr(cond_method, "reset_trace"):
                cond_method.reset_trace()
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
            diagnostics = dict(getattr(cond_method, "last_diagnostics", {}) or {})
            if hasattr(cond_method, "get_trace"):
                write_guidance_trace(method_dir, cond_method.get_trace())
            if hasattr(cond_method, "summarize_trace"):
                diagnostics.update(cond_method.summarize_trace())
            write_method_diagnostics(method_dir, diagnostics)
            recon_composite = clean_composite(recon_raw, measurement_clean, mask)
            ordered, labels = write_method_outputs(
                method_dir, measurement_noisy, mask, ref_img, recon_raw, recon_composite
            )
            group_key = (run_root, image_id)
            comparison_groups[group_key]["paths"].extend(ordered)
            comparison_groups[group_key]["labels"].extend([f"{method_name} {label}" for label in labels])

            row = OrderedDict()
            row["image_index"] = image_index
            row["image_id"] = image_id
            row["method"] = method_name
            row["conditioning_method"] = conditioning_name
            row["ablation_set"] = args.ablation_set
            row["seed"] = args.seed
            row["mask_mode"] = args.mask_mode
            row["sampling_steps"] = sampling_steps
            row["actual_num_reverse_updates"] = int(sampler.num_timesteps)
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

        for (run_root, grouped_image_id), group in comparison_groups.items():
            make_contact_sheet(
                group["paths"],
                group["labels"],
                run_root / grouped_image_id / "comparison_contact_sheet.png",
                cols=7,
            )

    write_summary_csv(rows, root / "metrics.csv")
    summary = summarize_by_keys(rows, ("sampling_steps", "method")) if args.ablation_set == "fewstep" else summarize_by_method(rows)
    write_summary_csv(summary, root / "summary_by_method.csv")
    write_summary_markdown(summary, root / "summary_by_method.md")
    print(f"Wrote Step 4 ablation outputs to {root}")


if __name__ == "__main__":
    main()
