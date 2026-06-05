"""Structure-proximal target construction for BCNS DPS Step 1."""

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .dps_adapter import hard_project_clean, hole_from_known, masked_mean_square, validate_image_tensor, validate_mask_tensor


@dataclass
class StructureProxResult:
    """Result from structure-proximal RGB target construction."""

    target: torch.Tensor
    target_structure: torch.Tensor
    initial_structure: torch.Tensor
    loss_history: List[float]
    diagnostics: dict


def rgb_to_luminance_dps(x: torch.Tensor) -> torch.Tensor:
    """Convert DPS-scale ``[B, 3, H, W]`` RGB to luminance in the same scale."""

    validate_image_tensor(x, "x")
    if x.shape[1] != 3:
        raise ValueError(f"x must have 3 RGB channels, got {x.shape[1]}.")
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
    return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def gaussian_kernel_2d(
    sigma: float,
    truncate: float = 3.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return a normalized ``[1, 1, K, K]`` Gaussian kernel."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if sigma == 0:
        return torch.ones((1, 1, 1, 1), dtype=dtype or torch.float32, device=device)
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype or torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def gaussian_smooth_depthwise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply replicated-edge Gaussian smoothing independently per channel."""

    validate_image_tensor(x, "x")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if sigma == 0:
        return x
    kernel = gaussian_kernel_2d(sigma, device=x.device, dtype=x.dtype)
    channels = x.shape[1]
    weight = kernel.expand(channels, 1, kernel.shape[-2], kernel.shape[-1])
    pad_y = kernel.shape[-2] // 2
    pad_x = kernel.shape[-1] // 2
    padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    return F.conv2d(padded, weight, groups=channels)


def structure_image(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return ``S_sigma(x) = GaussianSmooth(Luminance(x))`` as ``[B, 1, H, W]``."""

    return gaussian_smooth_depthwise(rgb_to_luminance_dps(x), sigma)


def normalized_known_luminance_smooth(
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return normalized known-side smoothed luminance in DPS scale."""

    validate_image_tensor(measurement, "measurement")
    validate_mask_tensor(mask_known, measurement, "mask_known")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    lum = rgb_to_luminance_dps(measurement)
    mask = mask_known.to(dtype=measurement.dtype)
    numerator = gaussian_smooth_depthwise(mask * lum, sigma)
    denominator = gaussian_smooth_depthwise(mask, sigma)
    return numerator / (denominator + eps)


def _target_structure(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    structure_sigma: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "projected_mu":
        return structure_image(hard_project_clean(mu, measurement, mask_known), structure_sigma)
    if target_mode == "measurement_only_smooth":
        return structure_image(measurement, structure_sigma)
    if target_mode == "normalized_known_smooth":
        return normalized_known_luminance_smooth(measurement, mask_known, structure_sigma)
    if target_mode == "mu":
        return structure_image(mu, structure_sigma)
    raise ValueError(
        "target_mode must be one of 'projected_mu', 'measurement_only_smooth', "
        "'normalized_known_smooth', or 'mu'."
    )


def _validate_target_structure(target_structure: torch.Tensor, mu: torch.Tensor) -> None:
    if not torch.is_tensor(target_structure):
        raise TypeError("target_structure must be a torch.Tensor.")
    if target_structure.ndim != 4 or target_structure.shape[1] != 1:
        raise ValueError(
            f"target_structure must have shape [B, 1, H, W], got {tuple(target_structure.shape)}."
        )
    if target_structure.shape[0] != mu.shape[0] or target_structure.shape[-2:] != mu.shape[-2:]:
        raise ValueError("target_structure batch/spatial dimensions must match mu.")
    if target_structure.device != mu.device:
        raise ValueError("target_structure must be on the same device as mu.")
    if target_structure.dtype != mu.dtype:
        raise ValueError("target_structure must have the same dtype as mu.")
    if not torch.is_floating_point(target_structure):
        raise TypeError("target_structure must be a floating point tensor.")


def structure_proximal_target_from_structure(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    target_structure: torch.Tensor,
    tau2: float,
    lambda_structure: float,
    structure_sigma: float,
    prox_steps: int,
    prox_step_size: float,
) -> StructureProxResult:
    """Build a detached RGB proximal target from an external scalar structure."""

    validate_image_tensor(mu, "mu")
    if measurement.shape != mu.shape:
        raise ValueError("measurement must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    _validate_target_structure(target_structure, mu)
    if tau2 <= 0:
        raise ValueError("tau2 must be positive.")
    if lambda_structure < 0:
        raise ValueError("lambda_structure must be non-negative.")
    if structure_sigma < 0:
        raise ValueError("structure_sigma must be non-negative.")
    if prox_steps < 0:
        raise ValueError("prox_steps must be non-negative.")
    if prox_step_size <= 0:
        raise ValueError("prox_step_size must be positive.")

    with torch.no_grad():
        projected = hard_project_clean(mu, measurement, mask_known)
        target_structure = target_structure.detach()
        initial_structure = structure_image(projected, structure_sigma).detach()

    if prox_steps == 0 or lambda_structure == 0:
        target = projected.detach()
        return StructureProxResult(
            target=target,
            target_structure=target_structure,
            initial_structure=initial_structure,
            loss_history=[],
            diagnostics={
                "prox_final_loss": 0.0,
                "target_disp": float(torch.linalg.norm((target - mu).reshape(-1)).item()),
                "known_mse": float(masked_mean_square(target - measurement, mask_known).item()),
            },
        )

    hole = hole_from_known(mask_known).to(dtype=mu.dtype)
    u = projected.detach()
    loss_history: List[float] = []
    for _ in range(prox_steps):
        u = u.detach().requires_grad_(True)
        fidelity = 0.5 / tau2 * masked_mean_square(u - mu.detach(), hole)
        structure_delta = structure_image(u, structure_sigma) - target_structure
        structure_loss = 0.5 * lambda_structure * masked_mean_square(structure_delta, hole)
        loss = fidelity + structure_loss
        grad = torch.autograd.grad(loss, u, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            u = u - prox_step_size * grad
            u = hard_project_clean(u, measurement, mask_known)
        loss_history.append(float(loss.detach().item()))

    target = u.detach()
    final_loss = loss_history[-1] if loss_history else 0.0
    diagnostics: Dict[str, float] = {
        "prox_final_loss": final_loss,
        "target_disp": float(torch.linalg.norm((target - mu).reshape(-1)).item()),
        "known_mse": float(masked_mean_square(target - measurement, mask_known).item()),
    }
    return StructureProxResult(
        target=target,
        target_structure=target_structure,
        initial_structure=initial_structure,
        loss_history=loss_history,
        diagnostics=diagnostics,
    )


def structure_proximal_target(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    tau2: float,
    lambda_structure: float,
    structure_sigma: float,
    prox_steps: int,
    prox_step_size: float,
    target_mode: str = "projected_mu",
) -> StructureProxResult:
    """Build a detached RGB structure-proximal target."""

    validate_image_tensor(mu, "mu")
    if measurement.shape != mu.shape:
        raise ValueError("measurement must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    with torch.no_grad():
        target_structure = _target_structure(
            mu, measurement, mask_known, structure_sigma, target_mode
        ).detach()
    return structure_proximal_target_from_structure(
        mu=mu,
        measurement=measurement,
        mask_known=mask_known,
        target_structure=target_structure,
        tau2=tau2,
        lambda_structure=lambda_structure,
        structure_sigma=structure_sigma,
        prox_steps=prox_steps,
        prox_step_size=prox_step_size,
    )
