"""Reusable evaluation metrics for BCNS-DPS inpainting outputs."""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .dps_adapter import (
    hole_from_known,
    masked_mean_square,
    split_known_hole_mse,
    validate_image_tensor,
    validate_mask_tensor,
)


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    validate_image_tensor(pred, "pred")
    if target.shape != pred.shape:
        raise ValueError("target must have the same shape as pred.")
    if target.device != pred.device:
        raise ValueError("target must be on the same device as pred.")
    if target.dtype != pred.dtype:
        raise ValueError("target must have the same dtype as pred.")


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    validate_mask_tensor(mask, x, "mask")
    mask_f = mask.to(dtype=x.dtype, device=x.device).expand_as(x)
    denom = mask_f.sum().clamp_min(torch.finfo(x.dtype).eps)
    return (x * mask_f).sum() / denom


def psnr(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, data_range: float = 2.0) -> float:
    """Return PSNR for DPS-scale tensors, capped at 100 dB for exact matches."""

    _validate_pair(pred, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive.")
    diff2 = (pred - target) ** 2
    mse = _masked_mean(diff2, mask)
    mse_value = float(mse.detach().item())
    if mse_value <= 0.0:
        return 100.0
    value = 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(mse_value)
    if not math.isfinite(value):
        return 100.0
    return float(value)


def ssim_simple(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 2.0,
    window_size: int = 11,
) -> float:
    """Return a lightweight SSIM score using depthwise torch convolution."""

    _validate_pair(pred, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive.")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    channels = pred.shape[1]
    weight = torch.ones(
        (channels, 1, window_size, window_size),
        dtype=pred.dtype,
        device=pred.device,
    ) / float(window_size * window_size)
    pad = window_size // 2
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="replicate")
    target_pad = F.pad(target, (pad, pad, pad, pad), mode="replicate")
    mu_x = F.conv2d(pred_pad, weight, groups=channels)
    mu_y = F.conv2d(target_pad, weight, groups=channels)

    sigma_x = F.conv2d(pred_pad * pred_pad, weight, groups=channels) - mu_x * mu_x
    sigma_y = F.conv2d(target_pad * target_pad, weight, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(pred_pad * target_pad, weight, groups=channels) - mu_x * mu_y

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(torch.finfo(pred.dtype).eps)
    value = float(_masked_mean(ssim_map, mask).detach().item())
    if not math.isfinite(value):
        return 0.0
    return value


def mse_regions(pred: torch.Tensor, target: torch.Tensor, mask_known: torch.Tensor) -> dict:
    """Return known, hole, and full MSE as Python floats."""

    values = split_known_hole_mse(pred, target, mask_known)
    return {key: float(value.detach().item()) for key, value in values.items()}


def mae_regions(pred: torch.Tensor, target: torch.Tensor, mask_known: torch.Tensor) -> dict:
    """Return known, hole, and full MAE as Python floats."""

    _validate_pair(pred, target)
    validate_mask_tensor(mask_known, pred, "mask_known")
    diff = (pred - target).abs()
    known = mask_known.to(dtype=pred.dtype, device=pred.device)
    hole = hole_from_known(mask_known).to(dtype=pred.dtype, device=pred.device)
    known_mae = _masked_mean(diff, known)
    hole_mae = _masked_mean(diff, hole)
    full_mae = diff.mean()
    return {
        "known_mae": float(known_mae.detach().item()),
        "hole_mae": float(hole_mae.detach().item()),
        "full_mae": float(full_mae.detach().item()),
    }


def lpips_optional(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> Optional[float]:
    """Return LPIPS if the optional dependency is installed, otherwise ``None``."""

    _validate_pair(pred, target)
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    try:
        pred_eval = pred
        target_eval = target
        if pred_eval.shape[1] == 1:
            pred_eval = pred_eval.repeat(1, 3, 1, 1)
            target_eval = target_eval.repeat(1, 3, 1, 1)
        loss_fn = lpips.LPIPS(net=net).to(device=pred.device)
        loss_fn.eval()
        with torch.no_grad():
            value = loss_fn(pred_eval.clamp(-1, 1), target_eval.clamp(-1, 1)).mean()
        return float(value.detach().item())
    except Exception:
        return None
