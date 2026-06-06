"""Structural diagnostics for BCNS-DPS inpainting results."""

import math
from typing import Dict

import torch
import torch.nn.functional as F

from .dps_adapter import hole_from_known, validate_image_tensor, validate_mask_tensor
from .navier_stokes import ns_rhs
from .operators import gradient_central, laplacian_5pt, masked_l2_norm
from .proximal import gaussian_smooth_depthwise, structure_image


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype, device=x.device)
    if mask_f.shape[1] == 1 and x.shape[1] != 1:
        mask_f = mask_f.expand_as(x)
    denom = mask_f.sum().clamp_min(torch.finfo(x.dtype).eps)
    return (x * mask_f).sum() / denom


def luminance_structure(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Return a scalar smoothed structure image for RGB or grayscale input."""

    validate_image_tensor(x, "x")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if x.shape[1] == 3:
        return structure_image(x, sigma)
    if x.shape[1] == 1:
        return gaussian_smooth_depthwise(x, sigma)
    raise ValueError("x must have either 1 grayscale channel or 3 RGB channels.")


def boundary_band(mask_known: torch.Tensor, width: int = 3) -> Dict[str, torch.Tensor]:
    """Return known-side, hole-side, and combined masks around the hole boundary."""

    if not torch.is_tensor(mask_known):
        raise TypeError("mask_known must be a torch.Tensor.")
    if mask_known.ndim != 4 or mask_known.shape[1] != 1:
        raise ValueError("mask_known must have shape [B, 1, H, W].")
    if width < 0:
        raise ValueError("width must be non-negative.")
    mask = mask_known.float().clamp(0, 1)
    hole = hole_from_known(mask).clamp(0, 1)
    if width == 0:
        known_band = mask * hole
        hole_band = hole * mask
    else:
        kernel = 2 * int(width) + 1
        known_near_hole = F.max_pool2d(hole, kernel_size=kernel, stride=1, padding=width)
        hole_near_known = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)
        known_band = mask * known_near_hole
        hole_band = hole * hole_near_known
    band = (known_band + hole_band).clamp(0, 1)
    return {"known_band": known_band, "hole_band": hole_band, "band": band}


def seam_mse(pred: torch.Tensor, target: torch.Tensor, mask_known: torch.Tensor, width: int = 3) -> float:
    """Return MSE in a narrow band around the inpainting boundary."""

    validate_image_tensor(pred, "pred")
    if target.shape != pred.shape:
        raise ValueError("target must have the same shape as pred.")
    validate_mask_tensor(mask_known, pred, "mask_known")
    band = boundary_band(mask_known[:, 0:1], width=width)["band"].to(dtype=pred.dtype, device=pred.device)
    value = _masked_mean((pred - target) ** 2, band)
    return _finite_float(value)


def gradient_mismatch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_known: torch.Tensor,
    width: int = 3,
    sigma: float = 1.0,
    h: float = 1.0,
) -> float:
    """Return boundary-band MSE between scalar structure gradients."""

    s_pred = luminance_structure(pred, sigma)
    s_target = luminance_structure(target, sigma)
    validate_mask_tensor(mask_known, pred, "mask_known")
    gx_pred, gy_pred = gradient_central(s_pred, h)
    gx_target, gy_target = gradient_central(s_target, h)
    band = boundary_band(mask_known[:, 0:1], width=width)["band"].to(dtype=pred.dtype, device=pred.device)
    value = _masked_mean((gx_pred - gx_target) ** 2 + (gy_pred - gy_target) ** 2, band)
    return _finite_float(value)


def isophote_angle_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_known: torch.Tensor,
    width: int = 3,
    sigma: float = 1.0,
    h: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Return boundary-band isophote mismatch as ``1 - cos(theta)``."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    s_pred = luminance_structure(pred, sigma)
    s_target = luminance_structure(target, sigma)
    validate_mask_tensor(mask_known, pred, "mask_known")
    gx_pred, gy_pred = gradient_central(s_pred, h)
    gx_target, gy_target = gradient_central(s_target, h)
    vx_pred, vy_pred = -gy_pred, gx_pred
    vx_target, vy_target = -gy_target, gx_target
    mag_pred = torch.sqrt(vx_pred * vx_pred + vy_pred * vy_pred)
    mag_target = torch.sqrt(vx_target * vx_target + vy_target * vy_target)
    dot = vx_pred * vx_target + vy_pred * vy_target
    cos = (dot / ((mag_pred + eps) * (mag_target + eps))).clamp(-1.0, 1.0)
    error = 1.0 - cos
    both_flat = (mag_pred <= eps) & (mag_target <= eps)
    error = torch.where(both_flat, torch.zeros_like(error), error)
    band = boundary_band(mask_known[:, 0:1], width=width)["band"].to(dtype=pred.dtype, device=pred.device)
    value = _masked_mean(error, band)
    return _finite_float(value)


def laplacian_mismatch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_known: torch.Tensor,
    width: int = 3,
    sigma: float = 1.0,
    h: float = 1.0,
) -> float:
    """Return boundary-band MSE between scalar structure Laplacians."""

    s_pred = luminance_structure(pred, sigma)
    s_target = luminance_structure(target, sigma)
    validate_mask_tensor(mask_known, pred, "mask_known")
    lap_pred = laplacian_5pt(s_pred, h)
    lap_target = laplacian_5pt(s_target, h)
    band = boundary_band(mask_known[:, 0:1], width=width)["band"].to(dtype=pred.dtype, device=pred.device)
    value = _masked_mean((lap_pred - lap_target) ** 2, band)
    return _finite_float(value)


def ns_residual_metric(
    x: torch.Tensor,
    mask_known: torch.Tensor,
    nu: float,
    kappa: float,
    smoothing_sigma: float,
    sigma: float = 1.0,
    h: float = 1.0,
) -> float:
    """Return RMS Navier-Stokes structural RHS norm inside the hole."""

    structure = luminance_structure(x, sigma)
    validate_mask_tensor(mask_known, x, "mask_known")
    mask_unknown = hole_from_known(mask_known[:, 0:1]).to(dtype=structure.dtype, device=structure.device)
    vorticity = laplacian_5pt(structure, h)
    rhs, _ = ns_rhs(structure, vorticity, nu=nu, kappa=kappa, smoothing_sigma=smoothing_sigma, h=h)
    value = masked_l2_norm(rhs, mask_unknown)
    return _finite_float(value)


def _finite_float(value: torch.Tensor) -> float:
    out = float(value.detach().item())
    if not math.isfinite(out):
        return 0.0
    return out
