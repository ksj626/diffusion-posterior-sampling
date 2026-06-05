"""Structure-image utilities for BCNS boundary estimation."""

import math

import torch
import torch.nn.functional as F

from .operators import _validate_scalar_field


def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, 3, H, W]`` RGB to luminance without changing value scale."""

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}.")
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
    return (rgb * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


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
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype or torch.float32)
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a scalar field with replicated-boundary Gaussian convolution."""

    _validate_scalar_field(x)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if sigma == 0:
        return x
    kernel = gaussian_kernel_2d(sigma, device=x.device, dtype=x.dtype)
    pad_y = kernel.shape[-2] // 2
    pad_x = kernel.shape[-1] // 2
    padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    return F.conv2d(padded, kernel)


def masked_normalized_gaussian_smooth(
    observed_luminance: torch.Tensor,
    mask_known: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ``(G * (M * y)) / (G * M + eps)`` from observed known pixels."""

    _validate_scalar_field(observed_luminance, "observed_luminance")
    if mask_known.shape != observed_luminance.shape:
        raise ValueError("mask_known must match observed_luminance shape.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if sigma == 0:
        return observed_luminance
    weighted = gaussian_smooth(observed_luminance * mask_known.to(observed_luminance.dtype), sigma)
    weights = gaussian_smooth(mask_known.to(observed_luminance.dtype), sigma)
    return weighted / weights.clamp_min(eps)
