"""Differentiable tensor finite-difference operators.

All scalar fields use shape ``[B, 1, H, W]``. The x direction is the image
column axis and the y direction is the row axis. Boundary derivatives use
one-sided differences; the five-point Laplacian uses replicated edge padding.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def _validate_scalar_field(x: torch.Tensor, name: str = "x") -> None:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(x.shape)}.")
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        raise ValueError(f"{name} must have spatial size at least 2x2.")
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must be a floating point tensor.")


def _validate_h(h: float) -> None:
    if h <= 0:
        raise ValueError(f"h must be positive, got {h!r}.")


def gradient_central(x: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``dx, dy`` for a scalar field using central interior differences."""

    _validate_scalar_field(x)
    _validate_h(h)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) / (2.0 * h)
    dx[..., :, 0] = (x[..., :, 1] - x[..., :, 0]) / h
    dx[..., :, -1] = (x[..., :, -1] - x[..., :, -2]) / h
    dy[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2.0 * h)
    dy[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / h
    dy[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / h
    return dx, dy


def gradient_forward(x: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return forward differences with backward differences on the far boundary."""

    _validate_scalar_field(x)
    _validate_h(h)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, :-1] = (x[..., :, 1:] - x[..., :, :-1]) / h
    dx[..., :, -1] = (x[..., :, -1] - x[..., :, -2]) / h
    dy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]) / h
    dy[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / h
    return dx, dy


def divergence(px: torch.Tensor, py: torch.Tensor, h: float) -> torch.Tensor:
    """Return a backward-difference divergence of vector field ``(px, py)``."""

    _validate_scalar_field(px, "px")
    _validate_scalar_field(py, "py")
    if px.shape != py.shape:
        raise ValueError("px and py must have identical shape.")
    _validate_h(h)
    div = torch.zeros_like(px)
    div[..., :, 0] += px[..., :, 0] / h
    div[..., :, 1:] += (px[..., :, 1:] - px[..., :, :-1]) / h
    div[..., 0, :] += py[..., 0, :] / h
    div[..., 1:, :] += (py[..., 1:, :] - py[..., :-1, :]) / h
    return div


def laplacian_5pt(x: torch.Tensor, h: float) -> torch.Tensor:
    """Return the replicated-edge five-point Laplacian ``Delta_h x``."""

    _validate_scalar_field(x)
    _validate_h(h)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    north = padded[..., :-2, 1:-1]
    south = padded[..., 2:, 1:-1]
    west = padded[..., 1:-1, :-2]
    east = padded[..., 1:-1, 2:]
    return (north + south + west + east - 4.0 * center) / (h * h)


def perpendicular_gradient(x: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``vx=-I_y, vy=I_x`` for ``v = nabla_perp I``."""

    dx, dy = gradient_central(x, h)
    return -dy, dx


def masked_l2_norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return RMS L2 norm over entries where ``mask`` is positive."""

    if x.shape != mask.shape:
        raise ValueError("x and mask must have identical shape.")
    mask_f = mask.to(dtype=x.dtype, device=x.device)
    denom = mask_f.sum().clamp_min(torch.finfo(x.dtype).eps)
    return torch.sqrt(((x * mask_f) ** 2).sum() / denom)
