"""Manufactured fields and synthetic structures for BCNS validation."""

import math
from typing import Tuple

import torch

from .operators import laplacian_5pt


def _grid(size: int, h: float, device=None, dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    if size <= 1:
        raise ValueError("size must be greater than 1.")
    coords = torch.arange(size, device=device, dtype=dtype) * h
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def make_poisson_manufactured_solution(
    size: int,
    h: float,
    pattern: str = "smooth_mixed",
    device=None,
    dtype=torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``I_gt`` and discrete ``w_gt = Delta_h I_gt``."""

    xx, yy = _grid(size, h, device=device, dtype=dtype)
    if pattern == "smooth_mixed":
        intensity = torch.sin(math.pi * xx) * torch.sin(2.0 * math.pi * yy) + 0.3 * xx + 0.2 * yy
    elif pattern == "quadratic":
        intensity = 0.5 + 0.1 * xx - 0.2 * yy + 0.7 * xx * xx - 0.3 * xx * yy + 0.4 * yy * yy
    elif pattern == "eigenmode":
        length = h * (size - 1)
        intensity = torch.sin(math.pi * xx / length) * torch.sin(math.pi * yy / length)
    else:
        raise ValueError(f"Unsupported manufactured pattern {pattern!r}.")
    intensity = intensity.view(1, 1, size, size)
    return intensity, laplacian_5pt(intensity, h)


def make_crossing_line_structure(
    size: int = 64,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Return a scalar image containing two crossing soft lines."""

    coords = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    line1 = torch.exp(-((yy - 0.35 * xx) ** 2) / 0.0025)
    line2 = torch.exp(-((yy + 0.55 * xx) ** 2) / 0.0025)
    return torch.clamp(line1 + line2, 0.0, 1.0).view(1, 1, size, size)


def make_curved_edge_structure(
    size: int = 64,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Return a scalar image with a smooth curved edge."""

    coords = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    curve = yy - 0.35 * torch.sin(math.pi * xx)
    return torch.sigmoid(-35.0 * curve).view(1, 1, size, size)


def make_text_like_structure(
    size: int = 64,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Return a simple text-like bar pattern."""

    img = torch.zeros((1, 1, size, size), device=device, dtype=dtype)
    t = max(1, size // 24)
    img[..., size // 4 : 3 * size // 4, size // 4 : size // 4 + t] = 1
    img[..., size // 4 : size // 4 + t, size // 4 : size // 2] = 1
    img[..., size // 2 : size // 2 + t, size // 4 : size // 2] = 1
    img[..., size // 4 : 3 * size // 4, 3 * size // 5 : 3 * size // 5 + t] = 1
    img[..., size // 4 : size // 4 + t, 3 * size // 5 : 4 * size // 5] = 1
    img[..., size // 2 : size // 2 + t, 3 * size // 5 : 4 * size // 5] = 1
    return img
