"""Synthetic missing-mask generation and mask geometry utilities.

The mask convention is fixed throughout BCNS Step 1: ``mask_unknown == 1``
means the pixel belongs to the missing region Omega.
"""

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


Box = Tuple[int, int, int, int]


def _validate_size(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")


def _validate_mask(mask_unknown: torch.Tensor) -> None:
    if mask_unknown.ndim != 4 or mask_unknown.shape[1] != 1:
        raise ValueError(
            f"mask_unknown must have shape [B, 1, H, W], got {tuple(mask_unknown.shape)}."
        )
    values = mask_unknown.detach()
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("mask_unknown must be binary with values 0 or 1.")


def make_center_box_mask(
    height: int,
    width: int,
    box_height: int,
    box_width: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return ``[1, 1, H, W]`` with ones inside a missing center box."""

    _validate_size(height, width)
    if not (0 < box_height <= height and 0 < box_width <= width):
        raise ValueError("box_height and box_width must fit within the image.")
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype or torch.float32)
    top = (height - box_height) // 2
    left = (width - box_width) // 2
    mask[..., top : top + box_height, left : left + box_width] = 1
    return mask


def make_disconnected_box_mask(
    height: int,
    width: int,
    boxes: Optional[Sequence[Box]] = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return separated rectangular missing components.

    Boxes are ``(top, left, box_height, box_width)``. If omitted, two separated
    boxes are placed in opposite quadrants.
    """

    _validate_size(height, width)
    if boxes is None:
        bh = max(2, height // 6)
        bw = max(2, width // 6)
        boxes = ((height // 4 - bh // 2, width // 4 - bw // 2, bh, bw),
                 (3 * height // 4 - bh // 2, 3 * width // 4 - bw // 2, bh, bw))
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype or torch.float32)
    for top, left, box_height, box_width in boxes:
        if top < 0 or left < 0 or top + box_height > height or left + box_width > width:
            raise ValueError(f"box {(top, left, box_height, box_width)} is outside the image.")
        mask[..., top : top + box_height, left : left + box_width] = 1
    return mask


def make_thin_scratch_mask(
    height: int,
    width: int,
    start: Optional[Tuple[int, int]] = None,
    end: Optional[Tuple[int, int]] = None,
    thickness: int = 1,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return a synthetic narrow line/scratch missing mask."""

    _validate_size(height, width)
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    if start is None:
        start = (height // 4, width // 5)
    if end is None:
        end = (3 * height // 4, 4 * width // 5)
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype or torch.float32)
    y0, x0 = start
    y1, x1 = end
    n = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    ys = torch.linspace(float(y0), float(y1), n, device=device).round().long()
    xs = torch.linspace(float(x0), float(x1), n, device=device).round().long()
    radius = thickness // 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = (ys + dy).clamp(0, height - 1)
            xx = (xs + dx).clamp(0, width - 1)
            mask[..., yy, xx] = 1
    return mask


def _dilate(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    out = mask
    for _ in range(iterations):
        out = (F.max_pool2d(out, kernel_size=3, stride=1, padding=1) > 0).to(mask.dtype)
    return out


def hole_boundary(mask_unknown: torch.Tensor) -> torch.Tensor:
    """Return pixels inside or immediately adjacent to Omega."""

    _validate_mask(mask_unknown)
    return _dilate(mask_unknown, 1)


def known_collar(mask_unknown: torch.Tensor, width: int) -> torch.Tensor:
    """Return the known-side observed band of requested width outside Omega."""

    _validate_mask(mask_unknown)
    if width <= 0:
        raise ValueError("width must be positive.")
    dilated = _dilate(mask_unknown, width)
    collar = (dilated > 0) & (mask_unknown <= 0)
    return collar.to(dtype=mask_unknown.dtype, device=mask_unknown.device)
