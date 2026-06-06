"""Synthetic missing-mask generation and mask geometry utilities.

The mask convention is fixed throughout BCNS Step 1: ``mask_unknown == 1``
means the pixel belongs to the missing region Omega.
"""

import math
import random
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


def _line_segment_mask(
    height: int,
    width: int,
    start: Tuple[float, float],
    end: Tuple[float, float],
    thickness: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype or torch.float32),
        torch.arange(width, device=device, dtype=dtype or torch.float32),
        indexing="ij",
    )
    y0, x0 = start
    y1, x1 = end
    dy = float(y1 - y0)
    dx = float(x1 - x0)
    denom = dx * dx + dy * dy
    if denom == 0:
        dist2 = (yy - float(y0)) ** 2 + (xx - float(x0)) ** 2
    else:
        t = ((xx - float(x0)) * dx + (yy - float(y0)) * dy) / denom
        t = t.clamp(0.0, 1.0)
        proj_x = float(x0) + t * dx
        proj_y = float(y0) + t * dy
        dist2 = (yy - proj_y) ** 2 + (xx - proj_x) ** 2
    radius = max(0.5, float(thickness) / 2.0)
    return (dist2 <= radius * radius).to(dtype=dtype or torch.float32).view(1, 1, height, width)


def make_thick_scratch_mask(
    height: int,
    width: int,
    thickness: int = 12,
    angle: Optional[float] = None,
    length_frac: float = 1.2,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return ``[1, 1, H, W]`` with thick scratch-like missing strokes."""

    _validate_size(height, width)
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    if length_frac <= 0:
        raise ValueError("length_frac must be positive.")
    if angle is None:
        angle = -35.0
    theta = math.radians(float(angle))
    length = float(max(height, width)) * float(length_frac)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    dy = math.sin(theta) * length / 2.0
    dx = math.cos(theta) * length / 2.0
    mask = _line_segment_mask(
        height,
        width,
        (cy - dy, cx - dx),
        (cy + dy, cx + dx),
        thickness,
        device=device,
        dtype=dtype,
    )
    offset = float(thickness) * 1.8
    mask = torch.maximum(
        mask,
        _line_segment_mask(
            height,
            width,
            (cy - dy - offset, cx - dx + offset * 0.4),
            (cy + dy - offset, cx + dx + offset * 0.4),
            max(1, thickness // 2),
            device=device,
            dtype=dtype,
        ),
    )
    return mask.to(dtype=dtype or torch.float32)


_BLOCK_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
}


def make_text_like_mask(
    height: int,
    width: int,
    text: str = "TEXT",
    font_scale: float = 1.0,
    thickness: int = 12,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return ``[1, 1, H, W]`` with block-letter text-like holes."""

    _validate_size(height, width)
    if font_scale <= 0:
        raise ValueError("font_scale must be positive.")
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    text = (text or "TEXT").upper()
    cell = max(2, int(round(min(height, width) / 32.0 * float(font_scale))))
    letter_w = 5 * cell
    letter_h = 7 * cell
    spacing = max(1, cell)
    total_w = len(text) * letter_w + max(0, len(text) - 1) * spacing
    top = max(0, (height - letter_h) // 2)
    left = max(0, (width - total_w) // 2)
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype or torch.float32)
    for idx, char in enumerate(text):
        pattern = _BLOCK_FONT.get(char, _BLOCK_FONT["T"])
        x0 = left + idx * (letter_w + spacing)
        for row, line in enumerate(pattern):
            for col, value in enumerate(line):
                if value != "1":
                    continue
                y_start = top + row * cell
                x_start = x0 + col * cell
                y_end = min(height, y_start + cell)
                x_end = min(width, x_start + cell)
                if y_start < height and x_start < width:
                    mask[..., y_start:y_end, x_start:x_end] = 1
    dilate_iters = max(0, int(round(float(thickness) / 8.0)) - 1)
    return _dilate(mask, dilate_iters) if dilate_iters else mask


def make_freeform_medium_mask(
    height: int,
    width: int,
    num_strokes: int = 4,
    thickness_range: Tuple[int, int] = (8, 24),
    length_range: Tuple[int, int] = (40, 140),
    seed: Optional[int] = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return ``[1, 1, H, W]`` with medium free-form missing strokes."""

    _validate_size(height, width)
    if num_strokes <= 0:
        raise ValueError("num_strokes must be positive.")
    lo_t, hi_t = thickness_range
    lo_l, hi_l = length_range
    if lo_t <= 0 or hi_t < lo_t:
        raise ValueError("thickness_range must be positive and ordered.")
    if lo_l <= 0 or hi_l < lo_l:
        raise ValueError("length_range must be positive and ordered.")
    rng = random.Random(seed)
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype or torch.float32)
    for _ in range(num_strokes):
        thickness = rng.randint(lo_t, hi_t)
        length = float(rng.randint(lo_l, hi_l))
        y0 = rng.uniform(0.15 * height, 0.85 * height)
        x0 = rng.uniform(0.15 * width, 0.85 * width)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        y1 = y0 + math.sin(angle) * length
        x1 = x0 + math.cos(angle) * length
        mask = torch.maximum(
            mask,
            _line_segment_mask(height, width, (y0, x0), (y1, x1), thickness, device=device, dtype=dtype),
        )
    return mask


def make_center_box_unknown_mask(
    height: int,
    width: int,
    box_size: int = 96,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return ``[1, 1, H, W]`` with a center-box unknown mask."""

    return make_center_box_mask(height, width, box_size, box_size, device=device, dtype=dtype)


def mask_metadata(mask_unknown: torch.Tensor) -> dict:
    """Return approximate geometry metadata for a binary unknown mask."""

    _validate_mask(mask_unknown)
    mask = mask_unknown.to(dtype=torch.float32)
    hole_ratio = float(mask.mean().item())
    known_ratio = 1.0 - hole_ratio
    kernel = torch.ones((1, 1, 3, 3), dtype=mask.dtype, device=mask.device)
    neighbor_count = F.conv2d(mask, kernel, padding=1)
    boundary = ((mask > 0) & (neighbor_count < 9)).to(mask.dtype)
    boundary_length_approx = float(boundary.sum().item())
    if bool((mask > 0).any().item()):
        coords = torch.nonzero(mask[0, 0] > 0, as_tuple=False)
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())
        bbox_area_ratio = float((y_max - y_min + 1) * (x_max - x_min + 1)) / float(
            mask.shape[-2] * mask.shape[-1]
        )
    else:
        bbox_area_ratio = 0.0
    starts = (mask > 0) & (F.pad(mask[..., :-1, :], (0, 0, 1, 0)) <= 0) & (
        F.pad(mask[..., :, :-1], (1, 0, 0, 0)) <= 0
    )
    num_components_approx = int(starts.sum().item())
    return {
        "hole_ratio": hole_ratio,
        "known_ratio": known_ratio,
        "boundary_length_approx": boundary_length_approx,
        "num_components_approx": num_components_approx,
        "bbox_area_ratio": bbox_area_ratio,
    }


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
