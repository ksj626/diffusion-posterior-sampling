"""Visualization helpers for BCNS Step 1 targets and smoke outputs."""

from pathlib import Path
from typing import List, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


PathLike = Union[str, Path]


def _as_chw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError("Only batch size 1 tensors can be visualized.")
        x = x[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected [1, C, H, W], [C, H, W], or [H, W], got {tuple(x.shape)}.")
    return x.detach().cpu()


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """Convert DPS image tensor ``[-1, 1]`` to uint8 HWC."""

    chw = _as_chw(x).float()
    if chw.shape[0] == 1:
        hw = ((chw[0].clamp(-1, 1) + 1.0) * 127.5).round().byte().numpy()
        return np.stack([hw, hw, hw], axis=-1)
    if chw.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {chw.shape[0]}.")
    hwc = chw.permute(1, 2, 0).clamp(-1, 1)
    return ((hwc + 1.0) * 127.5).round().byte().numpy()


def save_tensor_image(x: torch.Tensor, path: PathLike) -> None:
    """Save a DPS image tensor as PNG."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_image(x)).save(output)


def save_mask_image(mask_known: torch.Tensor, path: PathLike) -> None:
    """Save a known/hole mask where white is known and black is hole."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask = _as_chw(mask_known.float())[0].clamp(0, 1)
    arr = (mask * 255.0).round().byte().numpy()
    Image.fromarray(arr, mode="L").save(output)


def save_heatmap(x: torch.Tensor, path: PathLike, normalize: bool = True) -> None:
    """Save a scalar tensor heatmap."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = _as_chw(x.float())[0].numpy()
    if normalize:
        lo = float(arr.min())
        hi = float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap="magma")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_contact_sheet(
    image_paths: Sequence[PathLike],
    labels: Sequence[str],
    output_path: PathLike,
    cols: int = 4,
) -> None:
    """Create a labeled contact sheet from image paths."""

    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length.")
    if cols <= 0:
        raise ValueError("cols must be positive.")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("At least one image is required.")
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images) + 24
    rows = int(np.ceil(len(images) / float(cols)))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * cell_h
        sheet.paste(image, (x0, y0 + 24))
        draw.text((x0 + 4, y0 + 4), label, fill=(0, 0, 0))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
