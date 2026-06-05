"""Small diagnostics used by BCNS target-guidance tests and smoke scripts."""

from typing import Dict

import torch

from .dps_adapter import hole_from_known, validate_image_tensor, validate_mask_tensor


def tensor_norm(x: torch.Tensor) -> float:
    """Return the Euclidean norm as a Python float."""

    validate_image_tensor(x, "x")
    return float(torch.linalg.norm(x.reshape(-1)).item())


def mask_region_means(x: torch.Tensor, mask_known: torch.Tensor) -> Dict[str, float]:
    """Return mean absolute values on known and hole regions."""

    validate_image_tensor(x, "x")
    validate_mask_tensor(mask_known, x, "mask_known")
    known = mask_known.to(dtype=x.dtype).expand_as(x)
    hole = hole_from_known(mask_known).to(dtype=x.dtype).expand_as(x)
    eps = torch.finfo(x.dtype).eps
    return {
        "known_mean_abs": float((x.abs() * known).sum().div(known.sum().clamp_min(eps)).item()),
        "hole_mean_abs": float((x.abs() * hole).sum().div(hole.sum().clamp_min(eps)).item()),
    }


def update_norm(before: torch.Tensor, after: torch.Tensor) -> float:
    """Return ``||after - before||``."""

    if before.shape != after.shape:
        raise ValueError("before and after must have identical shape.")
    return tensor_norm(after - before)
