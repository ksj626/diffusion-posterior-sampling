"""Direct scalar-to-RGB luminance lift targets for BCNS debugging."""

import torch

from .dps_adapter import (
    hard_project_clean,
    hole_from_known,
    validate_image_tensor,
    validate_mask_tensor,
)
from .proximal import structure_image


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


def luminance_lift_rgb_target(
    mu: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
    target_structure: torch.Tensor,
    structure_sigma: float = 1.0,
    lift_scale: float = 1.0,
    mode: str = "equal_rgb",
) -> torch.Tensor:
    """Lift a detached scalar structure residual into an RGB target."""

    validate_image_tensor(mu, "mu")
    if mu.shape[1] != 3:
        raise ValueError(f"mu must have 3 RGB channels, got {mu.shape[1]}.")
    if measurement.shape != mu.shape:
        raise ValueError("measurement must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    _validate_target_structure(target_structure, mu)
    if structure_sigma < 0:
        raise ValueError("structure_sigma must be non-negative.")

    with torch.no_grad():
        hole = hole_from_known(mask_known).to(dtype=mu.dtype)
        delta_i = (target_structure.detach() - structure_image(mu, structure_sigma)) * hole
        lift_scale = float(lift_scale)
        if mode == "equal_rgb":
            delta_rgb = delta_i.repeat(1, 3, 1, 1)
        elif mode == "luma_weights":
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=mu.dtype, device=mu.device)
            weights = weights / (weights.pow(2).sum() + torch.finfo(mu.dtype).eps)
            delta_rgb = delta_i * weights.view(1, 3, 1, 1)
        else:
            raise ValueError("mode must be one of 'equal_rgb' or 'luma_weights'.")
        target = mu + lift_scale * delta_rgb
        target = hard_project_clean(target, measurement, mask_known)
    return target.detach()
