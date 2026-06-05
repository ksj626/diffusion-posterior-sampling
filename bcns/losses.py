"""Target-guidance losses for the BCNS DPS skeleton."""

import torch

from .dps_adapter import hole_from_known, masked_mean_square, validate_image_tensor, validate_mask_tensor


def target_discrepancy_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask_known: torch.Tensor,
    tau2: float,
) -> torch.Tensor:
    """Return ``0.5 / tau2 * mean_hole ||mu - stopgrad(target)||^2``."""

    validate_image_tensor(mu, "mu")
    if target.shape != mu.shape:
        raise ValueError("target must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    if tau2 <= 0:
        raise ValueError("tau2 must be positive.")
    hole = hole_from_known(mask_known).to(dtype=mu.dtype)
    return 0.5 / tau2 * masked_mean_square(mu - target.detach(), hole)
