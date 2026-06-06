"""Target-guidance losses for the BCNS DPS skeleton."""

import torch

from .dps_adapter import hole_from_known, masked_mean_square, validate_image_tensor, validate_mask_tensor
from .proximal import structure_image


def _as_scalar_mask(mask: torch.Tensor, ref: torch.Tensor, name: str = "mask") -> torch.Tensor:
    validate_mask_tensor(mask, ref, name)
    if mask.shape[1] == 1:
        return mask
    return mask[:, :1, :, :]


def _zero_like_ref(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=ref.dtype, device=ref.device)


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


def target_displacement_stats(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask_known: torch.Tensor,
) -> dict:
    """Return full/known/hole target MSE and RMS displacement diagnostics."""

    validate_image_tensor(mu, "mu")
    if target.shape != mu.shape:
        raise ValueError("target must have the same shape as mu.")
    validate_mask_tensor(mask_known, mu, "mask_known")
    diff = mu - target.detach()
    known_mse = masked_mean_square(diff, mask_known)
    hole_mse = masked_mean_square(diff, hole_from_known(mask_known))
    full_mse = (diff ** 2).mean()
    return {
        "target_disp_full": torch.sqrt(full_mse.clamp_min(0.0)),
        "target_disp_known": torch.sqrt(known_mse.clamp_min(0.0)),
        "target_disp_hole": torch.sqrt(hole_mse.clamp_min(0.0)),
        "target_mse_full": full_mse,
        "target_mse_known": known_mse,
        "target_mse_hole": hole_mse,
    }


def structure_displacement_loss(
    mu: torch.Tensor,
    target_structure: torch.Tensor,
    mask_known: torch.Tensor,
    structure_sigma: float,
    weight: float,
) -> tuple:
    """Return detached scalar-structure target loss and diagnostics."""

    validate_image_tensor(mu, "mu")
    if structure_sigma < 0:
        raise ValueError("structure_sigma must be non-negative.")
    weight = float(weight)
    zero = _zero_like_ref(mu)
    if target_structure is None or weight == 0.0:
        return zero, {"structure_loss": zero, "structure_disp_hole": zero}
    if weight < 0:
        raise ValueError("weight must be non-negative.")
    if not torch.is_tensor(target_structure):
        raise TypeError("target_structure must be a torch.Tensor or None.")
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
    scalar_mask_ref = target_structure
    mask_scalar = _as_scalar_mask(mask_known, mu, "mask_known").to(dtype=mu.dtype)
    validate_mask_tensor(mask_scalar, scalar_mask_ref, "mask_known")
    hole = hole_from_known(mask_scalar).to(dtype=mu.dtype)
    delta = structure_image(mu, structure_sigma) - target_structure.detach()
    mse = masked_mean_square(delta, hole)
    loss = 0.5 * weight * mse
    return loss, {
        "structure_loss": loss.detach(),
        "structure_disp_hole": torch.sqrt(mse.detach().clamp_min(0.0)),
    }
