"""Adapter utilities between DPS inpainting masks and BCNS hole masks."""

import torch


def validate_image_tensor(x: torch.Tensor, name: str = "tensor") -> None:
    """Validate an image-like tensor with shape ``[B, C, H, W]``."""

    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(x.shape)}.")
    if x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[2] <= 0 or x.shape[3] <= 0:
        raise ValueError(f"{name} must have positive dimensions, got {tuple(x.shape)}.")
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must be a floating point tensor.")


def validate_mask_tensor(mask: torch.Tensor, ref: torch.Tensor, name: str = "mask") -> None:
    """Validate a mask broadcastable as ``[B, 1, H, W]`` or ``[B, C, H, W]``."""

    validate_image_tensor(ref, "ref")
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.ndim != 4:
        raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, C, H, W].")
    if mask.device != ref.device:
        raise ValueError(f"{name} must be on the same device as ref.")
    if not torch.is_floating_point(mask):
        raise TypeError(f"{name} must be a floating point tensor.")
    if mask.shape[0] != ref.shape[0] or mask.shape[-2:] != ref.shape[-2:]:
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} is not broadcastable to ref shape {tuple(ref.shape)}."
        )
    if mask.shape[1] not in (1, ref.shape[1]):
        raise ValueError(
            f"{name} channel dimension must be 1 or {ref.shape[1]}, got {mask.shape[1]}."
        )


def dps_known_to_bcns_unknown(mask_known: torch.Tensor) -> torch.Tensor:
    """Convert DPS known mask, where ``1`` means observed, to BCNS unknown mask."""

    if not torch.is_tensor(mask_known):
        raise TypeError("mask_known must be a torch.Tensor.")
    return 1.0 - mask_known


def bcns_unknown_to_dps_known(mask_unknown: torch.Tensor) -> torch.Tensor:
    """Convert BCNS unknown mask, where ``1`` means missing, to DPS known mask."""

    if not torch.is_tensor(mask_unknown):
        raise TypeError("mask_unknown must be a torch.Tensor.")
    return 1.0 - mask_unknown


def hole_from_known(mask_known: torch.Tensor) -> torch.Tensor:
    """Return the inpainting hole mask from a DPS known mask."""

    return dps_known_to_bcns_unknown(mask_known)


def hard_project_clean(
    x_clean: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
) -> torch.Tensor:
    """Apply noiseless clean-space inpainting projection."""

    validate_image_tensor(x_clean, "x_clean")
    if measurement.shape != x_clean.shape:
        raise ValueError("measurement must have the same shape as x_clean.")
    validate_mask_tensor(mask_known, x_clean, "mask_known")
    mask = mask_known.to(dtype=x_clean.dtype)
    return mask * measurement + (1.0 - mask) * x_clean


def clean_composite(
    raw: torch.Tensor,
    measurement: torch.Tensor,
    mask_known: torch.Tensor,
) -> torch.Tensor:
    """Return final clean-space inpainting compositing."""

    return hard_project_clean(raw, measurement, mask_known)


def project_known_noisy(
    x_t: torch.Tensor,
    noisy_measurement: torch.Tensor,
    mask_known: torch.Tensor,
) -> torch.Tensor:
    """Project known pixels to noisy measurement values during sampling."""

    validate_image_tensor(x_t, "x_t")
    if noisy_measurement.shape != x_t.shape:
        raise ValueError("noisy_measurement must have the same shape as x_t.")
    validate_mask_tensor(mask_known, x_t, "mask_known")
    mask = mask_known.to(dtype=x_t.dtype)
    return mask * noisy_measurement + (1.0 - mask) * x_t


def masked_mean_square(
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return the mean of ``x**2`` over active masked entries."""

    validate_image_tensor(x, "x")
    validate_mask_tensor(mask, x, "mask")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    mask_f = mask.to(dtype=x.dtype)
    active = mask_f.expand_as(x)
    denom = active.sum().clamp_min(torch.as_tensor(eps, dtype=x.dtype, device=x.device))
    return ((x * active) ** 2).sum() / denom

def split_known_hole_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_known: torch.Tensor,
) -> dict:
    """Return known, hole, and full mean-square error scalar tensors."""

    validate_image_tensor(pred, "pred")
    if target.shape != pred.shape:
        raise ValueError("target must have the same shape as pred.")
    validate_mask_tensor(mask_known, pred, "mask_known")
    diff = pred - target
    known_mse = masked_mean_square(diff, mask_known)
    hole_mse = masked_mean_square(diff, hole_from_known(mask_known))
    full_mse = (diff ** 2).mean()
    return {"known_mse": known_mse, "hole_mse": hole_mse, "full_mse": full_mse}

