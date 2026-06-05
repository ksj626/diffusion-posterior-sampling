"""Debug target builders for BCNS target-guidance conditioning."""

from dataclasses import dataclass
from typing import Dict

import torch

from .dps_adapter import hard_project_clean, hole_from_known, validate_image_tensor, validate_mask_tensor


@dataclass
class TargetBuildResult:
    """Target tensor plus lightweight diagnostics."""

    target: torch.Tensor
    diagnostics: dict


class IdentityTargetBuilder:
    """Debug target builder where ``target = mu`` and guidance should vanish."""

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        validate_image_tensor(mu, "mu")
        if measurement.shape != mu.shape:
            raise ValueError("measurement must have the same shape as mu.")
        validate_mask_tensor(mask_known, mu, "mask_known")
        return TargetBuildResult(target=mu, diagnostics={"target_builder": "identity"})


class HardProjectionTargetBuilder:
    """Clean-space projection debug target for noiseless inpainting."""

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        target = hard_project_clean(mu, measurement, mask_known)
        return TargetBuildResult(target=target, diagnostics={"target_builder": "hard_projection"})


class SimpleHoleShiftTargetBuilder:
    """Synthetic target that shifts only hole pixels for autograd smoke tests."""

    def __init__(self, shift: float = 0.01):
        self.shift = float(shift)

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        validate_image_tensor(mu, "mu")
        if measurement.shape != mu.shape:
            raise ValueError("measurement must have the same shape as mu.")
        validate_mask_tensor(mask_known, mu, "mask_known")
        hole = hole_from_known(mask_known).to(dtype=mu.dtype)
        shifted = mu + self.shift * hole
        target = hard_project_clean(shifted, measurement, mask_known)
        return TargetBuildResult(
            target=target,
            diagnostics={"target_builder": "simple_hole_shift", "shift": self.shift},
        )


def get_target_builder(name: str, **kwargs):
    """Return a target builder by name."""

    builders = {
        "identity": IdentityTargetBuilder,
        "hard_projection": HardProjectionTargetBuilder,
        "simple_hole_shift": SimpleHoleShiftTargetBuilder,
    }
    if name not in builders:
        raise ValueError(f"Unsupported target builder {name!r}; expected one of {sorted(builders)}.")
    return builders[name](**kwargs)
