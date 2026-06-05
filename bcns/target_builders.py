"""Debug target builders for BCNS target-guidance conditioning."""

from dataclasses import dataclass
from typing import Dict

import torch

from .dps_adapter import hard_project_clean, hole_from_known, validate_image_tensor, validate_mask_tensor
from .proximal import structure_proximal_target


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


class StructureProxTargetBuilder:
    """Structure-proximal target builder for BCNS Step 1."""

    def __init__(
        self,
        tau2: float = 1.0,
        lambda_structure: float = 0.1,
        structure_sigma: float = 1.0,
        prox_steps: int = 1,
        prox_step_size: float = 0.1,
        target_mode: str = "projected_mu",
    ):
        self.tau2 = tau2
        self.lambda_structure = lambda_structure
        self.structure_sigma = structure_sigma
        self.prox_steps = prox_steps
        self.prox_step_size = prox_step_size
        self.target_mode = target_mode

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        result = structure_proximal_target(
            mu=mu,
            measurement=measurement,
            mask_known=mask_known,
            tau2=kwargs.get("tau2", self.tau2),
            lambda_structure=self.lambda_structure,
            structure_sigma=self.structure_sigma,
            prox_steps=self.prox_steps,
            prox_step_size=self.prox_step_size,
            target_mode=self.target_mode,
        )
        diagnostics = {
            "target_builder": "structure_prox",
            "lambda_structure": self.lambda_structure,
            "structure_sigma": self.structure_sigma,
            "prox_steps": self.prox_steps,
            "prox_step_size": self.prox_step_size,
            "target_mode": self.target_mode,
            "prox_final_loss": result.diagnostics["prox_final_loss"],
            "target_disp": result.diagnostics["target_disp"],
        }
        return TargetBuildResult(target=result.target, diagnostics=diagnostics)


def get_target_builder(name: str, **kwargs):
    """Return a target builder by name."""

    builders = {
        "identity": IdentityTargetBuilder,
        "hard_projection": HardProjectionTargetBuilder,
        "simple_hole_shift": SimpleHoleShiftTargetBuilder,
        "structure_prox": StructureProxTargetBuilder,
    }
    if name not in builders:
        raise ValueError(f"Unsupported target builder {name!r}; expected one of {sorted(builders)}.")
    return builders[name](**kwargs)
