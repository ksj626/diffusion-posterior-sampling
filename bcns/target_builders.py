"""Debug and structure target builders for BCNS target-guidance conditioning."""

from dataclasses import dataclass

import torch

from .dps_adapter import hard_project_clean, hole_from_known, validate_image_tensor, validate_mask_tensor
from .poisson_targets import build_poisson_structure_target
from .proximal import structure_proximal_target, structure_proximal_target_from_structure


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


class PoissonStructureTargetBuilder:
    """Harmonic/Poisson scalar structure target followed by RGB proximal target."""

    def __init__(
        self,
        target_builder: str = "poisson_structure",
        tau2: float = 1.0,
        lambda_structure: float = 0.05,
        structure_sigma: float = 1.0,
        prox_steps: int = 1,
        prox_step_size: float = 0.05,
        rhs_mode: str = None,
        boundary_mode: str = "normalized_known_smooth",
        poisson_method: str = "sor_rb",
        poisson_max_iter: int = 200,
        poisson_tol: float = 1e-4,
        poisson_omega: float = 1.7,
        poisson_h: float = 1.0,
        record_history: bool = True,
    ):
        if target_builder not in ("harmonic_structure", "poisson_structure"):
            raise ValueError("target_builder must be 'harmonic_structure' or 'poisson_structure'.")
        self.target_builder = target_builder
        self.tau2 = tau2
        self.lambda_structure = lambda_structure
        self.structure_sigma = structure_sigma
        self.prox_steps = prox_steps
        self.prox_step_size = prox_step_size
        if rhs_mode is None:
            rhs_mode = "zero" if target_builder == "harmonic_structure" else "projected_mu_laplacian"
        self.rhs_mode = rhs_mode
        self.boundary_mode = boundary_mode
        self.poisson_method = poisson_method
        self.poisson_max_iter = poisson_max_iter
        self.poisson_tol = poisson_tol
        self.poisson_omega = poisson_omega
        self.poisson_h = poisson_h
        self.record_history = record_history

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        tau2 = kwargs.get("tau2", self.tau2)
        poisson_result = build_poisson_structure_target(
            mu=mu,
            measurement=measurement,
            mask_known=mask_known,
            sigma=self.structure_sigma,
            rhs_mode=self.rhs_mode,
            boundary_mode=self.boundary_mode,
            poisson_method=self.poisson_method,
            poisson_max_iter=self.poisson_max_iter,
            poisson_tol=self.poisson_tol,
            poisson_omega=self.poisson_omega,
            poisson_h=self.poisson_h,
            record_history=self.record_history,
        )
        prox_result = structure_proximal_target_from_structure(
            mu=mu,
            measurement=measurement,
            mask_known=mask_known,
            target_structure=poisson_result.structure.to(dtype=mu.dtype, device=mu.device),
            tau2=tau2,
            lambda_structure=self.lambda_structure,
            structure_sigma=self.structure_sigma,
            prox_steps=self.prox_steps,
            prox_step_size=self.prox_step_size,
        )
        diagnostics = {
            "target_builder": self.target_builder,
            "lambda_structure": self.lambda_structure,
            "structure_sigma": self.structure_sigma,
            "prox_steps": self.prox_steps,
            "prox_step_size": self.prox_step_size,
            "prox_final_loss": prox_result.diagnostics["prox_final_loss"],
            "target_disp": prox_result.diagnostics["target_disp"],
        }
        diagnostics.update(poisson_result.diagnostics)
        diagnostics["poisson_structure_disp"] = poisson_result.diagnostics.get("target_disp")
        diagnostics["target_disp"] = prox_result.diagnostics["target_disp"]
        return TargetBuildResult(target=prox_result.target, diagnostics=diagnostics)


def get_target_builder(name: str, **kwargs):
    """Return a target builder by name."""

    builders = {
        "identity": IdentityTargetBuilder,
        "hard_projection": HardProjectionTargetBuilder,
        "simple_hole_shift": SimpleHoleShiftTargetBuilder,
        "structure_prox": StructureProxTargetBuilder,
    }
    if name == "harmonic_structure":
        return PoissonStructureTargetBuilder(target_builder="harmonic_structure", **kwargs)
    if name == "poisson_structure":
        return PoissonStructureTargetBuilder(target_builder="poisson_structure", **kwargs)
    if name not in builders:
        expected = sorted(list(builders) + ["harmonic_structure", "poisson_structure"])
        raise ValueError(f"Unsupported target builder {name!r}; expected one of {expected}.")
    return builders[name](**kwargs)
