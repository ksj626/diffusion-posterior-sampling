"""Debug and structure target builders for BCNS target-guidance conditioning."""

from dataclasses import dataclass
from typing import Optional

import torch

from .dps_adapter import (
    hard_project_clean,
    hole_from_known,
    masked_mean_square,
    validate_image_tensor,
    validate_mask_tensor,
)
from .flow_targets import build_flow_structure_target
from .luminance_lift import luminance_lift_rgb_target
from .poisson_targets import build_poisson_structure_target
from .proximal import (
    normalized_known_luminance_smooth,
    structure_image,
    structure_proximal_target,
    structure_proximal_target_from_structure,
)


@dataclass
class TargetBuildResult:
    """Target tensor plus lightweight diagnostics."""

    target: torch.Tensor
    diagnostics: dict
    target_structure: Optional[torch.Tensor] = None
    initial_structure: Optional[torch.Tensor] = None


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
        return TargetBuildResult(
            target=result.target,
            diagnostics=diagnostics,
            target_structure=result.target_structure,
            initial_structure=result.initial_structure,
        )


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
        solver_dt: float = None,
        solver_dt_ftcs: float = 0.2,
        solver_dt_be: float = 1.0,
        solver_dt_cn: float = 1.0,
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
        self.solver_dt = solver_dt
        self.solver_dt_ftcs = solver_dt_ftcs
        self.solver_dt_be = solver_dt_be
        self.solver_dt_cn = solver_dt_cn
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
            solver_dt=self.solver_dt,
            solver_dt_ftcs=self.solver_dt_ftcs,
            solver_dt_be=self.solver_dt_be,
            solver_dt_cn=self.solver_dt_cn,
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
        return TargetBuildResult(
            target=prox_result.target,
            diagnostics=diagnostics,
            target_structure=poisson_result.structure.to(dtype=mu.dtype, device=mu.device),
            initial_structure=prox_result.initial_structure,
        )


class FlowStructureTargetBuilder:
    """Finite BCNS vorticity-stream flow target followed by RGB proximal target."""

    def __init__(
        self,
        tau2: float = 1.0,
        lambda_structure: float = 0.05,
        structure_sigma: float = 1.0,
        prox_steps: int = 1,
        prox_step_size: float = 0.05,
        initial_mode: str = "projected_mu",
        boundary_mode: str = "normalized_known_smooth",
        boundary_vorticity_mode: str = "none",
        integrator: str = "imex_be",
        poisson_method: str = "sor_rb",
        poisson_max_iter: int = 100,
        poisson_tol: float = 1e-4,
        poisson_omega: float = 1.7,
        h: float = 1.0,
        dt: float = 1e-3,
        pseudo_time: float = 3e-3,
        nu: float = 0.1,
        kappa: float = 0.1,
        smoothing_sigma: float = 1.0,
        cfl: float = 0.25,
        check_cfl: bool = True,
        vorticity_boundary_mode: str = "none",
    ):
        self.tau2 = tau2
        self.lambda_structure = lambda_structure
        self.structure_sigma = structure_sigma
        self.prox_steps = prox_steps
        self.prox_step_size = prox_step_size
        self.initial_mode = initial_mode
        self.boundary_mode = boundary_mode
        self.boundary_vorticity_mode = boundary_vorticity_mode
        self.integrator = integrator
        self.poisson_method = poisson_method
        self.poisson_max_iter = poisson_max_iter
        self.poisson_tol = poisson_tol
        self.poisson_omega = poisson_omega
        self.h = h
        self.dt = dt
        self.pseudo_time = pseudo_time
        self.nu = nu
        self.kappa = kappa
        self.smoothing_sigma = smoothing_sigma
        self.cfl = cfl
        self.check_cfl = check_cfl
        self.vorticity_boundary_mode = vorticity_boundary_mode

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        tau2 = kwargs.get("tau2", self.tau2)
        flow_result = build_flow_structure_target(
            mu=mu,
            measurement=measurement,
            mask_known=mask_known,
            structure_sigma=self.structure_sigma,
            initial_mode=self.initial_mode,
            boundary_mode=self.boundary_mode,
            boundary_vorticity_mode=self.boundary_vorticity_mode,
            integrator=self.integrator,
            poisson_method=self.poisson_method,
            poisson_max_iter=self.poisson_max_iter,
            poisson_tol=self.poisson_tol,
            poisson_omega=self.poisson_omega,
            h=self.h,
            dt=self.dt,
            pseudo_time=self.pseudo_time,
            nu=self.nu,
            kappa=self.kappa,
            smoothing_sigma=self.smoothing_sigma,
            cfl=self.cfl,
            check_cfl=self.check_cfl,
            vorticity_boundary_mode=self.vorticity_boundary_mode,
        )
        prox_result = structure_proximal_target_from_structure(
            mu=mu,
            measurement=measurement,
            mask_known=mask_known,
            target_structure=flow_result.structure.to(dtype=mu.dtype, device=mu.device),
            tau2=tau2,
            lambda_structure=self.lambda_structure,
            structure_sigma=self.structure_sigma,
            prox_steps=self.prox_steps,
            prox_step_size=self.prox_step_size,
        )
        diagnostics = {
            "target_builder": "flow_structure",
            "lambda_structure": self.lambda_structure,
            "structure_sigma": self.structure_sigma,
            "prox_steps": self.prox_steps,
            "prox_step_size": self.prox_step_size,
            "prox_final_loss": prox_result.diagnostics["prox_final_loss"],
            "target_disp": prox_result.diagnostics["target_disp"],
            "flow_structure_disp": flow_result.diagnostics.get("flow_structure_disp"),
        }
        diagnostics.update(flow_result.diagnostics)
        diagnostics["target_disp"] = prox_result.diagnostics["target_disp"]
        return TargetBuildResult(
            target=prox_result.target,
            diagnostics=diagnostics,
            target_structure=flow_result.structure.to(dtype=mu.dtype, device=mu.device),
            initial_structure=prox_result.initial_structure,
        )


class LuminanceLiftTargetBuilder:
    """Lift a scalar structure target directly into RGB hole displacement."""

    def __init__(
        self,
        source: str = "flow",
        lift_scale: float = 1.0,
        mode: str = "equal_rgb",
        **source_params,
    ):
        if source not in ("flow", "poisson", "harmonic", "normalized"):
            raise ValueError("source must be one of 'flow', 'poisson', 'harmonic', or 'normalized'.")
        self.source = source
        self.lift_scale = float(lift_scale)
        self.mode = mode
        self.source_params = dict(source_params)

    def _source_result(self, mu, measurement, mask_known, tau2):
        if self.source == "flow":
            return FlowStructureTargetBuilder(**self.source_params)(
                mu=mu,
                measurement=measurement,
                mask_known=mask_known,
                tau2=tau2,
            )
        if self.source == "poisson":
            return PoissonStructureTargetBuilder(
                target_builder="poisson_structure",
                **self.source_params,
            )(mu=mu, measurement=measurement, mask_known=mask_known, tau2=tau2)
        if self.source == "harmonic":
            return PoissonStructureTargetBuilder(
                target_builder="harmonic_structure",
                **self.source_params,
            )(mu=mu, measurement=measurement, mask_known=mask_known, tau2=tau2)

        structure_sigma = float(self.source_params.get("structure_sigma", 1.0))
        target_structure = normalized_known_luminance_smooth(
            measurement=measurement,
            mask_known=mask_known,
            sigma=structure_sigma,
        ).detach()
        initial_structure = structure_image(hard_project_clean(mu, measurement, mask_known), structure_sigma).detach()
        return TargetBuildResult(
            target=hard_project_clean(mu, measurement, mask_known).detach(),
            diagnostics={
                "target_builder": "normalized_known_smooth",
                "structure_sigma": structure_sigma,
            },
            target_structure=target_structure,
            initial_structure=initial_structure,
        )

    def __call__(self, mu, measurement, mask_known, **kwargs) -> TargetBuildResult:
        validate_image_tensor(mu, "mu")
        if measurement.shape != mu.shape:
            raise ValueError("measurement must have the same shape as mu.")
        validate_mask_tensor(mask_known, mu, "mask_known")
        tau2 = kwargs.get("tau2", self.source_params.get("tau2", 1.0))
        source_result = self._source_result(mu, measurement, mask_known, tau2)
        if source_result.target_structure is None:
            raise ValueError("luminance lift source did not provide a scalar target_structure.")
        structure_sigma = float(self.source_params.get("structure_sigma", 1.0))
        target_structure = source_result.target_structure.to(dtype=mu.dtype, device=mu.device)
        target = luminance_lift_rgb_target(
            mu=mu,
            measurement=measurement,
            mask_known=mask_known,
            target_structure=target_structure,
            structure_sigma=structure_sigma,
            lift_scale=self.lift_scale,
            mode=self.mode,
        )
        hole = hole_from_known(mask_known).to(dtype=mu.dtype)
        scalar_hole = hole[:, :1, :, :]
        target_disp_hole = torch.sqrt(masked_mean_square(target - mu, hole).detach().clamp_min(0.0))
        structure_delta = target_structure.detach() - structure_image(mu, structure_sigma)
        structure_disp_hole = torch.sqrt(
            masked_mean_square(structure_delta, scalar_hole).detach().clamp_min(0.0)
        )
        diagnostics = dict(source_result.diagnostics)
        diagnostics.update(
            {
                "target_builder": f"luminance_lift_{self.source}",
                "lift_source": self.source,
                "lift_scale": self.lift_scale,
                "lift_mode": self.mode,
                "structure_sigma": structure_sigma,
                "target_disp_hole": float(target_disp_hole.item()),
                "structure_disp_hole": float(structure_disp_hole.item()),
            }
        )
        return TargetBuildResult(
            target=target,
            diagnostics=diagnostics,
            target_structure=target_structure.detach(),
            initial_structure=source_result.initial_structure,
        )


def get_target_builder(name: str, **kwargs):
    """Return a target builder by name."""

    builders = {
        "identity": IdentityTargetBuilder,
        "hard_projection": HardProjectionTargetBuilder,
        "simple_hole_shift": SimpleHoleShiftTargetBuilder,
        "structure_prox": StructureProxTargetBuilder,
        "flow_structure": FlowStructureTargetBuilder,
    }
    if name == "harmonic_structure":
        return PoissonStructureTargetBuilder(target_builder="harmonic_structure", **kwargs)
    if name == "poisson_structure":
        return PoissonStructureTargetBuilder(target_builder="poisson_structure", **kwargs)
    if name == "luminance_lift_flow":
        return LuminanceLiftTargetBuilder(source="flow", **kwargs)
    if name == "luminance_lift_poisson":
        return LuminanceLiftTargetBuilder(source="poisson", **kwargs)
    if name == "luminance_lift_harmonic":
        return LuminanceLiftTargetBuilder(source="harmonic", **kwargs)
    if name not in builders:
        expected = sorted(
            list(builders)
            + [
                "harmonic_structure",
                "poisson_structure",
                "luminance_lift_flow",
                "luminance_lift_poisson",
                "luminance_lift_harmonic",
            ]
        )
        raise ValueError(f"Unsupported target builder {name!r}; expected one of {expected}.")
    return builders[name](**kwargs)
