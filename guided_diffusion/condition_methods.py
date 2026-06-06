from abc import ABC, abstractmethod
import torch

from bcns.dps_adapter import project_known_noisy
from bcns.losses import (
    structure_displacement_loss,
    target_discrepancy_loss,
    target_displacement_stats,
)
from bcns.schedules import schedule_from_kwargs
from bcns.target_builders import get_target_builder

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        kwargs.pop('measurement', None)
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ in ('gaussian', 'clean'):
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t


@register_conditioning_method(name='projection_fixed')
class FixedProjection(ConditioningMethod):
    """Inpainting projection with explicit known-pixel replacement."""

    def conditioning(self, x_t, noisy_measurement, mask=None, **kwargs):
        if mask is None:
            raise ValueError("projection_fixed requires DPS inpainting mask where mask == 1 is known.")
        if noisy_measurement is None:
            raise ValueError("projection_fixed requires noisy_measurement.")
        return project_known_noisy(x_t, noisy_measurement, mask)


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name='mcg_fixed')
class FixedManifoldConstraintGradient(ManifoldConstraintGradient):
    """MCG baseline followed by explicit noisy known-pixel projection."""

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask=None, **kwargs):
        if mask is None:
            raise ValueError("mcg_fixed requires DPS inpainting mask where mask == 1 is known.")
        if noisy_measurement is None:
            raise ValueError("mcg_fixed requires noisy_measurement.")
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev,
            x_0_hat=x_0_hat,
            measurement=measurement,
            mask=mask,
            **kwargs,
        )
        x_t = x_t - norm_grad * self.scale
        x_t = project_known_noisy(x_t, noisy_measurement, mask)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="bcns_target")
class BCNSTargetGuidance(ConditioningMethod):
    """BCNS target-guidance conditioning for inpainting debug and PDE targets."""

    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)
        target_builder = kwargs.get("target_builder", "identity")
        target_builder_params = kwargs.get("target_builder_params", None) or {}
        self.target_builder = get_target_builder(target_builder, **target_builder_params)
        self.schedule = schedule_from_kwargs(
            gamma_max=kwargs.get("gamma_max", 0.05),
            tau2=kwargs.get("tau2", 1.0),
            pde_start_frac=kwargs.get("pde_start_frac", 0.7),
            ramp_power=kwargs.get("ramp_power", 2.0),
        )
        self.apply_noisy_known_projection = kwargs.get("apply_noisy_known_projection", False)
        self.apply_every_n_steps = int(kwargs.get("apply_every_n_steps", 1))
        self.reuse_last_target = kwargs.get("reuse_last_target", False)
        self.rgb_loss_weight = float(kwargs.get("rgb_loss_weight", 1.0))
        self.structure_loss_weight = float(kwargs.get("structure_loss_weight", 0.0))
        self.structure_loss_sigma = float(kwargs.get("structure_loss_sigma", 1.0))
        self.debug = kwargs.get("debug", False)
        self.last_diagnostics = {}
        self._last_target = None
        self._last_target_structure = None
        self._last_initial_structure = None
        self._last_target_diagnostics = {}

    def _gamma_and_tau2(self, x_t, **kwargs):
        t_index = kwargs.get("t_index", kwargs.get("idx", None))
        num_steps = kwargs.get("num_steps", None)
        if t_index is None or num_steps is None:
            return self.schedule.gamma_max, self.schedule.tau2
        gamma = self.schedule.gamma(int(t_index), int(num_steps))
        tau2 = self.schedule.tau2_value(int(t_index), int(num_steps))
        return gamma, tau2

    def _selected_for_target(self, t_index):
        if t_index is None or self.apply_every_n_steps <= 1:
            return True
        return int(t_index) % self.apply_every_n_steps == 0

    def _project_if_requested(self, x_t, noisy_measurement, mask):
        if not self.apply_noisy_known_projection:
            return x_t
        if noisy_measurement is None:
            raise ValueError("apply_noisy_known_projection=True requires noisy_measurement.")
        return project_known_noisy(x_t, noisy_measurement, mask)

    def _zero_return(
        self,
        x_t,
        noisy_measurement,
        mask,
        t_index,
        gamma,
        tau2,
        target_recomputed=False,
        target_reused=False,
    ):
        x_t = self._project_if_requested(x_t, noisy_measurement, mask)
        zero = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
        self.last_diagnostics = {
            "t_index": t_index,
            "gamma": gamma,
            "tau2": tau2,
            "loss": 0.0,
            "rgb_loss": 0.0,
            "structure_loss": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
            "target_disp": 0.0,
            "target_disp_full": 0.0,
            "target_disp_known": 0.0,
            "target_disp_hole": 0.0,
            "target_mse_full": 0.0,
            "target_mse_known": 0.0,
            "target_mse_hole": 0.0,
            "structure_disp_hole": 0.0,
            "apply_noisy_known_projection": self.apply_noisy_known_projection,
            "target_recomputed": bool(target_recomputed),
            "target_reused": bool(target_reused),
            "apply_every_n_steps": self.apply_every_n_steps,
            "rgb_loss_weight": self.rgb_loss_weight,
            "structure_loss_weight": self.structure_loss_weight,
            "structure_loss_sigma": self.structure_loss_sigma,
        }
        if self.debug:
            print(
                f"[BCNS] t={t_index} gamma={gamma:.6g} tau2={tau2:.6g} "
                f"loss=0 grad=0 target_disp=0 "
                f"recomputed={target_recomputed} reused={target_reused} "
                f"every={self.apply_every_n_steps} project={self.apply_noisy_known_projection}"
            )
        return x_t, zero

    def conditioning(
        self,
        x_prev,
        x_t,
        x_0_hat,
        measurement,
        noisy_measurement=None,
        mask=None,
        **kwargs,
    ):
        if mask is None:
            raise ValueError("BCNSTargetGuidance requires DPS inpainting mask where mask == 1 is known.")

        gamma, tau2 = self._gamma_and_tau2(x_t, **kwargs)
        t_index = kwargs.get("t_index", kwargs.get("idx", None))
        selected = self._selected_for_target(t_index)
        if gamma == 0:
            return self._zero_return(
                x_t,
                noisy_measurement,
                mask,
                t_index,
                gamma,
                tau2,
                target_recomputed=False,
                target_reused=False,
            )

        target_recomputed = False
        target_reused = False
        result_diagnostics = {}
        if selected:
            result = self.target_builder(
                mu=x_0_hat,
                measurement=measurement,
                mask_known=mask,
                tau2=tau2,
            )
            target = result.target.detach()
            target_structure = None
            initial_structure = None
            if result.target_structure is not None:
                target_structure = result.target_structure.detach()
            if result.initial_structure is not None:
                initial_structure = result.initial_structure.detach()
            result_diagnostics = dict(result.diagnostics)
            self._last_target = target
            self._last_target_structure = target_structure
            self._last_initial_structure = initial_structure
            self._last_target_diagnostics = dict(result_diagnostics)
            target_recomputed = True
        elif self.reuse_last_target and self._last_target is not None:
            target = self._last_target.detach()
            target_structure = None if self._last_target_structure is None else self._last_target_structure.detach()
            result_diagnostics = dict(self._last_target_diagnostics)
            target_reused = True
        else:
            return self._zero_return(
                x_t,
                noisy_measurement,
                mask,
                t_index,
                gamma,
                tau2,
                target_recomputed=False,
                target_reused=False,
            )

        disp_stats = target_displacement_stats(
            mu=x_0_hat,
            target=target,
            mask_known=mask,
        )
        rgb_loss = self.rgb_loss_weight * target_discrepancy_loss(
            mu=x_0_hat,
            target=target,
            mask_known=mask,
            tau2=tau2,
        )
        struct_loss, struct_diag = structure_displacement_loss(
            mu=x_0_hat,
            target_structure=target_structure,
            mask_known=mask,
            structure_sigma=self.structure_loss_sigma,
            weight=self.structure_loss_weight,
        )
        loss = rgb_loss + struct_loss
        if (not loss.requires_grad) or float(loss.detach().abs().item()) == 0.0:
            grad = torch.zeros_like(x_prev)
        else:
            grad = torch.autograd.grad(outputs=loss, inputs=x_prev, retain_graph=False, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(x_prev)
        grad_norm = torch.linalg.norm(grad.reshape(-1))
        x_t = x_t - self.scale * gamma * grad
        x_t = self._project_if_requested(x_t, noisy_measurement, mask)
        self.last_diagnostics = {
            "t_index": t_index,
            "gamma": gamma,
            "tau2": tau2,
            "loss": float(loss.detach().item()),
            "rgb_loss": float(rgb_loss.detach().item()),
            "structure_loss": float(struct_diag["structure_loss"].detach().item()),
            "total_loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "target_disp": float(disp_stats["target_disp_full"].detach().item()),
        }
        self.last_diagnostics.update(
            {key: float(value.detach().item()) for key, value in disp_stats.items()}
        )
        self.last_diagnostics["structure_disp_hole"] = float(
            struct_diag["structure_disp_hole"].detach().item()
        )
        self.last_diagnostics.update(result_diagnostics)
        self.last_diagnostics["apply_noisy_known_projection"] = self.apply_noisy_known_projection
        self.last_diagnostics["target_recomputed"] = bool(target_recomputed)
        self.last_diagnostics["target_reused"] = bool(target_reused)
        self.last_diagnostics["apply_every_n_steps"] = self.apply_every_n_steps
        self.last_diagnostics["rgb_loss_weight"] = self.rgb_loss_weight
        self.last_diagnostics["structure_loss_weight"] = self.structure_loss_weight
        self.last_diagnostics["structure_loss_sigma"] = self.structure_loss_sigma
        if self.debug:
            print(
                f"[BCNS] t={t_index} gamma={gamma:.6g} tau2={tau2:.6g} "
                f"loss={self.last_diagnostics['total_loss']:.6g} "
                f"rgb={self.last_diagnostics['rgb_loss']:.6g} "
                f"struct={self.last_diagnostics['structure_loss']:.6g} "
                f"grad={self.last_diagnostics['grad_norm']:.6g} "
                f"target_hole={self.last_diagnostics['target_disp_hole']:.6g} "
                f"struct_hole={self.last_diagnostics['structure_disp_hole']:.6g} "
                f"recomputed={target_recomputed} reused={target_reused} "
                f"every={self.apply_every_n_steps} project={self.apply_noisy_known_projection}"
            )
        return x_t, loss.detach()
