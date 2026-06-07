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


@register_conditioning_method(name="mcg_bcns")
class MCGBCNSGuidance(ConditioningMethod):
    """Combined MCG measurement guidance plus BCNS structural target guidance."""

    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.mcg_scale = float(kwargs.get("mcg_scale", 0.3))
        self.bcns_scale = float(kwargs.get("bcns_scale", 1.0))
        self.bcns_schedule = schedule_from_kwargs(
            gamma_max=kwargs.get("bcns_gamma_max", 0.5),
            tau2=kwargs.get("bcns_tau2", 1.0),
            pde_start_frac=kwargs.get("bcns_pde_start_frac", 0.5),
            ramp_power=kwargs.get("bcns_ramp_power", 2.0),
        )
        self.bcns_apply_every_n_steps = int(kwargs.get("bcns_apply_every_n_steps", 5))
        self.bcns_reuse_last_target = bool(kwargs.get("bcns_reuse_last_target", False))
        self.bcns_ratio_control = bool(kwargs.get("bcns_ratio_control", False))
        self.bcns_target_update_ratio = float(kwargs.get("bcns_target_update_ratio", 0.03))
        self.bcns_ratio_eps = float(kwargs.get("bcns_ratio_eps", 1e-8))
        self.bcns_ratio_clip_min = float(kwargs.get("bcns_ratio_clip_min", 0.0))
        self.bcns_ratio_clip_max = float(kwargs.get("bcns_ratio_clip_max", 100.0))
        if self.bcns_target_update_ratio < 0:
            raise ValueError("bcns_target_update_ratio must be non-negative.")
        if self.bcns_ratio_eps <= 0:
            raise ValueError("bcns_ratio_eps must be positive.")
        if self.bcns_ratio_clip_min < 0 or self.bcns_ratio_clip_max < self.bcns_ratio_clip_min:
            raise ValueError("bcns ratio clip bounds must satisfy 0 <= min <= max.")
        target_builder = kwargs.get("bcns_target_builder", "luminance_lift_poisson")
        target_builder_params = kwargs.get("bcns_target_builder_params", None) or {}
        self.bcns_target_builder_name = target_builder
        self.bcns_target_builder_params = dict(target_builder_params)
        self.bcns_target_builder = get_target_builder(target_builder, **target_builder_params)
        self.apply_noisy_known_projection = bool(kwargs.get("apply_noisy_known_projection", True))
        self.debug = bool(kwargs.get("debug", False))
        self.last_diagnostics = {}
        self.trace = []
        self._last_target = None
        self._last_target_diagnostics = {}

    def reset_trace(self):
        self.trace = []

    def get_trace(self):
        return list(self.trace)

    @staticmethod
    def _mean(values):
        return sum(values) / float(len(values)) if values else 0.0

    @staticmethod
    def _trace_number(value):
        if torch.is_tensor(value):
            value = value.detach().item()
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return int(value)
        return float(value)

    def summarize_trace(self):
        rows = self.get_trace()
        meas_grad_norms = [float(row.get("meas_grad_norm", 0.0)) for row in rows]
        bcns_grad_norms = [float(row.get("bcns_grad_norm", 0.0)) for row in rows]
        meas_update_norms = [float(row.get("meas_update_norm", 0.0)) for row in rows]
        bcns_update_norms = [float(row.get("bcns_update_norm", 0.0)) for row in rows]
        update_ratios = [float(row.get("bcns_to_meas_update_ratio", 0.0)) for row in rows]
        ratio_scales = [float(row.get("bcns_ratio_scale", 1.0)) for row in rows]
        return {
            "mean_meas_grad_norm": self._mean(meas_grad_norms),
            "mean_bcns_grad_norm": self._mean(bcns_grad_norms),
            "max_bcns_grad_norm": max(bcns_grad_norms) if bcns_grad_norms else 0.0,
            "sum_meas_update_norm": sum(meas_update_norms),
            "sum_bcns_update_norm": sum(bcns_update_norms),
            "max_bcns_to_meas_update_ratio": max(update_ratios) if update_ratios else 0.0,
            "mean_bcns_to_meas_update_ratio": self._mean(update_ratios),
            "bcns_ratio_scale_mean": self._mean(ratio_scales),
            "bcns_ratio_scale_max": max(ratio_scales) if ratio_scales else 0.0,
            "num_bcns_nonzero_steps": sum(1 for value in bcns_grad_norms if value > 0.0),
            "num_bcns_target_recomputed": sum(
                1 for row in rows if bool(row.get("bcns_target_recomputed", False))
            ),
            "num_bcns_target_reused": sum(
                1 for row in rows if bool(row.get("bcns_target_reused", False))
            ),
        }

    def _append_trace(self, diagnostics):
        keys = [
            "t_index",
            "num_steps",
            "meas_loss",
            "bcns_loss",
            "total_loss",
            "meas_grad_norm",
            "bcns_grad_norm",
            "meas_update_norm",
            "bcns_update_raw_norm",
            "bcns_update_norm",
            "bcns_to_meas_grad_ratio",
            "bcns_to_meas_update_ratio",
            "bcns_ratio_control",
            "bcns_target_update_ratio",
            "bcns_ratio_scale",
            "bcns_gamma",
            "bcns_target_recomputed",
            "bcns_target_reused",
            "bcns_target_skipped",
            "target_disp_hole",
            "target_disp_known",
            "target_disp_full",
            "apply_noisy_known_projection",
        ]
        self.trace.append({key: self._trace_number(diagnostics.get(key, 0.0)) for key in keys})

    def _gamma_and_tau2(self, **kwargs):
        t_index = kwargs.get("t_index", kwargs.get("idx", None))
        num_steps = kwargs.get("num_steps", None)
        if t_index is None or num_steps is None:
            return self.bcns_schedule.gamma_max, self.bcns_schedule.tau2
        gamma = self.bcns_schedule.gamma(int(t_index), int(num_steps))
        tau2 = self.bcns_schedule.tau2_value(int(t_index), int(num_steps))
        return gamma, tau2

    def _selected_for_bcns(self, t_index):
        if t_index is None or self.bcns_apply_every_n_steps <= 1:
            return True
        return int(t_index) % self.bcns_apply_every_n_steps == 0

    def _project_if_requested(self, x_t, noisy_measurement, mask):
        if not self.apply_noisy_known_projection:
            return x_t
        if noisy_measurement is None:
            raise ValueError("apply_noisy_known_projection=True requires noisy_measurement.")
        return project_known_noisy(x_t, noisy_measurement, mask)

    def _measurement_loss(self, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ in ("gaussian", "clean"):
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            loss = torch.linalg.norm(difference)
        elif self.noiser.__name__ == "poisson":
            ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - ax
            loss = torch.linalg.norm(difference) / measurement.abs()
            loss = loss.mean()
        else:
            raise NotImplementedError
        return loss

    def _zero_loss(self, ref):
        return torch.zeros((), dtype=ref.dtype, device=ref.device)

    def _bcns_loss(self, x_0_hat, target, mask, tau2):
        if target is None:
            return self._zero_loss(x_0_hat)
        return target_discrepancy_loss(
            mu=x_0_hat,
            target=target,
            mask_known=mask,
            tau2=tau2,
        )

    def _component_grad_norm(self, loss, x_0_hat):
        if (not loss.requires_grad) or float(loss.detach().abs().item()) == 0.0:
            return torch.zeros((), dtype=x_0_hat.dtype, device=x_0_hat.device)
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x_0_hat,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return torch.zeros((), dtype=x_0_hat.dtype, device=x_0_hat.device)
        return torch.linalg.norm(grad.reshape(-1))

    def _target_for_step(self, x_0_hat, measurement, mask, tau2, selected, gamma):
        if gamma == 0.0:
            return None, {}, False, False, True
        if selected:
            result = self.bcns_target_builder(
                mu=x_0_hat,
                measurement=measurement,
                mask_known=mask,
                tau2=tau2,
            )
            target = result.target.detach()
            diagnostics = dict(result.diagnostics)
            self._last_target = target
            self._last_target_diagnostics = dict(diagnostics)
            return target, diagnostics, True, False, False
        if self.bcns_reuse_last_target and self._last_target is not None:
            return self._last_target.detach(), dict(self._last_target_diagnostics), False, True, False
        return None, {}, False, False, True

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
            raise ValueError("MCGBCNSGuidance requires DPS inpainting mask where mask == 1 is known.")

        t_index = kwargs.get("t_index", kwargs.get("idx", None))
        num_steps = kwargs.get("num_steps", None)
        gamma, tau2 = self._gamma_and_tau2(**kwargs)
        selected = self._selected_for_bcns(t_index)

        meas_loss = self._measurement_loss(
            x_0_hat=x_0_hat,
            measurement=measurement,
            mask=mask,
        )
        target, target_diagnostics, target_recomputed, target_reused, target_skipped = self._target_for_step(
            x_0_hat=x_0_hat,
            measurement=measurement,
            mask=mask,
            tau2=tau2,
            selected=selected,
            gamma=gamma,
        )
        if target is None:
            bcns_loss = self._zero_loss(x_0_hat)
            disp_stats = {
                "target_disp_full": torch.zeros((), dtype=x_prev.dtype, device=x_prev.device),
                "target_disp_known": torch.zeros((), dtype=x_prev.dtype, device=x_prev.device),
                "target_disp_hole": torch.zeros((), dtype=x_prev.dtype, device=x_prev.device),
                "target_mse_full": torch.zeros((), dtype=x_prev.dtype, device=x_prev.device),
                "target_mse_known": torch.zeros((), dtype=x_prev.dtype, device=x_prev.device),
                "target_mse_hole": torch.zeros((), dtype=x_prev.dtype, device=x_prev.device),
            }
        else:
            disp_stats = target_displacement_stats(
                mu=x_0_hat,
                target=target,
                mask_known=mask,
            )
            bcns_loss = self._bcns_loss(
                x_0_hat=x_0_hat,
                target=target,
                mask=mask,
                tau2=tau2,
            )

        # The DPS UNet uses a custom checkpoint function that cannot be traversed
        # by two separate backward passes. Use one exact combined VJP for the
        # sampler update, and log component norms in x0-space for trace strength.
        meas_grad_norm = self._component_grad_norm(meas_loss, x_0_hat)
        bcns_grad_norm = self._component_grad_norm(bcns_loss, x_0_hat)
        meas_update_norm = self.mcg_scale * meas_grad_norm
        bcns_update_raw_norm = self.bcns_scale * gamma * bcns_grad_norm
        if self.bcns_ratio_control:
            raw_norm_value = float(bcns_update_raw_norm.detach().item())
            if raw_norm_value == 0.0 or not bool(torch.isfinite(bcns_update_raw_norm).item()):
                ratio_scale = torch.zeros((), dtype=x_prev.dtype, device=x_prev.device)
            else:
                target_norm = self.bcns_target_update_ratio * meas_update_norm
                ratio_scale = target_norm / (bcns_update_raw_norm + self.bcns_ratio_eps)
                ratio_scale = torch.where(
                    torch.isfinite(ratio_scale),
                    ratio_scale,
                    torch.zeros_like(ratio_scale),
                )
                ratio_scale = ratio_scale.clamp(
                    min=self.bcns_ratio_clip_min,
                    max=self.bcns_ratio_clip_max,
                )
        else:
            ratio_scale = torch.ones((), dtype=x_prev.dtype, device=x_prev.device)
        bcns_update_norm = ratio_scale * bcns_update_raw_norm
        eps = torch.finfo(x_prev.dtype).eps
        grad_ratio = bcns_grad_norm / (meas_grad_norm + eps)
        update_ratio = bcns_update_norm / (meas_update_norm + eps)

        effective_bcns_coeff = self.bcns_scale * gamma * ratio_scale.detach()
        combined_loss = self.mcg_scale * meas_loss + effective_bcns_coeff * bcns_loss
        if (not combined_loss.requires_grad) or float(combined_loss.detach().abs().item()) == 0.0:
            combined_update = torch.zeros_like(x_prev)
        else:
            combined_update = torch.autograd.grad(
                outputs=combined_loss,
                inputs=x_prev,
                retain_graph=False,
                allow_unused=True,
            )[0]
            if combined_update is None:
                combined_update = torch.zeros_like(x_prev)

        x_t = x_t - combined_update
        x_t = self._project_if_requested(x_t, noisy_measurement, mask)
        total_loss = meas_loss + bcns_loss

        self.last_diagnostics = {
            "t_index": t_index,
            "num_steps": num_steps,
            "meas_loss": float(meas_loss.detach().item()),
            "bcns_loss": float(bcns_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "loss": float(total_loss.detach().item()),
            "meas_grad_norm": float(meas_grad_norm.detach().item()),
            "bcns_grad_norm": float(bcns_grad_norm.detach().item()),
            "meas_update_norm": float(meas_update_norm.detach().item()),
            "bcns_update_raw_norm": float(bcns_update_raw_norm.detach().item()),
            "bcns_update_norm": float(bcns_update_norm.detach().item()),
            "bcns_to_meas_grad_ratio": float(grad_ratio.detach().item()),
            "bcns_to_meas_update_ratio": float(update_ratio.detach().item()),
            "bcns_ratio_control": self.bcns_ratio_control,
            "bcns_target_update_ratio": self.bcns_target_update_ratio,
            "bcns_ratio_eps": self.bcns_ratio_eps,
            "bcns_ratio_clip_min": self.bcns_ratio_clip_min,
            "bcns_ratio_clip_max": self.bcns_ratio_clip_max,
            "bcns_ratio_scale": float(ratio_scale.detach().item()),
            "bcns_gamma": float(gamma),
            "bcns_tau2": float(tau2),
            "bcns_target_recomputed": bool(target_recomputed),
            "bcns_target_reused": bool(target_reused),
            "bcns_target_skipped": bool(target_skipped),
            "apply_noisy_known_projection": self.apply_noisy_known_projection,
            "mcg_scale": self.mcg_scale,
            "bcns_scale": self.bcns_scale,
            "bcns_gamma_max": self.bcns_schedule.gamma_max,
            "bcns_apply_every_n_steps": self.bcns_apply_every_n_steps,
            "bcns_reuse_last_target": self.bcns_reuse_last_target,
            "bcns_target_builder": self.bcns_target_builder_name,
        }
        self.last_diagnostics.update(target_diagnostics)
        self.last_diagnostics.update(
            {key: float(value.detach().item()) for key, value in disp_stats.items()}
        )
        self._append_trace(self.last_diagnostics)
        if self.debug:
            print(
                f"[MCG+BCNS] t={t_index} meas={self.last_diagnostics['meas_loss']:.6g} "
                f"bcns={self.last_diagnostics['bcns_loss']:.6g} "
                f"g_meas={self.last_diagnostics['meas_grad_norm']:.6g} "
                f"g_bcns={self.last_diagnostics['bcns_grad_norm']:.6g} "
                f"ratio={self.last_diagnostics['bcns_to_meas_update_ratio']:.6g} "
                f"gamma={gamma:.6g} recomputed={target_recomputed} reused={target_reused} "
                f"skipped={target_skipped} project={self.apply_noisy_known_projection}"
            )
        return x_t, total_loss.detach()
        
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
        self.trace = []
        self._last_target = None
        self._last_target_structure = None
        self._last_initial_structure = None
        self._last_target_diagnostics = {}

    def reset_trace(self):
        self.trace = []

    def get_trace(self):
        return list(self.trace)

    @staticmethod
    def _mean(values):
        return sum(values) / float(len(values)) if values else 0.0

    def summarize_trace(self):
        rows = self.get_trace()
        grad_norms = [float(row.get("grad_norm", 0.0)) for row in rows]
        update_norms = [float(row.get("update_norm", 0.0)) for row in rows]
        rgb_losses = [float(row.get("rgb_loss", 0.0)) for row in rows]
        structure_losses = [float(row.get("structure_loss", 0.0)) for row in rows]
        target_disp_holes = [float(row.get("target_disp_hole", 0.0)) for row in rows]
        structure_disp_holes = [float(row.get("structure_disp_hole", 0.0)) for row in rows]
        return {
            "num_guidance_calls": len(rows),
            "num_target_recomputed": sum(1 for row in rows if bool(row.get("target_recomputed", False))),
            "num_target_reused": sum(1 for row in rows if bool(row.get("target_reused", False))),
            "num_nonzero_grad_steps": sum(1 for value in grad_norms if value > 0.0),
            "mean_grad_norm": self._mean(grad_norms),
            "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
            "sum_update_norm": sum(update_norms),
            "mean_update_norm": self._mean(update_norms),
            "max_update_norm": max(update_norms) if update_norms else 0.0,
            "mean_rgb_loss": self._mean(rgb_losses),
            "mean_structure_loss": self._mean(structure_losses),
            "max_structure_loss": max(structure_losses) if structure_losses else 0.0,
            "mean_target_disp_hole": self._mean(target_disp_holes),
            "max_target_disp_hole": max(target_disp_holes) if target_disp_holes else 0.0,
            "mean_structure_disp_hole": self._mean(structure_disp_holes),
            "max_structure_disp_hole": max(structure_disp_holes) if structure_disp_holes else 0.0,
        }

    @staticmethod
    def _trace_number(value):
        if torch.is_tensor(value):
            value = value.detach().item()
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return int(value)
        return float(value)

    def _append_trace(self, diagnostics):
        keys = [
            "t_index",
            "num_steps",
            "gamma",
            "tau2",
            "rgb_loss",
            "structure_loss",
            "total_loss",
            "grad_norm",
            "update_norm",
            "target_disp_full",
            "target_disp_known",
            "target_disp_hole",
            "target_mse_full",
            "target_mse_known",
            "target_mse_hole",
            "structure_disp_hole",
            "target_recomputed",
            "target_reused",
            "apply_noisy_known_projection",
            "apply_every_n_steps",
        ]
        self.trace.append({key: self._trace_number(diagnostics.get(key, 0.0)) for key in keys})

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
        num_steps,
        gamma,
        tau2,
        target_recomputed=False,
        target_reused=False,
    ):
        x_t = self._project_if_requested(x_t, noisy_measurement, mask)
        zero = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
        self.last_diagnostics = {
            "t_index": t_index,
            "num_steps": num_steps,
            "gamma": gamma,
            "tau2": tau2,
            "loss": 0.0,
            "rgb_loss": 0.0,
            "structure_loss": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
            "update_norm": 0.0,
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
        self._append_trace(self.last_diagnostics)
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
        num_steps = kwargs.get("num_steps", None)
        selected = self._selected_for_target(t_index)
        if gamma == 0:
            return self._zero_return(
                x_t,
                noisy_measurement,
                mask,
                t_index,
                num_steps,
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
                num_steps,
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
        update = self.scale * gamma * grad
        grad_norm = torch.linalg.norm(grad.reshape(-1))
        update_norm = torch.linalg.norm(update.reshape(-1))
        x_t = x_t - update
        x_t = self._project_if_requested(x_t, noisy_measurement, mask)
        self.last_diagnostics = {
            "t_index": t_index,
            "num_steps": num_steps,
            "gamma": gamma,
            "tau2": tau2,
            "loss": float(loss.detach().item()),
            "rgb_loss": float(rgb_loss.detach().item()),
            "structure_loss": float(struct_diag["structure_loss"].detach().item()),
            "total_loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "update_norm": float(update_norm.detach().item()),
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
        self._append_trace(self.last_diagnostics)
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
