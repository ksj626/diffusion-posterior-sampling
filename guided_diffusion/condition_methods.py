from abc import ABC, abstractmethod
import torch

from bcns.losses import target_discrepancy_loss
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
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
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
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


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
    """Minimal BCNS target-guidance skeleton for inpainting debug modes."""

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
        self.debug = kwargs.get("debug", False)

    def _gamma_and_tau2(self, x_t, **kwargs):
        t_index = kwargs.get("t_index", kwargs.get("idx", None))
        num_steps = kwargs.get("num_steps", None)
        if t_index is None or num_steps is None:
            # TODO: pass sampler timestep/index into conditioning so the
            # late-ramp schedule can be used during real BCNS guidance.
            return self.schedule.gamma_max, self.schedule.tau2
        gamma = self.schedule.gamma(int(t_index), int(num_steps))
        tau2 = self.schedule.tau2_value(int(t_index), int(num_steps))
        return gamma, tau2

    def conditioning(
        self,
        x_prev,
        x_t,
        x_0_hat,
        measurement,
        mask=None,
        **kwargs,
    ):
        if mask is None:
            raise ValueError("BCNSTargetGuidance requires DPS inpainting mask where mask == 1 is known.")

        gamma, tau2 = self._gamma_and_tau2(x_t, **kwargs)
        if gamma == 0:
            return x_t, torch.zeros((), device=x_t.device, dtype=x_t.dtype)

        result = self.target_builder(
            mu=x_0_hat,
            measurement=measurement,
            mask_known=mask,
        )
        target = result.target.detach()
        loss = target_discrepancy_loss(
            mu=x_0_hat,
            target=target,
            mask_known=mask,
            tau2=tau2,
        )
        grad = torch.autograd.grad(outputs=loss, inputs=x_prev, retain_graph=False)[0]
        x_t = x_t - self.scale * gamma * grad
        return x_t, loss.detach()
