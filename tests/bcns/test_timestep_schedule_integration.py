import torch

from bcns.schedules import BCNSSchedule
from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "gaussian"


def _fake_tensors():
    torch.manual_seed(13)
    shape = (1, 3, 8, 8)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = x_prev * 0.5 + 0.1
    measurement = torch.randn(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 8, 8), dtype=torch.float64)
    mask_known[..., 2:6, 2:6] = 0.0
    return x_prev, x_t, x_0_hat, measurement, mask_known


def test_schedule_gamma_zero_before_late_ramp_and_positive_late():
    schedule = BCNSSchedule(gamma_max=0.1, pde_start_frac=0.7, ramp_power=2.0)
    assert schedule.gamma(t_index=900, num_steps=1000) == 0.0
    assert schedule.gamma(t_index=0, num_steps=1000) > 0.0


def test_conditioning_uses_zero_gamma_early_and_nonzero_late():
    method = get_conditioning_method(
        "bcns_target",
        operator=None,
        noiser=_Noiser(),
        target_builder="simple_hole_shift",
        target_builder_params={"shift": 0.2},
        gamma_max=0.5,
        pde_start_frac=0.7,
        ramp_power=1.0,
    )
    x_prev, x_t, x_0_hat, measurement, mask_known = _fake_tensors()
    early, early_loss = method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        mask=mask_known,
        t_index=900,
        num_steps=1000,
    )
    assert torch.allclose(early, x_t)
    assert early_loss.item() == 0.0

    x_prev, x_t, x_0_hat, measurement, mask_known = _fake_tensors()
    late, late_loss = method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        mask=mask_known,
        t_index=0,
        num_steps=1000,
    )
    assert late_loss.item() > 0.0
    assert not torch.allclose(late, x_t)
    assert method.last_diagnostics["gamma"] > 0.0
