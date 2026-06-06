import torch

from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "clean"


def _fake_tensors(seed=47):
    torch.manual_seed(seed)
    shape = (1, 3, 8, 8)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = 0.5 * x_prev + 0.1
    measurement = torch.randn(shape, dtype=torch.float64)
    noisy_measurement = torch.randn(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 8, 8), dtype=torch.float64)
    mask_known[..., 2:6, 2:6] = 0.0
    return x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known


def _method(**kwargs):
    params = {
        "target_builder": "simple_hole_shift",
        "target_builder_params": {"shift": 0.2},
        "gamma_max": 0.5,
        "pde_start_frac": 0.0,
        "ramp_power": 1.0,
        "apply_every_n_steps": 10,
        "reuse_last_target": False,
    }
    params.update(kwargs)
    return get_conditioning_method("bcns_target", operator=None, noiser=_Noiser(), **params)


def test_frequency_recomputes_only_on_selected_timesteps():
    method = _method()
    x_prev, x_t, x_0_hat, measurement, noisy, mask = _fake_tensors()
    out, loss = method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy,
        mask=mask,
        t_index=20,
        num_steps=100,
    )
    assert loss.item() > 0.0
    assert method.last_diagnostics["target_recomputed"]
    assert not method.last_diagnostics["target_reused"]
    assert method.last_diagnostics["apply_every_n_steps"] == 10

    x_prev, x_t, x_0_hat, measurement, noisy, mask = _fake_tensors(seed=48)
    skipped, skipped_loss = method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy,
        mask=mask,
        t_index=21,
        num_steps=100,
    )
    assert skipped_loss.item() == 0.0
    assert torch.equal(skipped, x_t)
    assert not method.last_diagnostics["target_recomputed"]
    assert not method.last_diagnostics["target_reused"]


def test_skipped_timestep_still_applies_optional_projection():
    method = _method(apply_noisy_known_projection=True)
    x_prev, x_t, x_0_hat, measurement, noisy, mask = _fake_tensors()
    out, loss = method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy,
        mask=mask,
        t_index=21,
        num_steps=100,
    )
    known = mask.expand_as(x_t).bool()
    assert loss.item() == 0.0
    assert torch.equal(out[known], noisy[known])
    assert torch.equal(out[~known], x_t[~known])


def test_reuse_last_target_on_skipped_timestep_when_enabled():
    method = _method(reuse_last_target=True)
    x_prev, x_t, x_0_hat, measurement, noisy, mask = _fake_tensors()
    method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy,
        mask=mask,
        t_index=20,
        num_steps=100,
    )

    x_prev, x_t, x_0_hat, measurement, noisy, mask = _fake_tensors(seed=49)
    out, loss = method.conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy,
        mask=mask,
        t_index=21,
        num_steps=100,
    )
    assert loss.item() > 0.0
    assert not method.last_diagnostics["target_recomputed"]
    assert method.last_diagnostics["target_reused"]
    assert not torch.allclose(out, x_t)
