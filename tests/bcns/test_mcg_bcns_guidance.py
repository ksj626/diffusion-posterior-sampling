import torch

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.nn import checkpoint


class _Noiser:
    __name__ = "clean"


class _InpaintingOperator:
    def forward(self, data, **kwargs):
        mask = kwargs.get("mask", None)
        if mask is None:
            raise ValueError("mask required")
        return data * mask


def _inputs():
    torch.manual_seed(481)
    shape = (1, 3, 6, 6)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.zeros(shape, dtype=torch.float64)
    x_0_hat = x_prev
    measurement = torch.zeros(shape, dtype=torch.float64)
    noisy_measurement = torch.full(shape, 0.75, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 6, 6), dtype=torch.float64)
    mask_known[..., 2:5, 2:5] = 0.0
    return x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known


def _method(**kwargs):
    params = {
        "mcg_scale": 0.3,
        "bcns_scale": 1.0,
        "bcns_gamma_max": 1.0,
        "bcns_tau2": 1.0,
        "bcns_pde_start_frac": 0.0,
        "bcns_apply_every_n_steps": 1,
        "bcns_target_builder": "identity",
        "bcns_target_builder_params": {},
        "apply_noisy_known_projection": False,
    }
    params.update(kwargs)
    return get_conditioning_method("mcg_bcns", _InpaintingOperator(), _Noiser(), **params)


def test_mcg_bcns_is_registered_and_returns_same_shape():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _inputs()
    method = _method()
    out, loss = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=0,
        num_steps=1,
    )
    assert out.shape == x_t.shape
    assert loss.shape == torch.Size([])


def test_mcg_bcns_projection_enforces_noisy_known_pixels():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _inputs()
    method = _method(apply_noisy_known_projection=True)
    out, _ = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=0,
        num_steps=1,
    )
    assert torch.allclose(out * mask_known, noisy_measurement * mask_known)


def test_mcg_bcns_trace_contains_measurement_and_bcns_norms():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _inputs()
    method = _method(
        bcns_target_builder="simple_hole_shift",
        bcns_target_builder_params={"shift": 0.5},
    )
    method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=0,
        num_steps=1,
    )
    trace = method.get_trace()
    assert len(trace) == 1
    assert "meas_grad_norm" in trace[0]
    assert "bcns_grad_norm" in trace[0]
    assert "bcns_to_meas_update_ratio" in trace[0]


def test_mcg_update_still_happens_when_bcns_target_is_skipped():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _inputs()
    method = _method(
        bcns_target_builder="simple_hole_shift",
        bcns_target_builder_params={"shift": 0.5},
        bcns_pde_start_frac=1.0,
    )
    out, _ = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=1,
        num_steps=3,
    )
    row = method.get_trace()[0]
    assert row["bcns_target_skipped"] is True
    assert row["bcns_grad_norm"] == 0.0
    assert row["meas_grad_norm"] > 0.0
    assert not torch.allclose(out, x_t)


def test_active_simple_hole_shift_has_nonzero_bcns_gradient():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _inputs()
    method = _method(
        bcns_target_builder="simple_hole_shift",
        bcns_target_builder_params={"shift": 0.5},
    )
    out, _ = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=0,
        num_steps=1,
    )
    row = method.get_trace()[0]
    assert row["bcns_target_recomputed"] is True
    assert row["bcns_grad_norm"] > 0.0
    assert row["bcns_update_norm"] > 0.0
    assert not torch.allclose(out, x_t)


def test_active_bcns_uses_single_checkpoint_backward_pass():
    x_prev, x_t, _, measurement, noisy_measurement, mask_known = _inputs()

    def _run(tensor):
        return tensor * 1.0

    x_0_hat = checkpoint(_run, (x_prev,), (), True)
    method = _method(
        bcns_target_builder="simple_hole_shift",
        bcns_target_builder_params={"shift": 0.5},
    )
    out, _ = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=0,
        num_steps=1,
    )
    row = method.get_trace()[0]
    assert row["meas_grad_norm"] > 0.0
    assert row["bcns_grad_norm"] > 0.0
    assert not torch.allclose(out, x_t)
