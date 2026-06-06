import torch

from guided_diffusion.condition_methods import BCNSTargetGuidance


class _Noiser:
    __name__ = "clean"


def _inputs():
    torch.manual_seed(146)
    shape = (1, 3, 6, 6)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = x_prev
    measurement = torch.zeros(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 6, 6), dtype=torch.float64)
    mask_known[..., 2:5, 2:5] = 0.0
    return x_prev, x_t, x_0_hat, measurement, mask_known


def _method(target_builder="identity", target_builder_params=None, **kwargs):
    params = {
        "target_builder": target_builder,
        "target_builder_params": target_builder_params or {},
        "gamma_max": 1.0,
        "tau2": 1.0,
        "rgb_loss_weight": 1.0,
        "structure_loss_weight": 0.0,
    }
    params.update(kwargs)
    return BCNSTargetGuidance(operator=None, noiser=_Noiser(), **params)


def test_trace_length_increments_and_rows_have_expected_keys():
    x_prev, x_t, x_0_hat, measurement, mask_known = _inputs()
    method = _method()
    method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known, t_index=0, num_steps=1)
    trace = method.get_trace()
    expected = {
        "t_index",
        "num_steps",
        "gamma",
        "tau2",
        "rgb_loss",
        "structure_loss",
        "total_loss",
        "grad_norm",
        "update_norm",
        "target_disp_hole",
        "target_recomputed",
        "target_reused",
        "apply_noisy_known_projection",
        "apply_every_n_steps",
    }
    assert len(trace) == 1
    assert expected.issubset(trace[0])


def test_trace_summary_zero_loss_has_no_nonzero_grad_steps():
    x_prev, x_t, x_0_hat, measurement, mask_known = _inputs()
    method = _method()
    method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    summary = method.summarize_trace()
    assert summary["num_guidance_calls"] == 1
    assert summary["num_target_recomputed"] == 1
    assert summary["num_nonzero_grad_steps"] == 0
    assert summary["sum_update_norm"] == 0.0
    assert "mean_structure_disp_hole" in summary


def test_trace_summary_records_nonzero_gradient_and_update():
    x_prev, x_t, x_0_hat, measurement, mask_known = _inputs()
    method = _method(
        "simple_hole_shift",
        {"shift": 0.5},
        scale=2.0,
        gamma_max=0.5,
    )
    out, _ = method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    summary = method.summarize_trace()
    assert summary["num_nonzero_grad_steps"] == 1
    assert summary["mean_grad_norm"] > 0.0
    assert summary["sum_update_norm"] > 0.0
    assert method.get_trace()[0]["update_norm"] == summary["sum_update_norm"]
    assert not torch.allclose(out, x_t)


def test_reset_trace_clears_rows():
    x_prev, x_t, x_0_hat, measurement, mask_known = _inputs()
    method = _method("simple_hole_shift", {"shift": 0.1})
    method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    assert method.get_trace()
    method.reset_trace()
    assert method.get_trace() == []
