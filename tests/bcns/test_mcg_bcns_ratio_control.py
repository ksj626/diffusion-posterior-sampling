import math

import torch

from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "clean"


class _InpaintingOperator:
    def forward(self, data, **kwargs):
        mask = kwargs.get("mask", None)
        if mask is None:
            raise ValueError("mask required")
        return data * mask


def _inputs():
    torch.manual_seed(49)
    shape = (1, 3, 6, 6)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.zeros(shape, dtype=torch.float64)
    x_0_hat = x_prev
    measurement = torch.zeros(shape, dtype=torch.float64)
    noisy_measurement = torch.zeros(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 6, 6), dtype=torch.float64)
    mask_known[..., 2:5, 2:5] = 0.0
    return x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known


def _method(**kwargs):
    params = {
        "mcg_scale": 1.0,
        "bcns_scale": 1.0,
        "bcns_gamma_max": 1.0,
        "bcns_tau2": 1.0,
        "bcns_pde_start_frac": 0.0,
        "bcns_apply_every_n_steps": 1,
        "bcns_target_builder": "simple_hole_shift",
        "bcns_target_builder_params": {"shift": 0.5},
        "apply_noisy_known_projection": False,
    }
    params.update(kwargs)
    return get_conditioning_method("mcg_bcns", _InpaintingOperator(), _Noiser(), **params)


def _run(method):
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _inputs()
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
    return out, method.last_diagnostics


def test_ratio_control_false_preserves_normal_behavior_when_ratio_matches_raw():
    normal_out, normal_diag = _run(_method(bcns_ratio_control=False))
    matched_ratio = normal_diag["bcns_to_meas_update_ratio"]
    controlled_out, controlled_diag = _run(
        _method(
            bcns_ratio_control=True,
            bcns_target_update_ratio=matched_ratio,
            bcns_ratio_clip_max=100.0,
        )
    )
    assert controlled_diag["bcns_ratio_control"] is True
    assert abs(controlled_diag["bcns_ratio_scale"] - 1.0) < 1e-6
    assert torch.allclose(controlled_out, normal_out)


def test_ratio_control_targets_requested_nonzero_update_ratio():
    _, diag = _run(
        _method(
            bcns_ratio_control=True,
            bcns_target_update_ratio=0.1,
            bcns_ratio_clip_max=100.0,
        )
    )
    assert diag["bcns_update_raw_norm"] > 0.0
    assert abs(diag["bcns_to_meas_update_ratio"] - 0.1) < 1e-6


def test_ratio_control_zero_bcns_grad_has_zero_update_and_no_nan():
    _, diag = _run(
        _method(
            bcns_target_builder="identity",
            bcns_target_builder_params={},
            bcns_ratio_control=True,
            bcns_target_update_ratio=0.1,
        )
    )
    assert diag["bcns_update_raw_norm"] == 0.0
    assert diag["bcns_update_norm"] == 0.0
    assert diag["bcns_ratio_scale"] == 0.0
    assert math.isfinite(diag["bcns_to_meas_update_ratio"])


def test_ratio_scale_is_clipped_by_maximum():
    _, diag = _run(
        _method(
            bcns_ratio_control=True,
            bcns_target_update_ratio=10.0,
            bcns_ratio_clip_max=0.5,
        )
    )
    assert diag["bcns_ratio_scale"] <= 0.5
    assert diag["bcns_update_norm"] <= 0.5 * diag["bcns_update_raw_norm"] + 1e-12
