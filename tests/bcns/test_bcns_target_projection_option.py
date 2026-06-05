import pytest
import torch

from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "clean"


def _fake_tensors():
    torch.manual_seed(17)
    shape = (1, 3, 8, 8)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = x_prev * 0.5 + 0.1
    measurement = torch.randn(shape, dtype=torch.float64)
    noisy_measurement = torch.randn(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 8, 8), dtype=torch.float64)
    mask_known[..., 2:6, 2:6] = 0.0
    return x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known


def _method(apply_projection):
    return get_conditioning_method(
        "bcns_target",
        operator=None,
        noiser=_Noiser(),
        target_builder="simple_hole_shift",
        target_builder_params={"shift": 0.1},
        gamma_max=0.5,
        pde_start_frac=0.7,
        apply_noisy_known_projection=apply_projection,
    )


def test_projection_option_false_leaves_gamma_zero_output_unprojected():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _fake_tensors()
    out, loss = _method(False).conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=900,
        num_steps=1000,
    )
    known = mask_known.expand_as(x_t).bool()
    assert loss.item() == 0.0
    assert torch.equal(out, x_t)
    assert not torch.equal(out[known], noisy_measurement[known])


def test_projection_option_true_projects_known_region_even_when_gamma_zero():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _fake_tensors()
    out, loss = _method(True).conditioning(
        x_prev,
        x_t.clone(),
        x_0_hat,
        measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
        t_index=900,
        num_steps=1000,
    )
    known = mask_known.expand_as(x_t).bool()
    assert loss.item() == 0.0
    assert torch.equal(out[known], noisy_measurement[known])
    assert torch.equal(out[~known], x_t[~known])


def test_projection_option_true_requires_noisy_measurement():
    x_prev, x_t, x_0_hat, measurement, _, mask_known = _fake_tensors()
    with pytest.raises(ValueError, match="requires noisy_measurement"):
        _method(True).conditioning(
            x_prev,
            x_t.clone(),
            x_0_hat,
            measurement,
            noisy_measurement=None,
            mask=mask_known,
            t_index=900,
            num_steps=1000,
        )
