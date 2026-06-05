import torch

from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "clean"


class _InpaintingOperator:
    def forward(self, data, **kwargs):
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError("Require mask")
        return data * mask


def _fake_tensors():
    torch.manual_seed(29)
    shape = (1, 3, 8, 8)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = 0.25 * x_prev + 0.15
    mask_known = torch.ones((1, 1, 8, 8), dtype=torch.float64)
    mask_known[..., 2:6, 3:7] = 0.0
    measurement = torch.randn(shape, dtype=torch.float64) * mask_known
    noisy_measurement = torch.randn(shape, dtype=torch.float64)
    return x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known


def test_projection_fixed_sets_known_region_to_noisy_measurement():
    _, x_t, _, _, noisy_measurement, mask_known = _fake_tensors()
    method = get_conditioning_method("projection_fixed", operator=_InpaintingOperator(), noiser=_Noiser())
    out = method.conditioning(
        x_t=x_t.clone(),
        noisy_measurement=noisy_measurement,
        mask=mask_known,
    )
    known = mask_known.expand_as(x_t).bool()
    assert torch.equal(out[known], noisy_measurement[known])
    assert torch.equal(out[~known], x_t[~known])


def test_mcg_fixed_sets_known_region_to_noisy_measurement_after_gradient_step():
    x_prev, x_t, x_0_hat, measurement, noisy_measurement, mask_known = _fake_tensors()
    method = get_conditioning_method(
        "mcg_fixed",
        operator=_InpaintingOperator(),
        noiser=_Noiser(),
        scale=0.1,
    )
    out, norm = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        noisy_measurement=noisy_measurement,
        mask=mask_known,
    )
    known = mask_known.expand_as(x_t).bool()
    assert torch.isfinite(norm)
    assert torch.equal(out[known], noisy_measurement[known])


def test_fixed_conditioning_methods_register_without_replacing_legacy_names():
    for name in ("projection", "mcg", "projection_fixed", "mcg_fixed"):
        get_conditioning_method(name, operator=_InpaintingOperator(), noiser=_Noiser())
