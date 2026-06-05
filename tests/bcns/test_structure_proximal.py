import torch

from bcns.dps_adapter import hard_project_clean
from bcns.proximal import structure_image, structure_proximal_target


def _fake_inputs():
    torch.manual_seed(11)
    mu = torch.randn((1, 3, 12, 12), dtype=torch.float64)
    measurement = torch.randn_like(mu)
    mask_known = torch.ones((1, 1, 12, 12), dtype=torch.float64)
    mask_known[..., 3:9, 4:8] = 0.0
    measurement = measurement * mask_known
    return mu, measurement, mask_known


def test_structure_image_shape():
    mu, _, _ = _fake_inputs()
    out = structure_image(mu, sigma=1.0)
    assert out.shape == (1, 1, 12, 12)
    assert out.dtype == mu.dtype


def test_zero_prox_steps_equals_hard_projection():
    mu, measurement, mask_known = _fake_inputs()
    result = structure_proximal_target(
        mu,
        measurement,
        mask_known,
        tau2=1.0,
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=0,
        prox_step_size=0.1,
    )
    expected = hard_project_clean(mu, measurement, mask_known)
    assert torch.allclose(result.target, expected)


def test_known_pixels_equal_measurement_and_target_finite():
    mu, measurement, mask_known = _fake_inputs()
    result = structure_proximal_target(
        mu,
        measurement,
        mask_known,
        tau2=1.0,
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=2,
        prox_step_size=0.05,
        target_mode="measurement_only_smooth",
    )
    known = mask_known.expand_as(mu).bool()
    assert torch.allclose(result.target[known], measurement[known])
    assert torch.isfinite(result.target).all()
    assert result.target.shape == mu.shape


def test_lambda_zero_equals_hard_projection():
    mu, measurement, mask_known = _fake_inputs()
    result = structure_proximal_target(
        mu,
        measurement,
        mask_known,
        tau2=1.0,
        lambda_structure=0.0,
        structure_sigma=1.0,
        prox_steps=3,
        prox_step_size=0.1,
    )
    assert torch.allclose(result.target, hard_project_clean(mu, measurement, mask_known))


def test_target_modes_work():
    mu, measurement, mask_known = _fake_inputs()
    for mode in ("projected_mu", "measurement_only_smooth", "mu"):
        result = structure_proximal_target(
            mu,
            measurement,
            mask_known,
            tau2=1.0,
            lambda_structure=0.05,
            structure_sigma=1.0,
            prox_steps=1,
            prox_step_size=0.05,
            target_mode=mode,
        )
        assert torch.isfinite(result.target).all()
        assert result.target.shape == mu.shape
