import torch

from bcns.proximal import normalized_known_luminance_smooth, structure_image, structure_proximal_target


def test_normalized_known_smooth_shape_and_no_zero_leakage():
    measurement_full = torch.full((1, 3, 9, 9), 0.5, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 9, 9), dtype=torch.float64)
    mask_known[..., 3:6, 3:6] = 0.0
    measurement = measurement_full * mask_known

    normalized = normalized_known_luminance_smooth(measurement, mask_known, sigma=1.0)
    naive = structure_image(measurement, sigma=1.0)

    assert normalized.shape == (1, 1, 9, 9)
    assert normalized.dtype == measurement.dtype
    assert normalized[..., 4, 4].item() > 0.45
    assert naive[..., 4, 4].item() < normalized[..., 4, 4].item() - 0.05


def test_structure_proximal_target_accepts_normalized_known_smooth_mode():
    mu = torch.randn((1, 3, 10, 10), dtype=torch.float64)
    mask_known = torch.ones((1, 1, 10, 10), dtype=torch.float64)
    mask_known[..., 3:7, 3:7] = 0.0
    measurement = mu * mask_known
    result = structure_proximal_target(
        mu,
        measurement,
        mask_known,
        tau2=1.0,
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=1,
        prox_step_size=0.05,
        target_mode="normalized_known_smooth",
    )
    assert result.target.shape == mu.shape
    assert torch.isfinite(result.target).all()
