import torch

from bcns.dps_adapter import clean_composite, project_known_noisy, split_known_hole_mse


def _inputs():
    raw = torch.arange(12, dtype=torch.float64).view(1, 3, 2, 2)
    measurement = torch.full_like(raw, -3.0)
    noisy = torch.full_like(raw, 7.0)
    mask_known = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    return raw, measurement, noisy, mask_known


def test_clean_composite_known_equals_measurement_hole_equals_raw():
    raw, measurement, _, mask_known = _inputs()
    out = clean_composite(raw, measurement, mask_known)
    known = mask_known.expand_as(raw).bool()
    assert torch.equal(out[known], measurement[known])
    assert torch.equal(out[~known], raw[~known])


def test_project_known_noisy_known_equals_noisy_hole_equals_x_t():
    x_t, _, noisy, mask_known = _inputs()
    out = project_known_noisy(x_t, noisy, mask_known)
    known = mask_known.expand_as(x_t).bool()
    assert torch.equal(out[known], noisy[known])
    assert torch.equal(out[~known], x_t[~known])


def test_split_known_hole_mse_returns_expected_regions():
    pred = torch.zeros((1, 3, 2, 2), dtype=torch.float64)
    target = torch.zeros_like(pred)
    target[..., 0, 0] = 2.0
    target[..., 0, 1] = 3.0
    mask_known = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64)
    metrics = split_known_hole_mse(pred, target, mask_known)
    assert torch.allclose(metrics["known_mse"], torch.tensor(4.0, dtype=torch.float64))
    assert torch.allclose(metrics["hole_mse"], torch.tensor(3.0, dtype=torch.float64))
    assert torch.allclose(metrics["full_mse"], torch.tensor(13.0 / 4.0, dtype=torch.float64))
