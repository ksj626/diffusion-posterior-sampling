import math

import torch

from bcns.eval_metrics import mae_regions, mse_regions, psnr, ssim_simple


def test_identical_images_psnr_uses_finite_cap():
    x = torch.zeros(1, 3, 8, 8)

    assert psnr(x, x) == 100.0


def test_masked_psnr_only_depends_on_mask_region():
    target = torch.zeros(1, 3, 4, 4)
    pred = target.clone()
    pred[..., :, 2:] = 1.0
    known_left = torch.zeros(1, 1, 4, 4)
    known_left[..., :, :2] = 1.0
    known_right = 1.0 - known_left

    assert psnr(pred, target, known_left) == 100.0
    assert psnr(pred, target, known_right) < 100.0


def test_mse_and_mae_region_split_is_correct():
    target = torch.zeros(1, 1, 2, 2)
    pred = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    mask_known = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    mse = mse_regions(pred, target, mask_known)
    mae = mae_regions(pred, target, mask_known)

    assert mse["known_mse"] == 13.0
    assert mse["hole_mse"] == 29.0
    assert mse["full_mse"] == 21.0
    assert mae["known_mae"] == 3.0
    assert mae["hole_mae"] == 5.0
    assert mae["full_mae"] == 4.0


def test_ssim_simple_returns_finite_scalar():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 16)
    y = x + 0.01 * torch.randn_like(x)
    mask = torch.ones(1, 1, 16, 16)

    value = ssim_simple(x, y, mask=mask, window_size=5)

    assert isinstance(value, float)
    assert math.isfinite(value)
