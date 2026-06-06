import torch

from bcns.losses import target_displacement_stats


def _inputs():
    mu = torch.zeros((1, 3, 4, 4), dtype=torch.float64)
    mask_known = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    mask_known[..., 1:3, 1:3] = 0.0
    return mu, mask_known


def test_target_displacement_stats_split_known_only_difference():
    mu, mask_known = _inputs()
    target = mu.clone()
    target = target + 2.0 * mask_known
    stats = target_displacement_stats(mu, target, mask_known)
    assert stats["target_disp_known"].item() > 0.0
    assert stats["target_mse_known"].item() > 0.0
    assert stats["target_disp_hole"].item() == 0.0
    assert stats["target_mse_hole"].item() == 0.0


def test_target_displacement_stats_split_hole_only_difference():
    mu, mask_known = _inputs()
    hole = 1.0 - mask_known
    target = mu + 3.0 * hole
    stats = target_displacement_stats(mu, target, mask_known)
    assert stats["target_disp_hole"].item() > 0.0
    assert stats["target_mse_hole"].item() > 0.0
    assert stats["target_disp_known"].item() == 0.0
    assert stats["target_mse_known"].item() == 0.0
