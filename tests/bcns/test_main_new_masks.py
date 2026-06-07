import torch

from bcns.masks import (
    make_center_keep_unknown_mask,
    make_global_random_unknown_mask,
    mask_metadata,
)


def test_center_keep_96_geometry_and_metadata():
    mask = make_center_keep_unknown_mask(256, 256, box_size=96)
    top = (256 - 96) // 2
    left = (256 - 96) // 2

    assert mask.shape == (1, 1, 256, 256)
    assert torch.all(mask[..., top : top + 96, left : left + 96] == 0)
    assert mask[..., :top, :].mean().item() == 1.0
    assert mask[..., top + 96 :, :].mean().item() == 1.0

    metadata = mask_metadata(mask)
    assert metadata["mask_unknown_fraction"] == metadata["hole_ratio"]
    assert 0.0 < metadata["known_ratio"] < 1.0


def test_center_keep_128_geometry():
    mask = make_center_keep_unknown_mask(256, 256, box_size=128)
    top = (256 - 128) // 2
    left = (256 - 128) // 2

    assert torch.all(mask[..., top : top + 128, left : left + 128] == 0)
    assert mask.mean().item() == 1.0 - (128 * 128) / float(256 * 256)


def test_global_random_50_60_reproducible_fraction():
    mask_a = make_global_random_unknown_mask(64, 64, 0.5, 0.6, seed=123)
    mask_b = make_global_random_unknown_mask(64, 64, 0.5, 0.6, seed=123)

    assert torch.equal(mask_a, mask_b)
    fraction = mask_a.mean().item()
    assert 0.5 <= fraction <= 0.6
    metadata = mask_metadata(mask_a)
    assert 0.5 <= metadata["mask_unknown_fraction"] <= 0.6
