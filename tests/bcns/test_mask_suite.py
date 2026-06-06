import torch

from bcns.masks import (
    make_center_box_unknown_mask,
    make_freeform_medium_mask,
    make_text_like_mask,
    make_thick_scratch_mask,
    make_thin_scratch_mask,
    mask_metadata,
)


def _assert_binary_mask(mask):
    assert mask.shape == (1, 1, 64, 64)
    assert torch.isfinite(mask).all()
    assert torch.all((mask == 0) | (mask == 1))
    ratio = float(mask.mean().item())
    assert 0.0 < ratio < 0.8


def test_structured_masks_have_expected_shape_and_binary_values():
    masks = [
        make_thin_scratch_mask(64, 64, thickness=5),
        make_thick_scratch_mask(64, 64, thickness=12),
        make_text_like_mask(64, 64, thickness=12),
        make_freeform_medium_mask(64, 64, seed=7, length_range=(20, 60)),
        make_center_box_unknown_mask(64, 64, box_size=24),
    ]
    for mask in masks:
        _assert_binary_mask(mask)


def test_thick_scratch_has_larger_hole_ratio_than_thin_scratch():
    thin = make_thin_scratch_mask(64, 64, thickness=5)
    thick = make_thick_scratch_mask(64, 64, thickness=12)
    assert thick.mean().item() > thin.mean().item()


def test_mask_metadata_contains_required_keys_and_finite_values():
    mask = make_freeform_medium_mask(64, 64, seed=11, length_range=(20, 60))
    metadata = mask_metadata(mask)
    expected = {
        "hole_ratio",
        "known_ratio",
        "boundary_length_approx",
        "num_components_approx",
        "bbox_area_ratio",
    }
    assert expected.issubset(metadata)
    assert 0.0 < metadata["hole_ratio"] < 0.8
    assert 0.0 < metadata["known_ratio"] < 1.0
    assert metadata["boundary_length_approx"] > 0.0
    assert metadata["num_components_approx"] >= 1
    assert 0.0 < metadata["bbox_area_ratio"] <= 1.0
