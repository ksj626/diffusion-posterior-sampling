import math

import torch

from bcns.structural_metrics import (
    boundary_band,
    gradient_mismatch,
    isophote_angle_error,
    laplacian_mismatch,
    luminance_structure,
    ns_residual_metric,
)


def _center_box_known(size=16, box=6):
    mask = torch.ones(1, 1, size, size)
    start = (size - box) // 2
    mask[..., start : start + box, start : start + box] = 0.0
    return mask


def test_boundary_band_has_nonzero_support_for_center_box():
    mask = _center_box_known()

    bands = boundary_band(mask, width=2)

    assert bands["known_band"].sum().item() > 0
    assert bands["hole_band"].sum().item() > 0
    assert bands["band"].sum().item() > 0


def test_identical_images_have_zero_gradient_and_isophote_error():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 16)
    mask = _center_box_known()

    assert gradient_mismatch(x, x, mask, width=2) < 1e-12
    assert isophote_angle_error(x, x, mask, width=2) < 1e-6
    assert laplacian_mismatch(x, x, mask, width=2) < 1e-12


def test_ns_residual_metric_returns_finite_scalar():
    torch.manual_seed(1)
    x = torch.randn(1, 3, 16, 16)
    mask = _center_box_known()

    value = ns_residual_metric(x, mask, nu=0.1, kappa=0.1, smoothing_sigma=1.0)

    assert isinstance(value, float)
    assert math.isfinite(value)


def test_structural_metrics_support_grayscale_images_and_single_channel_masks():
    x = torch.linspace(-1, 1, 16 * 16).view(1, 1, 16, 16)
    mask = _center_box_known()

    structure = luminance_structure(x, sigma=1.0)
    value = gradient_mismatch(x, x, mask, width=2)

    assert structure.shape == (1, 1, 16, 16)
    assert value < 1e-12
