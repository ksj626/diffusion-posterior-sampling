import torch

from bcns.dps_adapter import hard_project_clean
from bcns.losses import target_displacement_stats
from bcns.luminance_lift import luminance_lift_rgb_target
from bcns.proximal import structure_image


def _inputs():
    torch.manual_seed(461)
    shape = (1, 3, 6, 6)
    mu = torch.randn(shape, dtype=torch.float64)
    measurement = torch.randn(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 6, 6), dtype=torch.float64)
    mask_known[..., 2:5, 2:5] = 0.0
    target_structure = structure_image(mu, 0.0).detach() + 0.4 * (1.0 - mask_known)
    return mu, measurement, mask_known, target_structure


def test_luminance_lift_target_shape_and_known_projection():
    mu, measurement, mask_known, target_structure = _inputs()
    target = luminance_lift_rgb_target(mu, measurement, mask_known, target_structure, structure_sigma=0.0)
    assert target.shape == mu.shape
    assert torch.allclose(target * mask_known, measurement * mask_known)


def test_luminance_lift_nonzero_hole_displacement_when_structure_differs():
    mu, measurement, mask_known, target_structure = _inputs()
    target = luminance_lift_rgb_target(mu, measurement, mask_known, target_structure, structure_sigma=0.0)
    stats = target_displacement_stats(mu, target, mask_known)
    assert stats["target_disp_hole"].item() > 0.0
    assert torch.isfinite(target).all()


def test_luminance_lift_scale_zero_matches_hard_projection():
    mu, measurement, mask_known, target_structure = _inputs()
    target = luminance_lift_rgb_target(
        mu,
        measurement,
        mask_known,
        target_structure,
        structure_sigma=0.0,
        lift_scale=0.0,
    )
    assert torch.allclose(target, hard_project_clean(mu, measurement, mask_known))


def test_luminance_lift_luma_weights_is_finite_and_projects_known_pixels():
    mu, measurement, mask_known, target_structure = _inputs()
    target = luminance_lift_rgb_target(
        mu,
        measurement,
        mask_known,
        target_structure,
        structure_sigma=0.0,
        mode="luma_weights",
    )
    assert torch.isfinite(target).all()
    assert torch.allclose(target * mask_known, measurement * mask_known)
