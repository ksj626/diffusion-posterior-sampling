import torch

from bcns.dps_adapter import dps_known_to_bcns_unknown
from bcns.poisson_targets import (
    build_poisson_structure_target,
    make_boundary_values,
    make_poisson_rhs,
)


def _inputs(size=12):
    coords = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    base = torch.sin(2.0 * xx) + 0.25 * yy * yy + 0.1 * xx * yy
    mu_scalar = base.view(1, 1, size, size)
    mu = mu_scalar.repeat(1, 3, 1, 1)
    mask_known = torch.ones((1, 1, size, size), dtype=torch.float64)
    mask_known[..., size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 0.0
    measurement = mu * mask_known
    return mu, measurement, mask_known


def test_harmonic_target_shape_and_finite_values():
    mu, measurement, mask_known = _inputs()
    result = build_poisson_structure_target(
        mu,
        measurement,
        mask_known,
        sigma=0.8,
        rhs_mode="zero",
        poisson_method="dense_reference",
        poisson_tol=1e-10,
    )
    assert result.structure.shape == (1, 1, 12, 12)
    assert torch.isfinite(result.structure).all()
    assert result.boundary_values.shape == result.structure.shape
    assert result.rhs_w.shape == result.structure.shape


def test_known_region_equals_boundary_values():
    mu, measurement, mask_known = _inputs()
    result = build_poisson_structure_target(
        mu,
        measurement,
        mask_known,
        sigma=1.0,
        rhs_mode="zero",
        poisson_method="dense_reference",
    )
    known = mask_known.bool()
    assert torch.allclose(result.structure[known], result.boundary_values[known])


def test_dense_harmonic_residual_is_below_tolerance():
    mu, measurement, mask_known = _inputs()
    result = build_poisson_structure_target(
        mu,
        measurement,
        mask_known,
        sigma=1.0,
        rhs_mode="zero",
        poisson_method="dense_reference",
        poisson_tol=1e-8,
    )
    assert result.diagnostics["poisson_final_residual"] < 1e-8
    assert result.diagnostics["poisson_converged"]


def test_rhs_mode_zero_gives_zero_rhs_in_hole():
    mu, measurement, mask_known = _inputs()
    rhs = make_poisson_rhs(
        mu,
        measurement,
        mask_known,
        sigma=1.0,
        rhs_mode="zero",
        h=1.0,
    )
    mask_unknown = dps_known_to_bcns_unknown(mask_known)
    assert torch.equal(rhs * mask_unknown, torch.zeros_like(rhs))


def test_rhs_mode_mu_laplacian_is_nonzero_for_nontrivial_image():
    mu, measurement, mask_known = _inputs()
    rhs = make_poisson_rhs(
        mu,
        measurement,
        mask_known,
        sigma=0.0,
        rhs_mode="mu_laplacian",
        h=1.0,
    )
    mask_unknown = dps_known_to_bcns_unknown(mask_known)
    assert (rhs * mask_unknown).abs().sum().item() > 0.0


def test_boundary_values_support_normalized_known_smooth():
    mu, measurement, mask_known = _inputs()
    boundary = make_boundary_values(
        measurement,
        mask_known,
        sigma=1.0,
        mode="normalized_known_smooth",
    )
    assert boundary.shape == (1, 1, 12, 12)
    assert torch.isfinite(boundary).all()
