import pytest
import torch

from bcns.dps_adapter import hard_project_clean
from bcns.flow_targets import (
    build_flow_structure_target,
    make_boundary_intensity,
    make_boundary_vorticity,
    make_initial_intensity,
)
from bcns.proximal import structure_image


def _inputs(size=10):
    coords = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    scalar = torch.sin(2.0 * xx) + 0.2 * yy + 0.1 * xx * yy
    mu = scalar.view(1, 1, size, size).repeat(1, 3, 1, 1)
    mask_known = torch.ones((1, 1, size, size), dtype=torch.float64)
    mask_known[..., 3:7, 3:7] = 0.0
    measurement = mu * mask_known
    return mu, measurement, mask_known


def test_initial_and_boundary_modes_return_scalar_fields():
    mu, measurement, mask_known = _inputs()
    for mode in ("mu", "projected_mu", "normalized_known_smooth"):
        out = make_initial_intensity(mu, measurement, mask_known, sigma=1.0, mode=mode)
        assert out.shape == (1, 1, 10, 10)
        assert torch.isfinite(out).all()
    boundary = make_boundary_intensity(measurement, mask_known, sigma=1.0)
    assert boundary.shape == (1, 1, 10, 10)
    assert torch.isfinite(boundary).all()


def test_boundary_vorticity_modes():
    mu, measurement, mask_known = _inputs()
    assert make_boundary_vorticity(measurement, mask_known, sigma=1.0, mode="none") is None
    fd = make_boundary_vorticity(measurement, mask_known, sigma=1.0, mode="fd_laplacian", h=1.0)
    assert fd.shape == (1, 1, 10, 10)
    assert torch.isfinite(fd).all()


def test_build_flow_structure_target_shape_finite_and_known_boundary():
    mu, measurement, mask_known = _inputs()
    result = build_flow_structure_target(
        mu,
        measurement,
        mask_known,
        integrator="imex_be",
        poisson_method="dense_reference",
        pseudo_time=1e-3,
        dt=1e-3,
        poisson_tol=1e-8,
    )
    assert result.structure.shape == (1, 1, 10, 10)
    assert torch.isfinite(result.structure).all()
    known = mask_known.bool()
    assert torch.allclose(result.structure[known], result.boundary_intensity[known])
    assert result.diagnostics["flow_integrator"] == "imex_be"
    assert result.diagnostics["flow_num_steps"] == 1


def test_pseudo_time_zero_returns_boundary_consistent_initial_intensity():
    mu, measurement, mask_known = _inputs()
    result = build_flow_structure_target(
        mu,
        measurement,
        mask_known,
        integrator="imex_be",
        poisson_method="dense_reference",
        pseudo_time=0.0,
        dt=1e-3,
    )
    initial = structure_image(hard_project_clean(mu, measurement, mask_known), sigma=1.0)
    expected = initial * (1.0 - mask_known) + result.boundary_intensity * mask_known
    assert torch.allclose(result.structure, expected)
    assert result.diagnostics["flow_num_steps"] == 0


def test_be_and_cn_modes_run_without_nan():
    mu, measurement, mask_known = _inputs()
    for integrator in ("imex_be", "imex_cn"):
        result = build_flow_structure_target(
            mu,
            measurement,
            mask_known,
            integrator=integrator,
            poisson_method="dense_reference",
            pseudo_time=1e-3,
            dt=1e-3,
        )
        assert torch.isfinite(result.structure).all()
        assert not result.diagnostics["flow_has_nan"]


def test_ftcs_oversized_dt_raises_clear_cfl_message():
    mu, measurement, mask_known = _inputs()
    with pytest.raises(ValueError, match="FTCS CFL check failed"):
        build_flow_structure_target(
            mu,
            measurement,
            mask_known,
            integrator="ftcs",
            poisson_method="dense_reference",
            pseudo_time=1.0,
            dt=10.0,
            check_cfl=True,
        )
