import torch

from bcns.config import PoissonSolverConfig
from bcns.diagnostics import max_abs_error, relative_l2_error
from bcns.manufactured import make_poisson_manufactured_solution
from bcns.masks import make_center_box_mask
from bcns.poisson import (
    solve_poisson,
    solve_poisson_cg,
    solve_poisson_dense_reference,
    solve_poisson_jacobi,
)


def _problem(size=20):
    h = 1.0 / (size - 1)
    intensity, rhs = make_poisson_manufactured_solution(size, h, dtype=torch.float64)
    mask = make_center_box_mask(size, size, 6, 6, dtype=torch.float64)
    return h, intensity, rhs, mask


def test_dense_reference_recovery():
    h, intensity, rhs, mask = _problem(18)
    result = solve_poisson_dense_reference(rhs, intensity, mask, h)
    assert max_abs_error(result.solution, intensity, mask) < 1e-8


def test_cg_matches_dense_reference():
    h, intensity, rhs, mask = _problem(18)
    dense = solve_poisson_dense_reference(rhs, intensity, mask, h)
    config = PoissonSolverConfig(method="cg", h=h, tol=1e-10, max_iter=500)
    cg = solve_poisson_cg(rhs, intensity, mask, config)
    assert relative_l2_error(cg.solution, dense.solution, mask) < 1e-6
    assert cg.residual_history[-1] < 1e-7


def test_sor_converges_by_orders_of_magnitude():
    h, intensity, rhs, mask = _problem(24)
    config = PoissonSolverConfig(method="sor_rb", h=h, tol=1e-9, max_iter=1000, omega=1.7)
    result = solve_poisson(rhs, intensity, mask, config)
    assert result.residual_history[-1] < result.residual_history[0] * 1e-3 or result.converged


def test_known_pixels_fixed_for_all_solvers():
    h, intensity, rhs, mask = _problem(16)
    known = (1.0 - mask).bool()
    configs = [
        PoissonSolverConfig(method="jacobi", h=h, max_iter=3, tol=1e-12),
        PoissonSolverConfig(method="gs_rb", h=h, max_iter=3, tol=1e-12),
        PoissonSolverConfig(method="sor_rb", h=h, max_iter=3, tol=1e-12),
        PoissonSolverConfig(method="cg", h=h, max_iter=20, tol=1e-12),
        PoissonSolverConfig(method="dense_reference", h=h, max_iter=1, tol=1e-12),
    ]
    for config in configs:
        result = solve_poisson(rhs, intensity, mask, config)
        assert torch.allclose(result.solution[known], intensity[known], atol=1e-12)
