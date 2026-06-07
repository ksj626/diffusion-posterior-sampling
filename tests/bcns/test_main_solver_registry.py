import math

import pytest
import torch

from bcns.config import PoissonSolverConfig
from bcns.poisson import solve_poisson
from bcns.target_builders import get_target_builder


def _tiny_problem():
    rhs = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    boundary = torch.zeros_like(rhs)
    mask_unknown = torch.zeros_like(rhs)
    mask_unknown[..., 2:6, 2:6] = 1.0
    return rhs, boundary, mask_unknown


@pytest.mark.parametrize("method", ["sor", "cg", "ftcs", "be", "cn"])
def test_poisson_solver_registry_accepts_main_solver_names(method):
    rhs, boundary, mask_unknown = _tiny_problem()
    config = PoissonSolverConfig(method=method, max_iter=2, tol=1e-4, dt=0.2)

    result = solve_poisson(rhs, boundary, mask_unknown, config)

    assert result.solution.shape == rhs.shape
    assert torch.isfinite(result.solution).all()
    assert math.isfinite(result.residual_history[-1])


def test_poisson_solver_registry_rejects_invalid_method():
    with pytest.raises(ValueError, match="Unsupported Poisson method"):
        PoissonSolverConfig(method="not_a_solver")


@pytest.mark.parametrize("method", ["sor", "cg", "ftcs", "be", "cn"])
def test_harmonic_and_poisson_builders_accept_main_solver_names(method):
    harmonic = get_target_builder("harmonic_structure", poisson_method=method, poisson_max_iter=2)
    poisson = get_target_builder("poisson_structure", poisson_method=method, poisson_max_iter=2)

    assert harmonic.poisson_method == method
    assert poisson.poisson_method == method
