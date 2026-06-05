import torch
import pytest

from bcns.config import FlowConfig, PoissonSolverConfig
from bcns.integrators import (
    ftcs_step,
    imex_backward_euler_step,
    imex_crank_nicolson_step,
)
from bcns.masks import make_center_box_mask


def _config(integrator="ftcs", dt=0.01, check_cfl=True):
    return FlowConfig(
        integrator=integrator,
        poisson=PoissonSolverConfig(method="cg", h=1.0, tol=1e-8, max_iter=100),
        dt=dt,
        pseudo_time=dt,
        nu=0.1,
        kappa=10.0,
        smoothing_sigma=0.0,
        check_cfl=check_cfl,
        implicit_max_iter=100,
        implicit_tol=1e-10,
    )


def test_ftcs_cfl_validation_raises():
    size = 12
    x = torch.arange(size, dtype=torch.float64).view(1, 1, 1, size).expand(1, 1, size, size)
    vorticity = torch.zeros_like(x)
    mask = torch.ones_like(x)
    with pytest.raises(ValueError):
        ftcs_step(x, vorticity, mask, None, _config(dt=10.0, check_cfl=True))


def test_pure_diffusion_integrators_produce_finite_tensors():
    size = 14
    intensity = torch.zeros((1, 1, size, size), dtype=torch.float64)
    vorticity = torch.ones_like(intensity)
    mask = torch.ones_like(intensity)
    for name, fn in (
        ("ftcs", ftcs_step),
        ("imex_be", imex_backward_euler_step),
        ("imex_cn", imex_crank_nicolson_step),
    ):
        result = fn(intensity, vorticity, mask, None, _config(integrator=name, dt=0.01))
        assert torch.isfinite(result.vorticity).all()


def test_cn_not_substantially_worse_than_be_on_constant_diffusion_mode():
    size = 14
    intensity = torch.zeros((1, 1, size, size), dtype=torch.float64)
    vorticity = torch.ones_like(intensity)
    mask = torch.ones_like(intensity)
    target = vorticity
    be = imex_backward_euler_step(intensity, vorticity, mask, None, _config("imex_be", dt=0.05))
    cn = imex_crank_nicolson_step(intensity, vorticity, mask, None, _config("imex_cn", dt=0.05))
    be_err = torch.linalg.norm(be.vorticity - target)
    cn_err = torch.linalg.norm(cn.vorticity - target)
    assert cn_err <= 5.0 * be_err + 1e-12


def test_hard_boundary_preserved_during_step():
    size = 16
    intensity = torch.zeros((1, 1, size, size), dtype=torch.float64)
    vorticity = torch.zeros_like(intensity)
    mask = make_center_box_mask(size, size, 4, 4, dtype=torch.float64)
    boundary = torch.full_like(vorticity, 7.0)
    result = ftcs_step(intensity, vorticity, mask, boundary, _config(dt=0.01))
    assert torch.allclose(result.vorticity[(1.0 - mask).bool()], boundary[(1.0 - mask).bool()])
