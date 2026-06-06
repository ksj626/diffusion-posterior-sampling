import torch

from bcns.target_builders import get_target_builder


def _inputs(size=10):
    torch.manual_seed(41)
    mu = torch.randn((1, 3, size, size), dtype=torch.float64)
    mask_known = torch.ones((1, 1, size, size), dtype=torch.float64)
    mask_known[..., 3:7, 3:7] = 0.0
    measurement = mu * mask_known
    return mu, measurement, mask_known


def test_flow_structure_builder_returns_finite_rgb_target_and_diagnostics():
    mu, measurement, mask_known = _inputs()
    builder = get_target_builder(
        "flow_structure",
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=1,
        prox_step_size=0.05,
        integrator="imex_be",
        poisson_method="dense_reference",
        pseudo_time=1e-3,
        dt=1e-3,
    )
    result = builder(mu, measurement, mask_known, tau2=1.0)
    known = mask_known.expand_as(mu).bool()
    assert result.target.shape == mu.shape
    assert torch.isfinite(result.target).all()
    assert torch.allclose(result.target[known], measurement[known])
    assert result.diagnostics["target_builder"] == "flow_structure"
    assert result.diagnostics["flow_integrator"] == "imex_be"
    assert "flow_runtime_sec" in result.diagnostics
    assert "flow_num_steps" in result.diagnostics
    assert "flow_structure_disp" in result.diagnostics
