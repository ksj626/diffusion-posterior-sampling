import torch

from bcns.target_builders import get_target_builder


def _inputs(size=10):
    torch.manual_seed(31)
    mu = torch.randn((1, 3, size, size), dtype=torch.float64)
    mask_known = torch.ones((1, 1, size, size), dtype=torch.float64)
    mask_known[..., 3:7, 3:7] = 0.0
    measurement = mu * mask_known
    return mu, measurement, mask_known


def _assert_builder_output(builder_name):
    mu, measurement, mask_known = _inputs()
    builder = get_target_builder(
        builder_name,
        lambda_structure=0.05,
        structure_sigma=1.0,
        prox_steps=1,
        prox_step_size=0.05,
        poisson_method="dense_reference",
        poisson_tol=1e-8,
    )
    result = builder(mu, measurement, mask_known, tau2=1.0)
    known = mask_known.expand_as(mu).bool()
    assert result.target.shape == mu.shape
    assert torch.isfinite(result.target).all()
    assert torch.allclose(result.target[known], measurement[known])
    assert result.diagnostics["target_builder"] == builder_name
    assert "poisson_num_iter" in result.diagnostics
    assert "poisson_final_residual" in result.diagnostics


def test_harmonic_structure_builder_returns_finite_rgb_target_and_diagnostics():
    _assert_builder_output("harmonic_structure")


def test_poisson_structure_builder_returns_finite_rgb_target_and_diagnostics():
    _assert_builder_output("poisson_structure")


def test_harmonic_structure_defaults_to_zero_rhs():
    mu, measurement, mask_known = _inputs()
    builder = get_target_builder("harmonic_structure", poisson_method="dense_reference")
    result = builder(mu, measurement, mask_known, tau2=1.0)
    assert result.diagnostics["rhs_mode"] == "zero"


def test_poisson_structure_defaults_to_projected_mu_laplacian_rhs():
    mu, measurement, mask_known = _inputs()
    builder = get_target_builder("poisson_structure", poisson_method="dense_reference")
    result = builder(mu, measurement, mask_known, tau2=1.0)
    assert result.diagnostics["rhs_mode"] == "projected_mu_laplacian"
