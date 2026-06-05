import torch

from bcns.boundary import estimate_boundary_data
from bcns.config import BoundaryConfig
from bcns.masks import make_center_box_mask, make_thin_scratch_mask
from bcns.structure import gaussian_smooth, masked_normalized_gaussian_smooth


def test_normalized_smoothing_does_not_leak_masked_zeros():
    size = 32
    field = torch.ones((1, 1, size, size), dtype=torch.float64)
    mask = make_center_box_mask(size, size, 8, 8, dtype=torch.float64)
    known = 1.0 - mask
    zero_masked = field * known
    direct = gaussian_smooth(zero_masked, sigma=2.0)
    normalized = masked_normalized_gaussian_smooth(zero_masked, known, sigma=2.0)
    boundary_probe = mask.bool()
    assert normalized[boundary_probe].mean() > direct[boundary_probe].mean() + 0.2
    assert torch.allclose(normalized[boundary_probe], torch.ones_like(normalized[boundary_probe]), atol=1e-6)


def test_polynomial_vorticity_estimates_quadratic_laplacian():
    size = 20
    y = torch.arange(size, dtype=torch.float64).view(size, 1)
    x = torch.arange(size, dtype=torch.float64).view(1, size)
    a3, a5 = 0.7, -0.2
    field = 1.0 + 0.3 * x - 0.1 * y + a3 * x * x + 0.4 * x * y + a5 * y * y
    image = field.view(1, 1, size, size)
    mask = make_center_box_mask(size, size, 2, 2, dtype=torch.float64)
    observed = image * (1.0 - mask)
    config = BoundaryConfig(
        gaussian_sigma=0.0,
        collar_width=4,
        polynomial_radius=4,
        vorticity_estimator="polynomial",
    )
    data = estimate_boundary_data(observed, mask, config)
    expected = torch.full_like(data.vorticity_trace, 2.0 * a3 + 2.0 * a5)
    assert data.diagnostics["fallback_ratio"] == 0.0
    assert torch.max((data.vorticity_trace[mask.bool()] - expected[mask.bool()]).abs()).item() < 1e-8


def test_polynomial_fallback_handles_thin_mask():
    size = 16
    field = torch.randn((1, 1, size, size), dtype=torch.float64)
    mask = make_thin_scratch_mask(size, size, thickness=1, dtype=torch.float64)
    config = BoundaryConfig(
        gaussian_sigma=0.0,
        collar_width=1,
        polynomial_radius=1,
        vorticity_estimator="polynomial",
    )
    data = estimate_boundary_data(field * (1.0 - mask), mask, config)
    assert torch.isfinite(data.vorticity_trace).all()
    assert data.diagnostics["fallback_count"] >= 0
