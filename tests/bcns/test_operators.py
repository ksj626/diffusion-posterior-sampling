import torch

from bcns.operators import (
    divergence,
    gradient_central,
    gradient_forward,
    laplacian_5pt,
    perpendicular_gradient,
)


def test_laplacian_constant_zero_interior_cpu_float64():
    x = torch.ones((1, 1, 8, 9), dtype=torch.float64)
    lap = laplacian_5pt(x, h=0.5)
    assert lap.dtype == torch.float64
    assert torch.allclose(lap[..., 1:-1, 1:-1], torch.zeros_like(lap[..., 1:-1, 1:-1]))


def test_central_gradient_linear_field_interior():
    h = 0.25
    height, width = 9, 10
    y = torch.arange(height, dtype=torch.float64).view(height, 1) * h
    x = torch.arange(width, dtype=torch.float64).view(1, width) * h
    field = (2.0 * x + 3.0 * y).view(1, 1, height, width)
    dx, dy = gradient_central(field, h)
    assert torch.allclose(dx[..., 1:-1, 1:-1], torch.full_like(dx[..., 1:-1, 1:-1], 2.0))
    assert torch.allclose(dy[..., 1:-1, 1:-1], torch.full_like(dy[..., 1:-1, 1:-1], 3.0))


def test_divergence_gradient_shape_dtype_preservation():
    x = torch.randn((2, 1, 7, 8), dtype=torch.float64)
    px, py = gradient_forward(x, h=1.0)
    div = divergence(px, py, h=1.0)
    assert div.shape == x.shape
    assert div.dtype == x.dtype


def test_perpendicular_gradient_convention():
    h = 1.0
    height, width = 6, 7
    y = torch.arange(height, dtype=torch.float64).view(height, 1)
    x = torch.arange(width, dtype=torch.float64).view(1, width)
    field = (5.0 * x + 2.0 * y).view(1, 1, height, width)
    vx, vy = perpendicular_gradient(field, h)
    assert torch.allclose(vx[..., 1:-1, 1:-1], torch.full_like(vx[..., 1:-1, 1:-1], -2.0))
    assert torch.allclose(vy[..., 1:-1, 1:-1], torch.full_like(vy[..., 1:-1, 1:-1], 5.0))
