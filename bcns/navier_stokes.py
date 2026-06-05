"""Boundary-conditioned Navier-Stokes structural flow operators."""

from typing import Any, Dict, Tuple

import torch

from .operators import (
    _validate_scalar_field,
    gradient_central,
    gradient_forward,
    laplacian_5pt,
    masked_l2_norm,
    perpendicular_gradient,
)
from .structure import gaussian_smooth


def anisotropic_diffusivity(
    w: torch.Tensor,
    kappa: float,
    smoothing_sigma: float,
    h: float,
) -> torch.Tensor:
    """Return ``g(|grad(G_sigma * w)|) = 1 / (1 + (s/kappa)^2)``."""

    _validate_scalar_field(w, "w")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    smoothed = gaussian_smooth(w, smoothing_sigma)
    gx, gy = gradient_central(smoothed, h)
    magnitude = torch.sqrt(gx * gx + gy * gy)
    return 1.0 / (1.0 + (magnitude / kappa) ** 2)


def anisotropic_diffusion_operator(
    w: torch.Tensor,
    diffusivity: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Return finite-volume ``div(g * grad(w))`` with zero flux at image edges."""

    _validate_scalar_field(w, "w")
    if diffusivity.shape != w.shape:
        raise ValueError("diffusivity must match w shape.")
    out = torch.zeros_like(w)
    g_x = 0.5 * (diffusivity[..., :, 1:] + diffusivity[..., :, :-1])
    flux_x = g_x * (w[..., :, 1:] - w[..., :, :-1]) / h
    out[..., :, :-1] += flux_x / h
    out[..., :, 1:] -= flux_x / h
    g_y = 0.5 * (diffusivity[..., 1:, :] + diffusivity[..., :-1, :])
    flux_y = g_y * (w[..., 1:, :] - w[..., :-1, :]) / h
    out[..., :-1, :] += flux_y / h
    out[..., 1:, :] -= flux_y / h
    return out


def jacobian_central(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Return diagnostic central-difference ``J(I,w) = I_x w_y - I_y w_x``."""

    _validate_scalar_field(intensity, "intensity")
    if vorticity.shape != intensity.shape:
        raise ValueError("vorticity must match intensity shape.")
    ix, iy = gradient_central(intensity, h)
    wx, wy = gradient_central(vorticity, h)
    return ix * wy - iy * wx


def _gradient_backward(x: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, 1:] = (x[..., :, 1:] - x[..., :, :-1]) / h
    dx[..., :, 0] = (x[..., :, 1] - x[..., :, 0]) / h
    dy[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / h
    dy[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / h
    return dx, dy


def jacobian_upwind(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Return upwind ``J(I,w) = nabla_perp(I) dot grad(w)`` for integration."""

    _validate_scalar_field(intensity, "intensity")
    if vorticity.shape != intensity.shape:
        raise ValueError("vorticity must match intensity shape.")
    vx, vy = perpendicular_gradient(intensity, h)
    fw_dx, fw_dy = gradient_forward(vorticity, h)
    bw_dx, bw_dy = _gradient_backward(vorticity, h)
    dw_dx = torch.where(vx >= 0, bw_dx, fw_dx)
    dw_dy = torch.where(vy >= 0, bw_dy, fw_dy)
    return vx * dw_dx + vy * dw_dy


def ns_rhs(
    intensity: torch.Tensor,
    vorticity: torch.Tensor,
    nu: float,
    kappa: float,
    smoothing_sigma: float,
    h: float,
    use_upwind: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return ``-J(I,w) + nu * div(g grad(w))`` plus scalar diagnostics."""

    if nu < 0:
        raise ValueError("nu must be non-negative.")
    jacobian = jacobian_upwind(intensity, vorticity, h) if use_upwind else jacobian_central(intensity, vorticity, h)
    diffusivity = anisotropic_diffusivity(vorticity, kappa, smoothing_sigma, h)
    diffusion = anisotropic_diffusion_operator(vorticity, diffusivity, h)
    rhs = -jacobian + nu * diffusion
    vx, vy = perpendicular_gradient(intensity, h)
    velocity_mag = torch.sqrt(vx * vx + vy * vy)
    full_mask = torch.ones_like(vorticity)
    diagnostics = {
        "max_velocity": float(velocity_mag.max().item()),
        "max_diffusivity": float(diffusivity.max().item()),
        "jacobian_norm": float(masked_l2_norm(jacobian, full_mask).item()),
        "diffusion_norm": float(masked_l2_norm(diffusion, full_mask).item()),
        "rhs_norm": float(masked_l2_norm(rhs, full_mask).item()),
    }
    return rhs, diagnostics
