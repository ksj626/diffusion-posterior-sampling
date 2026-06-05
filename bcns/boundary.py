"""Known-side boundary intensity and vorticity estimation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch

from .config import BoundaryConfig
from .masks import hole_boundary, known_collar
from .operators import _validate_scalar_field, laplacian_5pt
from .structure import masked_normalized_gaussian_smooth


@dataclass
class BoundaryData:
    """Boundary traces and masks estimated from observed pixels."""

    intensity_trace: torch.Tensor
    vorticity_trace: torch.Tensor
    boundary_mask: torch.Tensor
    collar_mask: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _validate_inputs(observed_luminance: torch.Tensor, mask_unknown: torch.Tensor) -> None:
    _validate_scalar_field(observed_luminance, "observed_luminance")
    if mask_unknown.shape != observed_luminance.shape:
        raise ValueError("mask_unknown must match observed_luminance shape.")
    if not torch.all((mask_unknown.detach() == 0) | (mask_unknown.detach() == 1)):
        raise ValueError("mask_unknown must be binary.")


def estimate_boundary_intensity(
    observed_luminance: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: BoundaryConfig,
) -> torch.Tensor:
    """Estimate an intensity trace using normalized known-side smoothing."""

    _validate_inputs(observed_luminance, mask_unknown)
    mask_known = 1.0 - mask_unknown.to(observed_luminance.dtype)
    smoothed = masked_normalized_gaussian_smooth(
        observed_luminance,
        mask_known,
        sigma=config.gaussian_sigma,
        eps=config.eps,
    )
    return torch.where(mask_unknown.bool(), smoothed, observed_luminance)


def estimate_boundary_vorticity_fd(
    observed_luminance: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: BoundaryConfig,
) -> torch.Tensor:
    """Estimate vorticity as the Laplacian of normalized smoothed luminance."""

    _validate_inputs(observed_luminance, mask_unknown)
    intensity = estimate_boundary_intensity(observed_luminance, mask_unknown, config)
    return laplacian_5pt(intensity, h=1.0)


def _quadratic_fit_value(
    values: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    center_y: int,
    center_x: int,
) -> torch.Tensor:
    dy = (yy.to(values.dtype) - float(center_y)).reshape(-1)
    dx = (xx.to(values.dtype) - float(center_x)).reshape(-1)
    design = torch.stack(
        [
            torch.ones_like(dx),
            dx,
            dy,
            dx * dx,
            dx * dy,
            dy * dy,
        ],
        dim=1,
    )
    target = values.reshape(-1, 1)
    coeff = torch.linalg.pinv(design).matmul(target).reshape(-1)
    return 2.0 * coeff[3] + 2.0 * coeff[5]


def _estimate_boundary_vorticity_polynomial_with_diagnostics(
    observed_luminance: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: BoundaryConfig,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    _validate_inputs(observed_luminance, mask_unknown)
    fd = estimate_boundary_vorticity_fd(observed_luminance, mask_unknown, config)
    collar = known_collar(mask_unknown, config.collar_width).bool()
    boundary = hole_boundary(mask_unknown).bool()
    out = fd.clone()
    radius = config.polynomial_radius
    fallback_count = 0
    fit_count = 0
    batch, _, height, width = observed_luminance.shape
    for b in range(batch):
        coords = torch.nonzero(boundary[b, 0], as_tuple=False)
        for coord in coords:
            y = int(coord[0].item())
            x = int(coord[1].item())
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            patch_known = collar[b, 0, y0:y1, x0:x1]
            if int(patch_known.sum().item()) < 6:
                fallback_count += 1
                continue
            local = torch.nonzero(patch_known, as_tuple=False)
            yy = local[:, 0] + y0
            xx = local[:, 1] + x0
            vals = observed_luminance[b, 0, yy, xx]
            try:
                lap = _quadratic_fit_value(vals, yy, xx, y, x)
            except RuntimeError:
                fallback_count += 1
                continue
            out[b, 0, y, x] = lap
            fit_count += 1
    total = fit_count + fallback_count
    diagnostics = {
        "fit_count": fit_count,
        "fallback_count": fallback_count,
        "fallback_ratio": float(fallback_count / total) if total else 0.0,
    }
    return out, diagnostics


def estimate_boundary_vorticity_polynomial(
    observed_luminance: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: BoundaryConfig,
) -> torch.Tensor:
    """Estimate boundary vorticity by local quadratic least-squares fitting."""

    out, _ = _estimate_boundary_vorticity_polynomial_with_diagnostics(
        observed_luminance, mask_unknown, config
    )
    return out


def estimate_boundary_data(
    observed_luminance: torch.Tensor,
    mask_unknown: torch.Tensor,
    config: BoundaryConfig,
) -> BoundaryData:
    """Estimate intensity and vorticity traces with masks and diagnostics."""

    intensity = estimate_boundary_intensity(observed_luminance, mask_unknown, config)
    boundary = hole_boundary(mask_unknown)
    collar = known_collar(mask_unknown, config.collar_width)
    diagnostics: Dict[str, Any] = {}
    if config.vorticity_estimator == "polynomial":
        vorticity, diagnostics = _estimate_boundary_vorticity_polynomial_with_diagnostics(
            observed_luminance, mask_unknown, config
        )
    else:
        vorticity = estimate_boundary_vorticity_fd(observed_luminance, mask_unknown, config)
    diagnostics["vorticity_estimator"] = config.vorticity_estimator
    return BoundaryData(
        intensity_trace=intensity,
        vorticity_trace=vorticity,
        boundary_mask=boundary,
        collar_mask=collar,
        diagnostics=diagnostics,
    )
