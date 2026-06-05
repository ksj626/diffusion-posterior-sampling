"""Configuration dataclasses for the BCNS numerical core."""

from dataclasses import dataclass, field


POISSON_METHODS = {"jacobi", "gs_rb", "sor_rb", "cg", "dense_reference"}
INTEGRATORS = {"ftcs", "imex_be", "imex_cn"}
VORTICITY_ESTIMATORS = {"fd", "finite_difference", "polynomial"}
VORTICITY_BOUNDARY_MODES = {"hard", "none"}


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")


@dataclass
class PoissonSolverConfig:
    """Configuration for masked Poisson reconstruction."""

    method: str
    max_iter: int = 5000
    tol: float = 1e-8
    omega: float = 1.7
    h: float = 1.0
    record_history: bool = True

    def __post_init__(self) -> None:
        if self.method not in POISSON_METHODS:
            raise ValueError(
                f"Unsupported Poisson method {self.method!r}; "
                f"expected one of {sorted(POISSON_METHODS)}."
            )
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        _require_positive("tol", self.tol)
        _require_positive("h", self.h)
        if not (0.0 < self.omega < 2.0):
            raise ValueError("omega must satisfy 0 < omega < 2 for SOR.")


@dataclass
class BoundaryConfig:
    """Configuration for known-side boundary trace estimation."""

    gaussian_sigma: float = 1.0
    collar_width: int = 3
    vorticity_estimator: str = "polynomial"
    polynomial_radius: int = 2
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _require_nonnegative("gaussian_sigma", self.gaussian_sigma)
        if self.collar_width <= 0:
            raise ValueError("collar_width must be positive.")
        if self.vorticity_estimator not in VORTICITY_ESTIMATORS:
            raise ValueError(
                f"Unsupported vorticity estimator {self.vorticity_estimator!r}; "
                f"expected one of {sorted(VORTICITY_ESTIMATORS)}."
            )
        if self.polynomial_radius <= 0:
            raise ValueError("polynomial_radius must be positive.")
        _require_positive("eps", self.eps)


@dataclass
class FlowConfig:
    """Configuration for BCNS pseudo-time evolution."""

    integrator: str
    poisson: PoissonSolverConfig
    dt: float = 1e-3
    pseudo_time: float = 0.05
    nu: float = 0.1
    kappa: float = 0.1
    smoothing_sigma: float = 1.0
    cfl: float = 0.25
    check_cfl: bool = True
    vorticity_boundary_mode: str = "hard"
    implicit_max_iter: int = field(default=500)
    implicit_tol: float = field(default=1e-8)

    def __post_init__(self) -> None:
        if self.integrator not in INTEGRATORS:
            raise ValueError(
                f"Unsupported integrator {self.integrator!r}; "
                f"expected one of {sorted(INTEGRATORS)}."
            )
        _require_positive("dt", self.dt)
        _require_nonnegative("pseudo_time", self.pseudo_time)
        _require_nonnegative("nu", self.nu)
        _require_positive("kappa", self.kappa)
        _require_nonnegative("smoothing_sigma", self.smoothing_sigma)
        _require_positive("cfl", self.cfl)
        if self.vorticity_boundary_mode not in VORTICITY_BOUNDARY_MODES:
            raise ValueError(
                f"Unsupported vorticity boundary mode {self.vorticity_boundary_mode!r}; "
                f"expected one of {sorted(VORTICITY_BOUNDARY_MODES)}."
            )
        if self.implicit_max_iter <= 0:
            raise ValueError("implicit_max_iter must be positive.")
        _require_positive("implicit_tol", self.implicit_tol)
