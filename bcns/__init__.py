"""Boundary-conditioned Navier-Stokes numerical core for DPS experiments.

This package is standalone in Step 1. It does not modify or call diffusion
sampling code.
"""

from .config import BoundaryConfig, FlowConfig, PoissonSolverConfig

__all__ = ["BoundaryConfig", "FlowConfig", "PoissonSolverConfig"]
