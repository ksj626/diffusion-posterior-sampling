# BCNS Step 1 Numerical PDE Core

This step adds a standalone numerical core for Boundary-Conditioned Navier-Stokes
experiments. It validates PDE, mask, boundary, Poisson, and pseudo-time flow
components before any diffusion-model integration.

No diffusion sampling code is modified in this step. In particular,
`guided_diffusion/condition_methods.py`, `guided_diffusion/gaussian_diffusion.py`,
`sample_condition.py`, and `configs/` are intentionally left unchanged.

## Conventions

Scalar structure fields use `[B, 1, H, W]`; RGB inputs only appear in boundary
utilities as `[B, 3, H, W]`. Missing masks use `mask_unknown == 1` for pixels in
the hole `Omega`, and `mask_known = 1 - mask_unknown` for observed pixels.

All spatial operators accept grid spacing `h`. The Poisson convention is
`Delta_h I = w` in `Omega` with Dirichlet intensity values on known pixels. CG
solves the equivalent SPD system `-Delta_h I = -w` restricted to unknown pixels;
this is only a sign transformation for the same reconstruction problem.

The structural flow uses `v = nabla_perp I = (-I_y, I_x)`, `w = Delta_h I`, and
`J(I,w) = I_x w_y - I_y w_x = nabla_perp I dot grad(w)`. The implemented RHS is
`dw/dtau = -J_upwind(I,w) + nu div(g grad(w))`, with
`g(s) = 1 / (1 + (s/kappa)^2)` and `s = |grad(G_sigma * w)|`.

## Solvers And Integrators

`bcns.poisson` implements Jacobi, red-black Gauss-Seidel, red-black SOR,
matrix-free CG, and a small-grid dense CPU reference solver. Red-black SOR is
used because its color updates are vectorizable and future-compatible with GPU
batches; no exact optimal omega claim is made for arbitrary masks.

`bcns.integrators` implements FTCS, IMEX backward Euler, and IMEX
Crank-Nicolson. FTCS checks a conservative advection plus explicit diffusion CFL
bound by default. IMEX schemes freeze the nonlinear diffusivity coefficient
within each step and solve implicit diffusion systems with matrix-free CG. IMEX
Crank-Nicolson relaxes the explicit diffusion restriction but does not remove
the explicit-advection CFL consideration.

## Boundary Estimation

Boundary intensity uses normalized known-side smoothing,
`G*(M*y) / (G*M + eps)`, to avoid leaking artificial zero values from the hole.
Boundary vorticity can be estimated either by finite differences or by local
quadratic fitting on the known-side collar. Polynomial fitting falls back to the
finite-difference estimator when a patch has insufficient known pixels.

## Limitations

This step does not implement BCNS local-MAP guidance, Tweedie estimate handling,
reverse diffusion updates, or image-quality claims. It only validates the
standalone numerical PDE core.

## Commands

Run tests in the requested conda environment:

```bash
conda run -n DPS python -m pytest tests/bcns -q
```

Run validation scripts:

```bash
conda run -n DPS python scripts/bcns/validate_poisson.py \
    --device cpu --dtype float64 --output-dir results/bcns_step1/poisson

conda run -n DPS python scripts/bcns/validate_boundary_estimation.py \
    --device cpu --dtype float64 --output-dir results/bcns_step1/boundary

conda run -n DPS python scripts/bcns/validate_integrators.py \
    --device cpu --dtype float64 --output-dir results/bcns_step1/integrators
```

Expected outputs are CSV metrics and PNG plots under `results/bcns_step1/`.
