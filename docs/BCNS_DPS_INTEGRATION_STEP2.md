# BCNS DPS Integration Step 2

Step 2 adds harmonic and Poisson scalar structure targets to the existing DPS target-guidance path. It connects the BCNS masked Poisson solver to the target builder pipeline, then uses the existing RGB structure-proximal objective to build a detached target for hole-only guidance.

This step is still not Navier-Stokes. It does not add vorticity transport, FTCS, IMEX-BE, IMEX-CN, CN, SOR/CG solver changes, or local-MAP diffusion solvers inside sampling.

## Targets

`normalized_known_smooth` is the Step 1.5 baseline. It smooths only observed luminance and normalizes by a smoothed known mask so zero-valued hole pixels do not leak into the target.

`harmonic_structure` solves the masked Laplace problem inside the hole:

```text
Delta_h I = 0
```

The known region is fixed to a normalized known-side boundary trace from the measurement. Harmonic targets are stable and easy to interpret, but they may oversmooth across large holes.

`poisson_structure` solves:

```text
Delta_h I = w_mu
```

The default source is `projected_mu_laplacian`, computed from `Delta_h S_sigma(hard_project_clean(mu, measurement, mask_known))`. This can preserve more structure than a purely harmonic fill while still anchoring the known region.

## Fixed Projection Baselines

The original DPS `projection` and `mcg` methods remain registered for reference. Step 2 adds fixed inpainting baselines instead:

- `projection_fixed`: directly replaces known pixels with the timestep-noisy measurement.
- `mcg_fixed`: applies the original MCG gradient step, then replaces known pixels with the timestep-noisy measurement.

These fixed rows are used in Step 2 comparisons so the raw known region is interpretable. Final clean composites are still reported for every method.

## Masks

Random masks are useful for broad smoke tests, but they are poor PDE debug masks because the hole geometry varies per image and may include disconnected speckles. Step 2 comparison supports deterministic `center_box` and `thin_scratch` masks so harmonic and Poisson behavior can be inspected more clearly.

## Commands

Run unit tests:

```bash
conda run -n DPS python -m pytest \
  tests/bcns/test_dps_adapter.py \
  tests/bcns/test_target_guidance.py \
  tests/bcns/test_structure_proximal.py \
  tests/bcns/test_projection_wrappers.py \
  tests/bcns/test_normalized_known_smooth.py \
  tests/bcns/test_bcns_target_projection_option.py \
  tests/bcns/test_poisson_targets.py \
  tests/bcns/test_poisson_target_builders.py \
  tests/bcns/test_fixed_projection_methods.py \
  -q

conda run -n DPS python -m pytest tests/bcns -q
```

Visualize target construction without a checkpoint:

```bash
conda run -n DPS python scripts/bcns/visualize_step2_poisson_targets.py \
  --output-dir results/bcns_step2/poisson_target_visualization
```

Run FFHQ center-box smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step2_compare.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_harmonic_projected_ffhq.yaml \
  --save_dir results/bcns_step2/ffhq_center_box \
  --gpu 0 \
  --num_images 2 \
  --seed 123 \
  --record_every 50 \
  --mask_mode center_box \
  --box_size 96
```

Run FFHQ thin-scratch smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step2_compare.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_harmonic_projected_ffhq.yaml \
  --save_dir results/bcns_step2/ffhq_thin_scratch \
  --gpu 0 \
  --num_images 2 \
  --seed 123 \
  --record_every 50 \
  --mask_mode thin_scratch \
  --scratch_thickness 5
```

## Expected Interpretation

Harmonic rows may be smoother in the hole. Poisson rows may retain more local structure when the source term is useful. Projected variants should keep the raw known region consistent with the timestep-noisy measurement, and clean composites should have near-zero known-region MSE for clean inpainting. The metrics CSV records Poisson iterations, residual, runtime, RHS mode, and method so these visual differences can be tied back to solver behavior.
