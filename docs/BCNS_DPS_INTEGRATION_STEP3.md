# BCNS DPS Integration Step 3

Step 3 adds finite-step BCNS vorticity-stream flow targets to the existing DPS target-guidance path. It uses the standalone BCNS flow engine to evolve a scalar structure target for a few pseudo-time steps, then feeds that terminal scalar structure into the existing RGB structure-proximal target builder.

This is finite-step target guidance. It is not exact posterior sampling, not exact local MAP, and not an unrolled differentiable PDE gradient. The PDE target is detached before the outer DPS hole-only discrepancy loss is evaluated.

## Flow Target

The scalar initial intensity is built from either the current clean estimate or its known-region projection:

```text
I_t^0 = S_sigma(M y + H mu_t)
```

or

```text
I_t^0 = S_sigma(mu_t)
```

The initial vorticity is:

```text
w_t^0 = Delta_h I_t^0
```

The finite BCNS pseudo-time flow evolves:

```text
partial_tau w = -J(I, w) + nu div(g grad w)
```

At each flow step, intensity is reconstructed by masked Poisson solve:

```text
Delta_h I = w
```

with known-side boundary intensity estimated from the measurement.

The terminal scalar structure is then used by the RGB proximal target:

```text
min_{M u = y} 1/(2 tau_t^2) |H(u - mu_t)|^2
            + lambda_s/2 |H(S_sigma u - I_tilde)|^2
```

The outer DPS guidance remains the detached-target hole discrepancy.

## Integrators

`imex_be` is the default for smoke runs because it treats diffusion implicitly and is more forgiving than fully explicit FTCS.

`imex_cn` is included as a controlled comparison with Crank-Nicolson diffusion. It should run without NaN on the smoke settings.

`ftcs` is available only as a controlled ablation. It uses a conservative CFL check and can intentionally raise when `dt` is too large. The default FTCS config uses smaller `dt` and shorter pseudo-time.

## Poisson Solvers

SOR is the default in smoke configs because it is already available on tensors and has predictable runtime. CG remains available through the shared Poisson solver config for ablation. Dense reference solves are used only in small CPU tests.

## Frequency Control

Flow targets are expensive, so `bcns_target` now supports:

- `apply_every_n_steps`: recompute targets only when `t_index % apply_every_n_steps == 0`.
- `reuse_last_target`: optionally reuse the last detached target on skipped timesteps.

Step 3 defaults to `apply_every_n_steps: 10` and `reuse_last_target: false`, so skipped timesteps apply only optional known-region projection and return zero guidance loss.

## Commands

Run targeted tests:

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
  tests/bcns/test_flow_targets.py \
  tests/bcns/test_flow_target_builders.py \
  tests/bcns/test_bcns_guidance_frequency.py \
  -q

conda run -n DPS python -m pytest tests/bcns -q
```

Visualize target construction without a checkpoint:

```bash
conda run -n DPS python scripts/bcns/visualize_step3_flow_targets.py \
  --output-dir results/bcns_step3/flow_target_visualization
```

Run FFHQ center-box smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step3_compare.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step3/ffhq_center_box \
  --gpu 0 --num_images 2 --seed 123 --record_every 50 \
  --mask_mode center_box --box_size 96
```

Run FFHQ thin-scratch smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step3_compare.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step3/ffhq_thin_scratch \
  --gpu 0 --num_images 2 --seed 123 --record_every 50 \
  --mask_mode thin_scratch --scratch_thickness 5
```

Run the one-image CN config smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step3_compare.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_cn_projected_ffhq.yaml \
  --save_dir results/bcns_step3/ffhq_center_box_cn \
  --gpu 0 --num_images 1 --seed 123 --record_every 50 \
  --mask_mode center_box --box_size 96
```

## Limitations

Step 3 has no carry-over PDE memory across diffusion steps, no full benchmark table, no learned training, no exact local MAP solve, and no unrolled PDE gradient through the flow. It is a finite-step operator-splitting target for controlled smoke comparison.
