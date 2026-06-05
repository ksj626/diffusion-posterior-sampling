# BCNS DPS Integration Step 1

Step 1 adds a structure-proximal target builder to the existing `bcns_target`
conditioning method. It is a debugging/proximal intermediate, not full
BCNS-Flow-DPS.

## Objective

The target builder approximates

```text
u_tilde ~= argmin_{M u = y}
    1 / (2 tau_t^2) * mean_hole ||u - mu_t||^2
    + lambda_s / 2 * mean_hole ||S_sigma(u) - I_tilde||^2
```

where `mu_t = pred_xstart`, `M` is the DPS known mask, `H = 1 - M` is the
hole, and `S_sigma` is smoothed DPS-scale luminance. The guidance loss remains
the Step 0 detached-target discrepancy:

```text
L_t = 1 / (2 tau_t^2) * mean_hole ||mu_t - sg[u_tilde]||^2
```

## What This Is Not

This step does not implement Poisson targets, Navier-Stokes flow, CN/SOR/CG
inside diffusion, local-MAP residual solves, or benchmark automation.

## Schedule Integration

`GaussianDiffusion.p_sample_loop` now passes `t_index`, `num_steps`, and `time`
into conditioning methods. `BCNSTargetGuidance` uses these values to apply the
late-ramp `BCNSSchedule`.

## Config Parameters

`structure_prox` accepts `tau2`, `lambda_structure`, `structure_sigma`,
`prox_steps`, `prox_step_size`, and `target_mode`. Supported target modes are
`projected_mu`, `measurement_only_smooth`, and `mu`.

The ImageNet smoke config uses the ImageNet model config but the existing `ffhq`
dataset loader because this repository currently defines only an `ffhq` dataset
class.

## Commands

```bash
conda run -n DPS python -m pytest \
  tests/bcns/test_dps_adapter.py \
  tests/bcns/test_target_guidance.py \
  tests/bcns/test_structure_proximal.py \
  tests/bcns/test_timestep_schedule_integration.py \
  -q

conda run -n DPS python scripts/bcns/visualize_step1_targets.py \
  --output-dir results/bcns_step1/target_visualization

conda run -n DPS python scripts/bcns/run_bcns_step1_smoke.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_structure_prox_ffhq.yaml \
  --save_dir results/bcns_step1/ffhq_structure_prox \
  --gpu 0 --num_images 2 --seed 123 --record_every 50

conda run -n DPS python scripts/bcns/run_bcns_step1_smoke.py \
  --model_config configs/imagenet_model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_structure_prox_imagenet.yaml \
  --save_dir results/bcns_step1/imagenet_structure_prox \
  --gpu 0 --num_images 2 --seed 123 --record_every 50
```

Smoke outputs include `input/`, `mask/`, `label/`, `recon/`, `progress/`,
`diagnostics.csv`, per-image contact sheets, and an aggregate `contact_sheet.png`.
