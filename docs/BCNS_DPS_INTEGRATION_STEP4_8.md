# BCNS DPS Integration Step 4.8

## MCG plus BCNS guidance

Step 4.8 combines the stronger DPS measurement guidance baseline with BCNS structural targets. Earlier standalone BCNS runs were usually too weak, and projected BCNS variants often collapsed toward `projection_fixed`. The new candidate keeps the measurement-gradient pull from MCG and adds a separate BCNS target-gradient term.

The conditioning update is:

```text
x_{t-1} = Pi_t[
    x'_{t-1}
    - gamma_m grad L_meas
    - gamma_b gamma_t grad L_BCNS
]
```

where:

```text
L_meas = || y - A(x0_hat) ||
L_BCNS = 0.5 / tau_t^2 * mean_hole((mu_t - stopgrad(target_t))^2)
```

For inpainting, `Pi_t` is the optional noisy known-region projection:

```text
Pi_t(x) = M y_t + H x
```

This is a projected operator-splitting sampler. It is a practical debug sampler for comparing structural guidance terms, not an exact DPS posterior sampler.

## Registered method

The new conditioning method is:

```text
mcg_bcns
```

It runs the MCG measurement gradient every conditioning call. The BCNS target is controlled by:

```text
bcns_gamma_max
bcns_pde_start_frac
bcns_ramp_power
bcns_apply_every_n_steps
bcns_reuse_last_target
bcns_target_builder
bcns_target_builder_params
```

The main target builders for Step 4.8 are:

```text
luminance_lift_harmonic
luminance_lift_poisson
luminance_lift_flow
```

## Trace diagnostics

Each `mcg_bcns` conditioning call appends one trace row with measurement and BCNS losses, component gradient/update norms, BCNS-to-MCG ratios, target recompute/reuse/skip flags, and target displacement splits. The actual sampler update uses one combined VJP through `x_prev` so DPS gradient checkpointing is traversed only once; the component norms are logged in `x_0_hat` space as guidance-strength diagnostics.

Important trace columns:

```text
meas_grad_norm
bcns_grad_norm
meas_update_norm
bcns_update_norm
bcns_to_meas_update_ratio
bcns_gamma
bcns_target_recomputed
bcns_target_reused
bcns_target_skipped
target_disp_hole
```

The summary diagnostics include:

```text
mean_meas_grad_norm
mean_bcns_grad_norm
max_bcns_grad_norm
sum_meas_update_norm
sum_bcns_update_norm
max_bcns_to_meas_update_ratio
mean_bcns_to_meas_update_ratio
num_bcns_nonzero_steps
num_bcns_target_recomputed
num_bcns_target_reused
```

## Ablation sets

`mcg_bcns_main` compares:

```text
ps
projection_fixed
mcg_fixed
mcg_bcns_lift_harmonic
mcg_bcns_lift_poisson
mcg_bcns_lift_flow_strong
```

`mcg_bcns_pde_strength` sweeps harmonic and Poisson lift scale and flow pseudo-time / viscosity / lift strength.

`mcg_bcns_projection_effect` compares projected and unprojected MCG+BCNS variants while evaluation still saves raw and clean-composite outputs.

## Metrics to inspect

Quality and structure:

```text
composite_hole_mse
composite_seam_mse
composite_gradient_mismatch
composite_isophote_error
composite_laplacian_mismatch
```

Guidance strength:

```text
meas_grad_norm
bcns_grad_norm
meas_update_norm
bcns_update_norm
bcns_to_meas_update_ratio
lift_scale
bcns_target_builder
flow_pseudo_time
flow_dt
flow_nu
```

Useful success signs:

1. `mcg_bcns_*` outputs are not pixel-identical to `mcg_fixed`.
2. `bcns_to_meas_update_ratio` is not near zero.
3. Some BCNS variants improve structural metrics over `mcg_fixed`.
4. Hole MSE does not catastrophically degrade.
5. Very strong flow settings may produce artifacts; this is expected in the strength sweep.

## Manual GPU commands

Run these manually for real experiments. Each process sees one GPU through `CUDA_VISIBLE_DEVICES`, so `--gpu 0` is used inside the script.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/main_thick24_100_gpu0 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_main

CUDA_VISIBLE_DEVICES=1 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/main_freeform_100_gpu1 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode freeform_medium \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_main

CUDA_VISIBLE_DEVICES=2 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/main_text_100_gpu2 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode text_mask \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_main

CUDA_VISIBLE_DEVICES=3 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/main_thick24_50_gpu3 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 25 \
  --sampling_steps 50 \
  --ablation_set mcg_bcns_main

CUDA_VISIBLE_DEVICES=4 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/pde_strength_thick24_gpu4 \
  --gpu 0 \
  --num_images 4 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_pde_strength

CUDA_VISIBLE_DEVICES=5 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/pde_strength_freeform_gpu5 \
  --gpu 0 \
  --num_images 4 \
  --seed 123 \
  --mask_mode freeform_medium \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_pde_strength

CUDA_VISIBLE_DEVICES=6 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/projection_effect_gpu6 \
  --gpu 0 \
  --num_images 4 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_projection_effect

CUDA_VISIBLE_DEVICES=7 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_8/main_thick24_250_gpu7 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 250 \
  --ablation_set mcg_bcns_main
```
