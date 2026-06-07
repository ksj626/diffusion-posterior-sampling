# BCNS DPS Integration Step 4.9

## Why Step 4.9 exists

Step 4.8 made MCG plus BCNS runnable, but its early comparisons had a scale confound:

```text
mcg_fixed may have used scale = 1.0
mcg_bcns used mcg_scale = 0.3
```

Step 4.9 adds scale-matched comparisons, ratio-controlled BCNS strength, and a compact hole-geometry sweep so the next experiments can answer whether BCNS helps when measurement guidance strength is fair.

No new PDE solver or RGB Navier-Stokes model is added in this step.

## Scale-matched methods

The runner now has explicit MCG baselines:

```text
mcg_fixed_s0.3
mcg_fixed_s1.0
```

These call the existing `mcg_fixed` conditioning method with explicit `scale` values. The `mcg_bcns_scale_match` ablation compares:

```text
ps
projection_fixed
mcg_fixed_s0.3
mcg_bcns_lift_harmonic_m0.3
mcg_bcns_lift_poisson_m0.3
mcg_fixed_s1.0
mcg_bcns_lift_harmonic_m1.0
mcg_bcns_lift_poisson_m1.0
```

Every MCG and MCG-BCNS row records `mcg_scale`.

## Ratio-controlled BCNS guidance

`mcg_bcns` now supports optional ratio control:

```text
bcns_ratio_control
bcns_target_update_ratio
bcns_ratio_eps
bcns_ratio_clip_min
bcns_ratio_clip_max
```

Without ratio control, the update remains:

```text
x_t <- x_t - mcg_scale * grad L_meas - bcns_scale * gamma * grad L_BCNS
```

With ratio control, the BCNS term is rescaled so its diagnostic update norm targets a chosen fraction of the measurement update:

```text
target_norm = bcns_target_update_ratio * ||meas_update||
ratio_scale = target_norm / (||bcns_update_raw|| + eps)
ratio_scale = clamp(ratio_scale, clip_min, clip_max)
bcns_update = ratio_scale * bcns_update_raw
```

Because DPS uses checkpointed UNet blocks, `mcg_bcns` still performs one combined VJP through `x_prev`. The ratio scale is applied to the BCNS loss coefficient before that single backward pass.

Important trace and metrics fields:

```text
bcns_ratio_control
bcns_target_update_ratio
bcns_ratio_scale
bcns_ratio_scale_mean
bcns_ratio_scale_max
meas_update_norm
bcns_update_raw_norm
bcns_update_norm
bcns_to_meas_update_ratio
```

## Hole variant sweep

The compact hole sweep uses:

```text
ps
projection_fixed
mcg_fixed_s1.0
mcg_bcns_harmonic_best
mcg_bcns_poisson_best
```

The initial “best” BCNS settings are:

```text
mcg_scale = 1.0
bcns_ratio_control = true
bcns_target_update_ratio = 0.03
apply_noisy_known_projection = true
```

Use the wrapper:

```text
scripts/bcns/run_hole_variant_sweep.py
```

It runs the main Step 4 ablation runner once per mask mode, saves each result under:

```text
<save_dir>/<mask_mode>/
```

and writes:

```text
metrics_hole_variants.csv
summary_hole_variants.csv
summary_hole_variants.md
```

Default variants:

```text
thin_scratch
thick_scratch_12
thick_scratch_24
text_mask
freeform_medium
center_box_96
center_box_128
```

## Fairness logging

The runner now records the shared initial noise for each image:

```text
initial_noise_seed
initial_noise_mean
initial_noise_std
```

All methods for one image clone the same `x_start_base`, and the sampler RNG is reset before each method run.

## Interpretation

Scale-matched comparison:

```text
mcg_bcns_m1.0 improves structural metrics over mcg_fixed_s1.0
```

means BCNS is plausibly helpful.

If:

```text
mcg_bcns_m1.0 ≈ mcg_fixed_s1.0
```

then BCNS is likely too weak or redundant for that mask/data setting.

Ratio sweep:

```text
0.01 or 0.03
```

is the expected stable range. If only `0.1` changes the output, artifacts are more likely. If no ratio changes the output, the current BCNS target source may not be informative enough.

Hole variant sweep:

```text
thin_scratch: expected small differences
thick_scratch/text/freeform: most informative
center_box: less PDE-friendly and more semantic-prior dependent
```

## Manual GPU commands

Each process sees one GPU through `CUDA_VISIBLE_DEVICES`, so use `--gpu 0` inside the script.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/scale_match_thick24_gpu0 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_scale_match

CUDA_VISIBLE_DEVICES=1 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/scale_match_text_gpu1 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode text_mask \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_scale_match

CUDA_VISIBLE_DEVICES=2 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/scale_match_freeform_gpu2 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode freeform_medium \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_scale_match

CUDA_VISIBLE_DEVICES=3 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/ratio_thick24_gpu3 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_ratio_sweep

CUDA_VISIBLE_DEVICES=4 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/ratio_freeform_gpu4 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode freeform_medium \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_ratio_sweep

CUDA_VISIBLE_DEVICES=5 python scripts/bcns/run_hole_variant_sweep.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/hole_variant_100_gpu5 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set hole_variant_sweep \
  --hole_variants thin_scratch,thick_scratch_12,thick_scratch_24,text_mask,freeform_medium,center_box_96,center_box_128

CUDA_VISIBLE_DEVICES=6 python scripts/bcns/run_hole_variant_sweep.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/hole_variant_50_gpu6 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --record_every 25 \
  --sampling_steps 50 \
  --ablation_set hole_variant_sweep \
  --hole_variants thin_scratch,thick_scratch_12,thick_scratch_24,text_mask,freeform_medium,center_box_96,center_box_128

CUDA_VISIBLE_DEVICES=7 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_9/ratio_text_50_gpu7 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode text_mask \
  --record_every 25 \
  --sampling_steps 50 \
  --ablation_set mcg_bcns_ratio_sweep
```

## Limitations

Step 4.9 does not add RGB PDE evolution, new PDE solvers, training, or final benchmark automation. It is a fairness and diagnostics layer for deciding whether MCG plus BCNS deserves larger experiments.
