# BCNS DPS Integration Step 4.10

## Purpose

Step 4.10 expands the MCG-BCNS experiment surface after Step 4.9 showed small but consistent structural gains from ratio-controlled harmonic guidance. This step does not add a PDE solver, RGB Navier-Stokes, or benchmark automation. It only adds stronger ablation configs and ultra-few-step support.

The new sweeps ask:

1. Does stronger BCNS update ratio help or artifact?
2. Is `bcns_ratio_clip_max` limiting the actual update ratio?
3. Does more frequent BCNS target injection matter?
4. Can longer scalar flow evolution become useful when ratio/lift are stronger?
5. What happens in ultra-few-step sampling at 5, 10, 25, 50, and 100 steps?

## Ultra-Few-Step Sampling

The runner now accepts:

```text
5, 10, 25, 50, 100, 250, 1000
```

The sampler path is unchanged. The diffusion config still receives:

```python
config["timestep_respacing"] = str(sampling_steps)
```

This makes 5/10-step smoke-style comparisons possible without adding a new sampler.

## Strong Ratio Sweep

Use:

```bash
--ablation_set mcg_bcns_strong_ratio_sweep
```

This compares `mcg_fixed_s1.0` against harmonic and Poisson luminance-lift MCG-BCNS with:

```text
bcns_target_update_ratio = 0.1, 0.3, 1.0, 3.0
bcns_ratio_clip_max = 1000
bcns_apply_every_n_steps = 2
mcg_scale = 1.0
```

`r0.3` and `r1.0` may improve structural metrics more visibly than earlier sweeps, but can create artifacts. `r3.0` is intentionally aggressive.

## Clip Sweep

Use:

```bash
--ablation_set mcg_bcns_clip_sweep
```

This keeps harmonic guidance fixed at ratio `0.3` or `1.0` while sweeping:

```text
bcns_ratio_clip_max = 100, 300, 1000, 3000, 10000
```

If `bcns_to_meas_update_ratio` remains below the requested ratio when clip is low, the clip is probably limiting BCNS. If metrics worsen sharply as clip increases, the target is strong enough but unstable.

## Apply-Every Sweep

Use:

```bash
--ablation_set mcg_bcns_apply_every_sweep
```

This sweeps:

```text
bcns_apply_every_n_steps = 10, 5, 2, 1
```

for harmonic ratios `0.1` and `0.3`, and Poisson ratio `0.3`.

`apply_every=1` injects BCNS every reverse step. It can increase structural influence, but is slower and may become unstable.

## Flow Evolution Sweep

Use:

```bash
--ablation_set mcg_bcns_flow_evolution_sweep
```

This reintroduces scalar NS-flow targets with stronger lift/ratio and longer pseudo-time:

```text
pseudo_time = 0.003, 0.01, 0.03, 0.1, 0.3
dt = 0.001
nu = 0.1 or 0.2
lift_scale = 4, 8, 16
bcns_target_update_ratio = 0.1, 0.3, 1.0
```

`pseudo_time=0.3` is extreme. It may fail, converge poorly, or produce artifacts. That is acceptable for a strength sweep.

## Ultra-Few-Step Set

Use:

```bash
--ablation_set mcg_bcns_ultra_fewstep
```

Compact method list:

```text
ps
projection_fixed
mcg_fixed_s1.0
harmonic_r0.3_clip1000_every2
poisson_r0.3_clip1000_every2
flow_strong_r0.3_clip1000_every2
```

This set is intended for `--sampling_steps 5`, `10`, `25`, `50`, and `100`.

## Large Hole Variants

The existing structured masks remain available:

```text
thin_scratch
thick_scratch_12
thick_scratch_24
text_mask
freeform_medium
center_box_96
center_box_128
```

The larger center boxes are included because stronger BCNS guidance may be most visible when holes are hard enough for simple local projection to struggle. Center boxes are also more semantic-prior dependent, so interpret MSE and structural metrics together.

## Manual Commands

Do not run these during implementation. Each process sees one GPU via `CUDA_VISIBLE_DEVICES`, so use `--gpu 0` inside the script.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/strong_ratio_thick24_gpu0 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_strong_ratio_sweep

CUDA_VISIBLE_DEVICES=1 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/strong_ratio_center96_gpu1 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode center_box_96 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_strong_ratio_sweep

CUDA_VISIBLE_DEVICES=2 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/clip_text_gpu2 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode text_mask \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_clip_sweep

CUDA_VISIBLE_DEVICES=3 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/apply_every_freeform_gpu3 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode freeform_medium \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_apply_every_sweep

CUDA_VISIBLE_DEVICES=4 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/flow_center96_gpu4 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode center_box_96 \
  --record_every 50 \
  --sampling_steps 100 \
  --ablation_set mcg_bcns_flow_evolution_sweep

CUDA_VISIBLE_DEVICES=5 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/ultra5_center128_gpu5 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode center_box_128 \
  --record_every 1 \
  --sampling_steps 5 \
  --ablation_set mcg_bcns_ultra_fewstep

CUDA_VISIBLE_DEVICES=6 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/ultra10_thick24_gpu6 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 2 \
  --sampling_steps 10 \
  --ablation_set mcg_bcns_ultra_fewstep

CUDA_VISIBLE_DEVICES=7 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_10/ultra25_text_gpu7 \
  --gpu 0 \
  --num_images 8 \
  --seed 123 \
  --mask_mode text_mask \
  --record_every 5 \
  --sampling_steps 25 \
  --ablation_set mcg_bcns_ultra_fewstep
```

## Interpretation

Useful fields:

```text
bcns_target_update_ratio
bcns_ratio_clip_max
bcns_ratio_scale_mean
bcns_to_meas_update_ratio
bcns_apply_every_n_steps
flow_pseudo_time
lift_scale
composite_hole_mse
composite_gradient_mismatch
composite_isophote_error
composite_laplacian_mismatch
```

Potential outcomes:

1. `r0.3` improves structure without hurting MSE: promising range.
2. `r1.0` improves structure but artifacts appear: maybe useful with stronger regularization later.
3. Higher clip increases actual ratio but worsens output: clipping was protecting stability.
4. `apply_every=1` helps metrics: BCNS signal was too sparse.
5. Flow pseudo-time `0.3` fails or artifacts: expected for the extreme setting.
