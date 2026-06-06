# BCNS DPS Integration Step 4.6: Guidance Trace And Strength Debug

Step 4.5 activated direct scalar-structure loss, but the saved projected BCNS
samples were still nearly identical to `projection_fixed`. That can happen even
when `structure_loss > 0`: late scheduling, infrequent target recomputation,
small `gamma_max`, small scalar-loss weights, and known-pixel projection can make
the accumulated sample update too small to show up in final PNGs.

Step 4.6 is a debugging step. It does not add new PDE solvers, RGB
Navier-Stokes, carry-over PDE memory, or a final benchmark. It adds trace and
strength controls so the next experiment can answer whether BCNS guidance is
actually moving the reverse trajectory.

## Per-Step Guidance Trace

`BCNSTargetGuidance` now records one compact row per conditioning call. Runners
save the trace as:

```text
<image_id>/<method>/guidance_trace.csv
```

Important fields:

- `num_guidance_calls`: number of conditioning calls in the reverse chain.
- `num_target_recomputed`: steps that rebuilt the detached target.
- `num_target_reused`: skipped steps that reused a previous target.
- `num_nonzero_grad_steps`: steps with `grad_norm > 0`.
- `sum_update_norm`: accumulated `||scale * gamma * grad||_2` over the chain.
- `mean_update_norm`, `max_update_norm`: average and largest per-step update.
- `mean_structure_loss`, `max_structure_loss`: scalar loss strength over time.
- `mean_target_disp_hole`, `max_target_disp_hole`: RGB target movement in holes.
- `mean_structure_disp_hole`, `max_structure_disp_hole`: scalar target movement.

The method-level `metrics.csv` also includes these summary fields, so comparison
tables no longer depend on the last timestep only.

## Guidance Strength Sweeps

The new `guidance_strength` ablation uses flow scalar targets with structure-only
guidance and compares projected and non-projected variants. The default is a
small paired sweep:

```text
(gamma_max, structure_loss_weight) =
(0.02, 0.1), (0.1, 1), (0.5, 10), (1.0, 50)
```

Add `--full_strength_grid` to expand over all gamma/weight combinations and
`apply_every_n_steps` values `{5, 10}`. The expected success signal is not image
quality yet; it is a larger nonzero `sum_update_norm` and nonzero hole structure
displacement.

The `flow_strength_extended` ablation fixes `gamma_max=0.5`,
`structure_loss_weight=10`, `apply_every_n_steps=5`, and `sampling_steps=100`,
then varies pseudo-time, `dt`, and `nu`. Diagnostics include `flow_pseudo_time`,
`flow_dt`, `flow_nu`, `flow_num_steps`, and `flow_runtime_sec`.

## Luminance-Lift Target

The RGB proximal target can still yield nearly zero hole RGB displacement. The
debug luminance-lift bridge builds an existing scalar target first, then applies:

```text
delta_I = target_structure - S_sigma(mu)
delta_RGB = lift(delta_I) inside the hole
target = hard_project_clean(mu + lift_scale * delta_RGB, measurement, mask_known)
```

Registered target builders:

```text
luminance_lift_flow
luminance_lift_poisson
luminance_lift_harmonic
```

Modes:

- `equal_rgb`: repeats the scalar delta into all RGB channels.
- `luma_weights`: distributes through normalized luminance weights so the RGB
  luminance change approximately matches the scalar delta.

This is a practical debug bridge, not the final theoretical method. It checks
whether a scalar PDE target can influence RGB samples when deliberately lifted
into the hole.

## Manual GPU Commands

Because these are real diffusion runs, run them manually on the available GPUs.
Each process sees one GPU through `CUDA_VISIBLE_DEVICES`, so pass `--gpu 0`
inside the script.

### Guidance strength, 100-step thin scratch

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/guidance_strength_gpu0 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set guidance_strength
```

### Guidance strength, 50-step thin scratch

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/guidance_strength_50_gpu1 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 25 --sampling_steps 50 \
  --ablation_set guidance_strength
```

### Flow strength extended, 100-step

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/flow_strength_gpu2 \
  --gpu 0 --num_images 2 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set flow_strength_extended
```

### Luminance lift, 100-step

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/luminance_lift_gpu3 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set luminance_lift
```

### Few-step comparison, parallel runs

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/fewstep_25_gpu4 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 25 --sampling_steps 25 \
  --ablation_set guidance_activation
```

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/fewstep_50_gpu5 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 25 --sampling_steps 50 \
  --ablation_set guidance_activation
```

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/fewstep_100_gpu6 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set guidance_activation
```

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_6/fewstep_250_gpu7 \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 50 --sampling_steps 250 \
  --ablation_set guidance_activation
```

## Expected Interpretation

This step succeeds when BCNS rows show nonzero `sum_update_norm`,
`num_nonzero_grad_steps`, `target_disp_hole`, or `structure_disp_hole`, and the
stronger/lifted variants stop being pixel-identical to `projection_fixed`. It is
not expected to settle final image quality.

Limitations remain: no RGB PDE, no carry-over PDE memory, and no final benchmark.
