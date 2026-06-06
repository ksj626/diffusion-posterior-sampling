# BCNS DPS Integration Step 4.5: Guidance Activation And Few-Step Ablations

Step 4 showed that projected BCNS variants could become visually identical to
`projection_fixed`. The scalar PDE target moved, but most RGB target
displacement was in known pixels. Because the outer target loss is hole-only,
the effective RGB guidance gradient could be zero.

Step 4.5 adds an optional detached scalar structure target loss:

```text
0.5 * structure_loss_weight * mean_hole((S_sigma(mu_t) - stopgrad(I_target))^2)
```

This is not direct Navier-Stokes residual backpropagation. The scalar target is
still produced by the existing normalized, harmonic, Poisson, or finite-flow
builders and is detached before the outer loss.

## Guidance Modes

- RGB target guidance matches `mu_t` to a detached RGB proximal target in the
  hole. This is the Step 4 behavior when `rgb_loss_weight=1` and
  `structure_loss_weight=0`.
- Structure target guidance matches `S_sigma(mu_t)` to a detached scalar target
  in the hole. This can activate gradients even when RGB target displacement is
  only in known pixels.
- Combined guidance uses both terms and logs `rgb_loss`, `structure_loss`,
  `total_loss`, `grad_norm`, and known/hole/full target displacement metrics.

Projected variants still apply noisy known-region projection during sampling.
Non-projected variants remain useful for separating hard data consistency from
BCNS target effects.

## Few-Step Motivation

The 1000-step DPS chain can be strong enough that small BCNS target updates are
hard to distinguish. Step 4.5 supports controlled 25/50/100/250/1000 reverse
update comparisons through the existing `timestep_respacing` sampler path.
Conditioning receives the reduced-chain timestep index and the actual number of
reverse updates, so the late-ramp schedule remains meaningful under few-step
runs.

## Commands

Guidance activation visualization:

```bash
conda run -n DPS python scripts/bcns/visualize_guidance_activation.py \
  --output-dir results/bcns_step4_5/guidance_activation_viz \
  --mask_mode thin_scratch
```

Guidance activation smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_5/guidance_activation \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 100 --ablation_set guidance_activation \
  --sampling_steps 100
```

Few-step smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_5/fewstep \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 50 --ablation_set fewstep
```

Proximal strength smoke:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step4_5/proximal_strength \
  --gpu 0 --num_images 2 --seed 123 \
  --mask_mode center_box --box_size 96 \
  --record_every 100 --ablation_set proximal_strength \
  --sampling_steps 100
```

## Outputs

Step 4.5 metrics include `sampling_steps`, `actual_num_reverse_updates`,
`target_disp_full`, `target_disp_known`, `target_disp_hole`, `structure_loss`,
`structure_disp_hole`, and `grad_norm`. Few-step artifacts are written under
`steps_0025/`, `steps_0050/`, `steps_0100/`, `steps_0250/`, and `steps_1000/`.

Useful checks:

- Structure-guided rows should have positive `structure_loss` and `grad_norm`
  whenever the scalar target differs in the hole.
- `target_disp_known` can be positive while `target_disp_hole` is zero; that is
  the Step 4 collapse mode.
- 25/50/100-step runs may expose BCNS seam or structure benefits more clearly
  than the full 1000-step chain.

## Limitations

Step 4.5 does not add new PDE solvers, RGB Navier-Stokes, carry-over PDE memory,
training, or a full benchmark harness. It is still detached scalar target
matching inside the existing DPS conditioning path.
