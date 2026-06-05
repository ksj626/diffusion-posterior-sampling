# BCNS DPS Integration Step 1.5

## Purpose

Step 1 outputs could look like unconditional samples because the raw DPS sample was
reported directly and the default `target_mode: projected_mu` is nearly a no-op
for a hole-only discrepancy loss. Step 1.5 makes the inpainting smoke pipeline
interpretable before any Poisson, Navier-Stokes, or BCNS flow work is added.

This step does not claim BCNS improves inpainting. It checks known-region
consistency, nonzero structure-target guidance, fair comparison setup, and
smoke-level numerical stability.

## Conditioning Methods

- `ps`: DPS posterior sampling applies a measurement-discrepancy gradient. It
  does not hard-project known pixels by itself.
- `projection`: existing DPS projection conditioning. It is included as a
  baseline comparison method.
- `mcg`: DPS manifold constraint gradient plus projection.
- `bcns_target`: target-guidance conditioning using a detached target in the
  hole region. Step 1.5 adds optional noisy known-region projection for this
  method only.

The original DPS raw reconstruction is not automatically clean-composited for
inpainting. Step 1.5 reports both raw and composited outputs.

## Known-Region Handling

Final clean-space compositing is

```text
x_comp = M y + (1 - M) x_raw
```

where `M` is the DPS known mask and `y` is the clean inpainting measurement. For
clean inpainting, `composite_known_mse` should be near zero, while
`raw_known_mse` may be nonzero depending on the method. The composite hole MSE
should match the raw hole MSE.

Optional per-step noisy projection is

```text
x_t <- M y_t + (1 - M) x_t
```

where `y_t` is the diffused noisy measurement at the current sampler step. Set
`apply_noisy_known_projection: true` on `bcns_target` to use it. The projected
BCNS structure-prox smoke variant should preserve known pixels in the raw sample
better than the non-projected variant.

## Structure Target

`normalized_known_smooth` computes

```text
I_tilde = G_sigma * (M * Lum(y)) / (G_sigma * M + eps)
```

This avoids the zero-valued hole in the masked measurement leaking into the
smoothed structure target. The returned structure is DPS-scale luminance.

`target_mode: projected_mu` remains available as a sanity/debug target, but it
usually produces zero hole displacement because the proximal target starts from
`hard_project_clean(mu, measurement, mask_known)`. Smoke configs now default to
`normalized_known_smooth`.

## Commands

```bash
conda run -n DPS python -m pytest tests/bcns -q

conda run -n DPS python scripts/bcns/visualize_step1_targets.py \
  --output-dir results/bcns_step1_5/target_visualization

conda run -n DPS python scripts/bcns/run_bcns_step1_smoke.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_structure_prox_ffhq.yaml \
  --save_dir results/bcns_step1_5/ffhq_structure_prox_norm \
  --gpu 0 --num_images 2 --seed 123 --record_every 50

conda run -n DPS python scripts/bcns/run_bcns_step1_5_compare.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_structure_prox_ffhq.yaml \
  --save_dir results/bcns_step1_5/ffhq_compare \
  --gpu 0 --num_images 2 --seed 123 --record_every 50
```

The comparison runner writes `metrics.csv` plus per-image comparison contact
sheets. It runs `ps`, `projection`, `mcg`, `bcns_identity`,
`bcns_simple_shift`, `bcns_structure_prox_norm`, and
`bcns_structure_prox_norm_projected` on the same images, masks, and starting
noise.

## ImageNet Note

This repo currently registers only the `ffhq` dataset in `data/dataloader.py`.
The ImageNet model checkpoint/config can exist locally, but a true ImageNet
smoke config needs an actual ImageNet-compatible dataset class and root. Step
1.5 does not add dataset support.

## Limitations

- No Poisson integration.
- No Navier-Stokes flow.
- No FTCS, IMEX, CN, SOR, CG, or local-MAP diffusion solver.
- No claim of BCNS quality improvement.
