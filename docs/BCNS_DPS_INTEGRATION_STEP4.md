# BCNS DPS Integration Step 4: Evaluation and Ablation Harness

Step 4 adds reusable evaluation metrics and ablation runners for the BCNS-DPS
inpainting path. It does not add new PDE solvers, training code, exact MAP
updates, or carry-over PDE memory. The goal is to measure whether the existing
target builders improve hole and boundary structure, not just full-image MSE.

## Metrics

Pixel metrics are computed for both raw sampler output and final clean
composite output. DPS tensors are assumed to live in `[-1, 1]`, so PSNR uses
`data_range=2.0`.

- `PSNR`: `20 log10(data_range) - 10 log10(MSE)`, with exact matches capped at
  `100.0` dB for stable CSV summaries.
- `SSIM`: a lightweight torch-only windowed SSIM implementation; no extra
  dependency is required.
- `known/hole/full MSE` and `known/hole/full MAE`: region splits using
  `mask_known == 1`, where the hole is `1 - mask_known`.
- `LPIPS`: optional. If the `lpips` package is missing or cannot initialize,
  the metric is left blank instead of failing the run.

Structural metrics are intended to expose boundary-crossing behavior:

- `seam MSE`: pixel MSE in a narrow band around the hole boundary.
- `gradient mismatch`: MSE between gradients of smoothed luminance structures.
- `isophote error`: `1 - cos(theta)` between normalized perpendicular
  gradients in the boundary band.
- `Laplacian mismatch`: MSE between scalar structure Laplacians in the boundary
  band.
- `NS residual`: RMS norm of the existing BCNS vorticity-stream RHS inside the
  hole. This is diagnostic only; lower residual is not automatically better
  perceptual quality.

## Raw vs Composite

`recon_raw` is the sampler output. `recon_composite` replaces known pixels with
the clean measurement after sampling. For clean inpainting, composite known MSE
should be near zero. Hole metrics are the fairest way to compare actual
reconstruction quality, while raw known MSE exposes whether a sampler drifted in
observed regions.

## Projected vs Non-Projected

Projected BCNS variants use `apply_noisy_known_projection=True` during sampling.
Non-projected variants use the same structural target guidance but leave known
pixels unprojected at each conditioning call. The Step 4 `projection_effect`
ablation separates structural target effects from hard known-region projection.

## Ablation Runner

The main runner is:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step4/ffhq_ablation \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 100 --ablation_set smoke
```

Supported `--ablation_set` values:

- `smoke`: `ps`, `projection_fixed`, `mcg_fixed`, structure-prox, harmonic,
  Poisson, and BE flow.
- `flow_integrator`: BE, CN, and conservative FTCS flow targets.
- `flow_strength`: BE flow sweep over pseudo-time and viscosity.
- `frequency`: BE flow sweep over `apply_every_n_steps` and target reuse.
- `poisson_solver`: Poisson target sweep over `sor_rb`/`cg` and iteration caps.
- `projection_effect`: projected and non-projected Poisson/flow variants.

Each run writes `metrics.csv`, `summary_by_method.csv`,
`summary_by_method.md`, `config_used.yaml`, per-image contact sheets, per-method
outputs, and per-method diagnostics.

## Structural Visualization

For one saved image result:

```bash
conda run -n DPS python scripts/bcns/visualize_structural_metrics.py \
  --result_dir results/bcns_step4/ffhq_ablation/00000 \
  --output_dir results/bcns_step4/ffhq_ablation/00000/structural_metrics
```

The script saves a boundary band, label/reconstruction gradient magnitudes,
absolute Laplacian map, NS residual heatmap, a simple isophote overlay, and a
contact sheet.

## Smoke Commands

Thin scratch:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step4/ffhq_thin_scratch_smoke \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 100 --ablation_set smoke
```

Center box:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step4/ffhq_center_box_smoke \
  --gpu 0 --num_images 4 --seed 123 \
  --mask_mode center_box --box_size 96 \
  --record_every 100 --ablation_set smoke
```

Frequency ablation:

```bash
conda run -n DPS python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_projected_ffhq.yaml \
  --save_dir results/bcns_step4/ffhq_frequency_ablation \
  --gpu 0 --num_images 2 --seed 123 \
  --mask_mode thin_scratch --scratch_thickness 5 \
  --record_every 100 --ablation_set frequency
```

## Interpretation and Limits

Projected methods should reduce raw known-region error. BCNS methods should be
judged mainly by hole, seam, gradient, isophote, and Laplacian metrics. NS
residual is a PDE consistency diagnostic, not a perceptual score. Step 4 does
not require LPIPS, does not provide a full benchmark, and does not implement
carry-over PDE memory or exact local MAP.
