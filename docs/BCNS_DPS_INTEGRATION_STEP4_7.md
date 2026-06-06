# BCNS DPS Integration Step 4.7: Structured Masks And Lift Candidates

Step 4.6 showed that fully converged 1000-step DPS is already a strong
inpainting baseline, thin scratch masks are often too easy, and the most useful
BCNS-style signal is likely in few-step sampling with harder structured
occlusions. Step 4.7 adds the mask and ablation harness needed to test that
setting.

No new PDE solver, RGB Navier-Stokes method, or final benchmark is added here.

## Structured Mask Suite

BCNS mask helpers use `mask_unknown == 1`; the DPS runner converts to
`mask_known = 1 - mask_unknown`.

New structured masks:

- `thin_scratch`: legacy thin line scratch, usually easy.
- `thick_scratch_12` and `thick_scratch_24`: wider crossing strokes.
- `text_mask`: block-letter occlusion without external font files.
- `freeform_medium`: medium random strokes.
- `center_box_96` and `center_box_128`: large square holes for sanity checks.

Each generated mask logs:

```text
hole_ratio
known_ratio
boundary_length_approx
num_components_approx
bbox_area_ratio
```

Thin scratch masks can be too easy because the missing region is narrow and
projection plus DPS priors often recover enough local texture. Huge semantic
holes are also not ideal for PDE claims, because structural boundary continuation
is less informative when the hole asks for high-level semantic synthesis.
Thick scratches, text-like masks, and medium freeform masks are the intended
middle ground.

## Luminance-Lift Candidate Sets

The Step 4.7 candidate methods focus on harmonic and Poisson scalar targets
with luminance-lift transfer. The distinction:

- NS-flow target source: evolves an initial scalar structure with the finite
  BCNS flow target machinery; currently weaker in short tests.
- Harmonic/Poisson target source: solves scalar structure inside the hole from
  known boundary structure and optional RHS.
- Luminance-lift transfer: directly lifts scalar residual
  `target_structure - S_sigma(mu)` into RGB hole displacement, then hard
  projects known pixels.

New ablation sets:

- `lift_candidates`: baselines plus harmonic/Poisson lift scales `0.5`, `1.0`,
  `2.0`, with projected `1.0` variants.
- `lift_fewstep`: steps `{25,50,100,250,1000}` over baselines and harmonic/
  Poisson lift `1.0` projected/non-projected candidates.
- `lift_scale`: scales `{0.25,0.5,1.0,2.0,4.0}` across harmonic/Poisson and
  projected/non-projected variants.

Rows include plotting fields:

```text
method_family
target_source
transfer_mode
projected
lift_scale
sampling_steps
```

## Manual Commands

Each process sees one GPU through `CUDA_VISIBLE_DEVICES`, so use `--gpu 0`
inside the runner.

### Visualize mask suite

```bash
python scripts/bcns/visualize_mask_suite.py \
  --output-dir results/bcns_step4_7/mask_suite \
  --height 256 \
  --width 256
```

### Mask difficulty sweep, 100 steps

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/mask_thick12_gpu0 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode thick_scratch_12 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_candidates
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/mask_thick24_gpu1 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode thick_scratch_24 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_candidates
```

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/mask_text_gpu2 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode text_mask \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_candidates
```

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/mask_freeform_gpu3 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode freeform_medium \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_candidates
```

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/mask_center96_gpu4 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode center_box_96 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_candidates
```

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/mask_center128_gpu5 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode center_box_128 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_candidates
```

### Few-step sweep on expected hard masks

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/fewstep_thick12_gpu6 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode thick_scratch_12 \
  --record_every 50 \
  --ablation_set lift_fewstep
```

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/fewstep_text_gpu7 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode text_mask \
  --record_every 50 \
  --ablation_set lift_fewstep
```

### Lift-scale sweep

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bcns/run_bcns_step4_ablation.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml \
  --save_dir results/bcns_step4_7/lift_scale_thick12_gpu0 \
  --gpu 0 --num_images 8 --seed 123 \
  --mask_mode thick_scratch_12 \
  --record_every 50 --sampling_steps 100 \
  --ablation_set lift_scale
```

## What To Inspect

Important fields:

```text
composite_hole_mse
composite_seam_mse
composite_gradient_mismatch
composite_isophote_error
composite_laplacian_mismatch
target_disp_hole
sum_update_norm
hole_ratio
boundary_length_approx
lift_scale
sampling_steps
projected
```

Expected interpretation:

- Thin scratch may stay saturated and too easy.
- Thick scratch, text, and freeform masks should reveal larger method
  differences.
- 1000-step DPS may still dominate; 50/100-step results are more important.
- Harmonic/Poisson luminance-lift should create nonzero target displacement.
- Very large lift scales may create visible artifacts.

Limitations: no RGB PDE yet, no new solver, and no final benchmark.
