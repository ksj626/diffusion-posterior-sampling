# BCNS DPS Integration Step 0

This step adds a minimal target-guidance conditioning skeleton for debugging
mask conventions, target construction, and autograd updates inside DPS. It does
not add full Boundary-Conditioned Navier-Stokes flow guidance.

## Not Implemented Yet

This step intentionally does not implement Poisson or harmonic targets inside
diffusion, finite-step PDE flow targets, local-MAP residual minimization,
`BCNSFlowDPS`, sampler changes, dataset changes, checkpoint loading, or model
inference requirements.

## Mask Conventions

The DPS inpainting operator uses `y = data * mask`, so inside DPS conditioning
`mask == 1` means a known observed pixel.

The standalone BCNS numerical helpers use `mask_unknown == 1` to mean a missing
pixel in the hole. The adapter utilities in `bcns/dps_adapter.py` explicitly
convert between these conventions with `1 - mask`.

## Conditioning Math

The debug target loss is

```text
L_t = 1 / (2 tau_t^2) * mean_hole ||mu_t - sg[target_t]||^2
```

The reverse proposal update is

```text
x_{t-1} = x'_{t-1} - gamma_t * grad L_t
```

In the current DPS sampler, timestep information is not passed into
conditioning methods. The `BCNSSchedule` API is present for later use, but
`bcns_target` currently uses constant `gamma_max` unless a future sampler passes
`t_index` and `num_steps`.

## Debug Target Builders

`identity` returns `target = mu`, so the loss and update should be zero.

`hard_projection` enforces observed clean pixels in the target while leaving
hole pixels equal to `mu`, so the hole-only target loss is also zero.

`simple_hole_shift` shifts hole pixels by a small constant. It is deliberately
synthetic and exists only to verify nonzero finite autograd guidance without a
PDE target.

## Commands

```bash
conda run -n DPS python -m pytest tests/bcns/test_dps_adapter.py tests/bcns/test_target_guidance.py -q
conda run -n DPS python scripts/bcns/smoke_target_guidance.py
```

The smoke script does not require checkpoints or a diffusion model.
