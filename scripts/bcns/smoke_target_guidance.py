#!/usr/bin/env python
"""Smoke test BCNS target-guidance conditioning without model checkpoints."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcns.debug_utils import update_norm
from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "gaussian"


def _make_tensors():
    torch.manual_seed(7)
    shape = (1, 3, 8, 8)
    x_prev = torch.randn(shape, requires_grad=True)
    x_t = torch.randn(shape)
    x_0_hat = x_prev * 0.5 + 0.1
    measurement = torch.randn(shape)
    mask_known = torch.ones((1, 1, 8, 8))
    mask_known[..., 2:6, 2:6] = 0.0
    return x_prev, x_t, x_0_hat, measurement, mask_known


def _run_builder(name, **params):
    x_prev, x_t, x_0_hat, measurement, mask_known = _make_tensors()
    method = get_conditioning_method(
        "bcns_target",
        operator=None,
        noiser=_Noiser(),
        target_builder=name,
        target_builder_params=params,
        gamma_max=0.05,
        tau2=1.0,
    )
    out, loss = method.conditioning(
        x_prev=x_prev,
        x_t=x_t.clone(),
        x_0_hat=x_0_hat,
        measurement=measurement,
        mask=mask_known,
    )
    print(
        f"{name}: loss={float(loss.item()):.8f}, "
        f"update_norm={update_norm(x_t, out):.8f}, "
        f"known_sum={float(mask_known.sum().item()):.1f}, "
        f"hole_sum={float((1.0 - mask_known).sum().item()):.1f}"
    )


def main():
    _run_builder("identity")
    _run_builder("hard_projection")
    _run_builder("simple_hole_shift", shift=0.05)


if __name__ == "__main__":
    main()
