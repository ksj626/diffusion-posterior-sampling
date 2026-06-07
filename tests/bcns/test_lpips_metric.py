import math
import sys
import types

import torch

from bcns import eval_metrics
from bcns.eval_metrics import hole_focused_lpips, lpips_metric
from bcns.eval_table import evaluate_inpainting_result


class _FakeLPIPS:
    def __init__(self, net="alex"):
        self.net = net

    def to(self, device=None):
        return self

    def eval(self):
        return self

    def __call__(self, pred, target):
        return (pred - target).abs().mean(dim=(1, 2, 3), keepdim=True)


def test_lpips_metric_import_path_with_fake_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", types.SimpleNamespace(LPIPS=_FakeLPIPS))
    eval_metrics._LPIPS_CACHE.clear()
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.ones_like(pred) * 0.25

    value = lpips_metric(pred, target)

    assert isinstance(value, float)
    assert math.isfinite(value)
    assert value > 0.0


def test_hole_focused_lpips_returns_finite_scalar():
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.ones_like(pred) * 0.5
    mask_known = torch.ones(1, 1, 8, 8)
    mask_known[..., 2:6, 2:6] = 0.0

    value = hole_focused_lpips(pred, target, mask_known, loss_fn=_FakeLPIPS())

    assert isinstance(value, float)
    assert math.isfinite(value)
    assert value > 0.0


def test_evaluation_summary_includes_lpips_columns():
    label = torch.zeros(1, 3, 8, 8)
    raw = label.clone()
    raw[..., 2:6, 2:6] = 0.1
    mask_known = torch.ones(1, 1, 8, 8)
    mask_known[..., 2:6, 2:6] = 0.0
    measurement = label * mask_known
    composite = raw * (1.0 - mask_known) + measurement * mask_known

    row = evaluate_inpainting_result(
        recon_raw=raw,
        recon_composite=composite,
        label=label,
        measurement=measurement,
        mask_known=mask_known,
        require_lpips=True,
        lpips_loss_fn=_FakeLPIPS(),
        boundary_width=1,
    )

    for key in ("raw_lpips", "composite_lpips", "raw_hole_lpips", "composite_hole_lpips"):
        assert key in row
        assert isinstance(row[key], float)
        assert math.isfinite(row[key])
