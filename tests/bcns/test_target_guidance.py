import torch

from bcns.losses import target_discrepancy_loss
from bcns.target_builders import (
    HardProjectionTargetBuilder,
    IdentityTargetBuilder,
    SimpleHoleShiftTargetBuilder,
)
from guided_diffusion.condition_methods import get_conditioning_method


class _Noiser:
    __name__ = "gaussian"


def _fake_tensors():
    torch.manual_seed(3)
    shape = (1, 3, 8, 8)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = x_prev * 0.5 + 0.1
    measurement = torch.randn(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 8, 8), dtype=torch.float64)
    mask_known[..., 2:6, 2:6] = 0.0
    return x_prev, x_t, x_0_hat, measurement, mask_known


def test_identity_target_builder_returns_mu():
    _, _, mu, measurement, mask_known = _fake_tensors()
    result = IdentityTargetBuilder()(mu, measurement, mask_known)
    assert result.target is mu
    assert torch.equal(result.target, mu)


def test_target_discrepancy_loss_zero_for_identity_and_detaches_target():
    _, _, mu, measurement, mask_known = _fake_tensors()
    identity_target = IdentityTargetBuilder()(mu, measurement, mask_known).target
    identity_loss = target_discrepancy_loss(mu, identity_target, mask_known, tau2=1.0)
    assert torch.allclose(identity_loss, torch.zeros((), dtype=mu.dtype))

    independent_target = (mu.detach() + 0.25).requires_grad_(True)
    detached_loss = target_discrepancy_loss(mu, independent_target, mask_known, tau2=1.0)
    grad_target = torch.autograd.grad(detached_loss, independent_target, allow_unused=True)[0]
    assert grad_target is None


def test_hard_projection_target_known_equals_measurement_hole_equals_mu():
    _, _, mu, measurement, mask_known = _fake_tensors()
    target = HardProjectionTargetBuilder()(mu, measurement, mask_known).target
    known = mask_known.expand_as(mu).bool()
    assert torch.equal(target[known], measurement[known])
    assert torch.equal(target[~known], mu[~known])


def test_simple_hole_shift_creates_hole_displacement():
    _, _, mu, measurement, mask_known = _fake_tensors()
    shift = 0.05
    target = SimpleHoleShiftTargetBuilder(shift=shift)(mu, measurement, mask_known).target
    known = mask_known.expand_as(mu).bool()
    hole_delta = target[~known] - mu[~known]
    assert torch.allclose(hole_delta, torch.full_like(hole_delta, shift))
    assert torch.equal(target[known], measurement[known])


def test_bcns_target_identity_returns_same_x_t():
    x_prev, x_t, x_0_hat, measurement, mask_known = _fake_tensors()
    method = get_conditioning_method(
        "bcns_target",
        operator=None,
        noiser=_Noiser(),
        target_builder="identity",
        gamma_max=0.05,
        tau2=1.0,
    )
    out, loss = method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    assert torch.allclose(out, x_t)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_bcns_target_simple_hole_shift_changes_x_t_with_finite_loss():
    x_prev, x_t, x_0_hat, measurement, mask_known = _fake_tensors()
    method = get_conditioning_method(
        "bcns_target",
        operator=None,
        noiser=_Noiser(),
        target_builder="simple_hole_shift",
        target_builder_params={"shift": 0.1},
        gamma_max=0.5,
        tau2=1.0,
    )
    out, loss = method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert not torch.allclose(out, x_t)


def test_existing_conditioning_methods_still_register():
    for name in ("vanilla", "projection", "mcg", "ps", "ps+", "bcns_target"):
        get_conditioning_method(name, operator=None, noiser=_Noiser())
