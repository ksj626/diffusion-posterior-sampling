import torch

from bcns.losses import structure_displacement_loss, target_discrepancy_loss
from bcns.proximal import structure_image
from bcns.target_builders import TargetBuildResult
from guided_diffusion.condition_methods import BCNSTargetGuidance


class _Noiser:
    __name__ = "clean"


class _KnownOnlyBuilder:
    def __call__(self, mu, measurement, mask_known, **kwargs):
        target = mu.detach() + 5.0 * mask_known.to(dtype=mu.dtype)
        return TargetBuildResult(target=target, diagnostics={"target_builder": "known_only"})


class _StructureHoleBuilder:
    def __call__(self, mu, measurement, mask_known, **kwargs):
        target_structure = structure_image(mu, 0.0).detach() + 2.0 * (1.0 - mask_known)
        return TargetBuildResult(
            target=mu.detach(),
            diagnostics={"target_builder": "structure_hole"},
            target_structure=target_structure,
        )


def _inputs():
    torch.manual_seed(91)
    shape = (1, 3, 6, 6)
    x_prev = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_t = torch.randn(shape, dtype=torch.float64)
    x_0_hat = x_prev * 0.25
    measurement = torch.zeros(shape, dtype=torch.float64)
    mask_known = torch.ones((1, 1, 6, 6), dtype=torch.float64)
    mask_known[..., 2:5, 2:5] = 0.0
    return x_prev, x_t, x_0_hat, measurement, mask_known


def _method(rgb_weight=1.0, structure_weight=0.0):
    method = BCNSTargetGuidance(
        operator=None,
        noiser=_Noiser(),
        target_builder="identity",
        gamma_max=1.0,
        tau2=1.0,
        rgb_loss_weight=rgb_weight,
        structure_loss_weight=structure_weight,
        structure_loss_sigma=0.0,
    )
    return method


def test_rgb_known_only_target_gives_zero_hole_loss_and_no_update():
    x_prev, x_t, x_0_hat, measurement, mask_known = _inputs()
    method = _method(rgb_weight=1.0, structure_weight=0.0)
    method.target_builder = _KnownOnlyBuilder()
    out, loss = method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    assert loss.item() == 0.0
    assert method.last_diagnostics["target_disp_known"] > 0.0
    assert method.last_diagnostics["target_disp_hole"] == 0.0
    assert method.last_diagnostics["grad_norm"] == 0.0
    assert torch.allclose(out, x_t)


def test_structure_hole_target_activates_loss_and_gradient():
    x_prev, x_t, x_0_hat, measurement, mask_known = _inputs()
    method = _method(rgb_weight=0.0, structure_weight=0.1)
    method.target_builder = _StructureHoleBuilder()
    out, loss = method.conditioning(x_prev, x_t.clone(), x_0_hat, measurement, mask=mask_known)
    assert loss.item() > 0.0
    assert method.last_diagnostics["rgb_loss"] == 0.0
    assert method.last_diagnostics["structure_loss"] > 0.0
    assert method.last_diagnostics["structure_disp_hole"] > 0.0
    assert method.last_diagnostics["grad_norm"] > 0.0
    assert not torch.allclose(out, x_t)


def test_structure_loss_disabled_matches_old_target_loss():
    x_prev, _, x_0_hat, measurement, mask_known = _inputs()
    builder = _StructureHoleBuilder()
    result = builder(x_0_hat, measurement, mask_known)
    rgb_loss = target_discrepancy_loss(x_0_hat, result.target, mask_known, tau2=1.0)
    struct_loss, _ = structure_displacement_loss(
        x_0_hat,
        result.target_structure,
        mask_known,
        structure_sigma=0.0,
        weight=0.0,
    )
    assert torch.allclose(rgb_loss + struct_loss, rgb_loss)
    assert torch.allclose(rgb_loss, torch.zeros_like(rgb_loss))
