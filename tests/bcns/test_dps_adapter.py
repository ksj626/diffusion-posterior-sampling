import torch

from bcns.dps_adapter import (
    bcns_unknown_to_dps_known,
    dps_known_to_bcns_unknown,
    hard_project_clean,
    masked_mean_square,
)


def test_known_to_unknown_conversion_round_trips():
    mask_known = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    mask_unknown = dps_known_to_bcns_unknown(mask_known)
    assert torch.equal(mask_unknown, torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]))
    assert torch.equal(bcns_unknown_to_dps_known(mask_unknown), mask_known)


def test_hard_project_clean_uses_measurement_on_known_and_clean_on_hole():
    x_clean = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
    measurement = torch.full_like(x_clean, -2.0)
    mask_known = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    projected = hard_project_clean(x_clean, measurement, mask_known)
    mask_c = mask_known.expand_as(x_clean).bool()
    assert torch.equal(projected[mask_c], measurement[mask_c])
    assert torch.equal(projected[~mask_c], x_clean[~mask_c])


def test_masked_mean_square_broadcasts_channel_mask():
    x = torch.ones((1, 3, 2, 2), dtype=torch.float64)
    x[..., :, 0, 0] = 2.0
    mask = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    mask[..., 0, 0] = 1.0
    assert torch.allclose(masked_mean_square(x, mask), torch.tensor(4.0, dtype=torch.float64))
