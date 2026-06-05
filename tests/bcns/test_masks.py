from collections import deque

import torch

from bcns.masks import (
    known_collar,
    make_center_box_mask,
    make_disconnected_box_mask,
)


def _component_count(mask):
    arr = mask[0, 0].bool()
    seen = torch.zeros_like(arr)
    count = 0
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            if not arr[y, x] or seen[y, x]:
                continue
            count += 1
            queue = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and arr[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
    return count


def test_center_box_mask_area_and_convention():
    mask = make_center_box_mask(12, 14, 4, 6, dtype=torch.float64)
    assert mask.shape == (1, 1, 12, 14)
    assert mask.sum().item() == 24
    assert mask.max().item() == 1


def test_known_collar_excludes_unknown_pixels():
    mask = make_center_box_mask(16, 16, 4, 4)
    collar = known_collar(mask, width=2)
    assert torch.all(collar[mask.bool()] == 0)
    assert collar.sum().item() > 0


def test_disconnected_boxes_remain_disconnected():
    mask = make_disconnected_box_mask(
        20,
        20,
        boxes=((3, 3, 3, 3), (13, 13, 3, 3)),
    )
    assert _component_count(mask) == 2
