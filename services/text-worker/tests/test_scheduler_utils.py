from __future__ import annotations

import torch

from app.scheduler import _merge_past, _past_nbytes, _split_past


def test_past_split_merge_roundtrip() -> None:
    past = ((torch.ones(2, 4, 3, 8), torch.zeros(2, 4, 3, 8)),)
    first = _split_past(past, 0)
    second = _split_past(past, 1)
    merged = _merge_past([first, second])
    assert torch.equal(merged[0][0], past[0][0])
    assert torch.equal(merged[0][1], past[0][1])


def test_past_nbytes_counts_tensors() -> None:
    past = ((torch.ones(1, 1, 2, dtype=torch.float32), torch.ones(1, 1, 2, dtype=torch.float32)),)
    assert _past_nbytes(past) == 16

