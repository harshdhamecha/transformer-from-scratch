"""Tests for causal mask, padding mask, and combining them."""

import torch
from transformer.masks import causal_mask



def test_causal_mask():
    T = 4
    mask = causal_mask(T)
    assert mask.shape == (T, T)
    assert mask.dtype == torch.bool
    assert torch.equal(torch.tril(torch.ones(T, T, dtype=torch.bool)), mask)
    