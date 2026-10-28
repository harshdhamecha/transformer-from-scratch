"""Tests for causal mask, padding mask, and combining them."""

import torch
from transformer.masks import causal_mask
from transformer.attention import attention


def test_causal_mask():
    # M1.9: Causal mask helper
    T = 4
    mask = causal_mask(T)
    mask_written = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True]
    ])
    assert mask.shape == (T, T)
    assert mask.dtype == torch.bool
    assert torch.equal(mask_written, mask)


def test_leakage():
    # M1.4: No future leakage
    q1 = torch.randn(2, 6, 4)
    k1 = torch.randn(2, 6, 4)
    v1 = torch.randn(2, 6, 4)
    out1, _ = attention(q1, k1, v1, mask=causal_mask(6))
    k2 = k1.clone()
    v2 = v1.clone()
    k2[:, 5, :] = 0.0034
    v2[:, 5, :] = 0.0026
    out2, _ = attention(q1, k2, v2, mask=causal_mask(6))
    torch.testing.assert_close(out1[:, :5, :], out2[:, :5, :])
    assert torch.all(out1[:, 5, :] != out2[:, 5, :]) 
    


def test_masked_keys():
    # M1.5: Masked keys get zero weight.
    query = torch.randn(1, 5, 3)  # (batch, T_q, d_k)
    key = torch.randn(1, 4, 3)  # (batch, T_k, d_k)
    value = torch.randn(1, 4, 3)  # (batch, T_k, d_v)
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False]
    ])
    _, weights = attention(query, key, value, mask=mask)
    assert weights[:, :, -2:].sum() == 0


def test_masked_row():
    # M1.6: Fully masked row
    query = torch.randn(1, 2, 3)  # (batch, T_q, d_k)
    key = torch.randn(1, 2, 3)  # (batch, T_k, d_k)
    value = torch.randn(1, 2, 3)  # (batch, T_k, d_v)
    mask = torch.tensor([[False, False], [True, True]])
    output1, _ = attention(query, key, value, mask=mask)
    output2, _ = attention(query, key, value)
    assert not torch.isnan(output1).any()
    assert torch.all(output1[:, 1, :] == output2[:, 1, :])
    torch.testing.assert_close(value.mean(dim=1), output1[:, 0, :])