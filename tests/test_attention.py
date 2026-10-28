"""Tests for scaled dot-product attention and multi-head attention."""


import torch
from transformer.attention import attention
from transformer.masks import causal_mask
import torch.nn.functional as F


def test_attention():
    # M1.1: Shapes
    query = torch.randn(2, 5, 4)  # (batch, T_q, d_k)
    key = torch.randn(2, 7, 4)  # (batch, T_k, d_k)
    value = torch.randn(2, 7, 6)  # (batch, T_k, d_v)
    output, weights = attention(query, key, value)
    assert output.shape == (2, 5, 6)
    assert weights.shape == (2, 5, 7)
    torch.testing.assert_close(weights.sum(axis=2), torch.ones(2, 5))


def test_attention_weights():
    # M1.2: Rows sum to 1
    # M1.3: Causal zeros
    query = torch.randn(2, 6, 4)  # (batch, T_q, d_k)
    key = torch.randn(2, 6, 4)  # (batch, T_k, d_k)
    value = torch.randn(2, 6, 4)  # (batch, T_k, d_v)
    _, weights = attention(query, key, value, mask=causal_mask(6))
    assert torch.all(torch.triu(weights, diagonal=1) == 0)
    torch.testing.assert_close(weights.sum(axis=2), torch.ones(2, 6), atol=1e-6, rtol=0)


def test_attention_extra_dim():
    # M1.8: Extra dimensions
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 3, 5, 4)
    value = torch.randn(2, 3, 5, 4)
    output, weights = attention(query, key, value)
    assert output.shape == (2, 3, 5, 4)
    assert weights.shape == (2, 3, 5, 5)
    key2 = key.clone()
    value2 = value.clone()
    key2[:, 1, :, :] = torch.randn(2, 5, 4)
    value2[:, 1, :, :] = torch.randn(2, 5, 4)
    output2, weights2 = attention(query, key2, value2)
    assert torch.all(output[:, 0, :, :] == output2[:, 0, :, :])
    assert torch.all(output[:, 2, :, :] == output2[:, 2, :, :])
    assert torch.all(output[:, 1, :, :] != output2[:, 1, :, :])


def test_attention_torch():
    # M1.7: Reference Match
    q1 = torch.randn(2, 6, 4)
    k1 = torch.randn(2, 6, 4)
    v1 = torch.randn(2, 6, 4)
    torch_out1 = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=causal_mask(6))
    out1, _ = attention(q1, k1, v1, mask=causal_mask(6))
    torch.testing.assert_close(torch_out1, out1, atol=1e-5, rtol=0)
    torch_out2 = F.scaled_dot_product_attention(q1, k1, v1)
    out2, _ = attention(q1, k1, v1)
    torch.testing.assert_close(torch_out2, out2, atol=1e-5, rtol=0)
    q2 = torch.randn(2, 3, 5, 4)
    k2 = torch.randn(2, 3, 7, 4)
    v2 = torch.randn(2, 3, 7, 6)
    torch_out3 = F.scaled_dot_product_attention(q2, k2, v2)
    out3, _ = attention(q2, k2, v2)
    torch.testing.assert_close(torch_out3, out3, atol=1e-5, rtol=0)

