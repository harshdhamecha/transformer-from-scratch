"""Tests for scaled dot-product attention and multi-head attention."""


import torch
from transformer.attention import attention
from transformer.masks import causal_mask



def test_attention():
    query = torch.rand(2, 5, 4)  # (batch, T_q, d_k)
    key = torch.rand(2, 7, 4)  # (batch, T_k, d_k)
    value = torch.rand(2, 7, 6)  # (batch, T_k, d_v)
    output, weights = attention(query, key, value)
    assert output.shape == (2, 5, 6)
    assert weights.shape == (2, 5, 7)
    torch.testing.assert_close(weights.sum(axis=2), torch.ones(2, 5))


def test_attention_weights():
    query = torch.rand(2, 6, 4)  # (batch, T_q, d_k)
    key = torch.rand(2, 6, 4)  # (batch, T_k, d_k)
    value = torch.rand(2, 6, 4)  # (batch, T_k, d_v)
    _, weights = attention(query, key, value, mask=causal_mask(6))
    torch.testing.assert_close(torch.triu(weights, diagonal=1), torch.zeros_like(weights))
    torch.testing.assert_close(weights.sum(axis=2), torch.ones(2, 6))


def test_attention_extra_dim():
    query = torch.rand(2, 3, 5, 4)  # (batch, T_q, d_k)
    key = torch.rand(2, 3, 5, 4)  # (batch, T_k, d_k)
    value = torch.rand(2, 3, 5, 4)  # (batch, T_k, d_v)
    output, weights = attention(query, key, value)
    assert output.shape == (2, 3, 5, 4)
    assert weights.shape == (2, 3, 5, 5)


def test_attention_torch():
    pass
    # TODO: 7. Reference Match

