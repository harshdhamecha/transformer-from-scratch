"""Tests for LayerNorm, position-wise feed-forward, positional encoding, and token embeddings."""

import torch.nn.functional as F
import torch
from transformer.layers import LayerNorm, Embeddings, FeedForward, PositionalEncoding



def test_layer_norm():
    # M3.1: Reference check
    B, T, d_model = 2, 3, 4
    eps = 0.1
    x = torch.randn(B, T, d_model)
    weight = torch.randn(d_model)
    bias = torch.randn(d_model)
    layernorm1 = LayerNorm(d_model, eps)
    assert torch.equal(layernorm1.gamma, torch.ones(d_model))
    assert torch.equal(layernorm1.beta, torch.zeros(d_model))
    with torch.no_grad():
        layernorm1.gamma.copy_(weight)
        layernorm1.beta.copy_(bias)
    output1 = layernorm1(x)
    output2 = F.layer_norm(x, (d_model,), weight=weight, bias=bias, eps=eps)
    torch.testing.assert_close(output1, output2, atol=1e-5, rtol=0)


def test_embedding():
    # M3.4: Embedding Output shapes
    vocab_size = 10
    B, T, d_model = 2, 3, 4
    token = torch.randint(low=0, high=vocab_size, size=(B, T))
    embedding = Embeddings(vocab_size, d_model)
    output = embedding(token)
    assert output.shape == (B, T, d_model)


def test_ffn():
    # M3.4: Output shapes and params
    B, T, d_model = 2, 3, 4
    d_ff = 16
    x = torch.randn(B, T, d_model)
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    output = ffn(x)
    assert output.shape == (B, T, d_model)
    params1 = 2 * (d_model * d_ff) + d_model + d_ff
    params2 = sum(p.numel() for p in ffn.parameters())
    assert params1 == params2


def test_pe():
    # M3.2: 1st row check
    # M3.3: Parameters is empty, table key in state_dict() and it doesn't require grad
    # M3.4: Output value and shape
    d_model, max_len = 4, 10
    B, T = 2, 3
    input = torch.zeros(B, T, d_model)
    pe = PositionalEncoding(d_model, max_len, dropout=0.0)
    first_row = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    params = sum(p.numel() for p in pe.parameters())
    assert params == 0
    assert torch.equal(pe.lookup_table[0], first_row)
    assert 'lookup_table' in pe.state_dict().keys()
    assert not pe.lookup_table.requires_grad
    torch.testing.assert_close(pe.lookup_table[1, 2].item(), 0.01, atol=1e-4, rtol=0)
    output = pe(input)
    torch.testing.assert_close(output[0, 0, :], pe.lookup_table[0])
    torch.testing.assert_close(output[0, 1, :], pe.lookup_table[1])
    assert output.shape == input.shape
