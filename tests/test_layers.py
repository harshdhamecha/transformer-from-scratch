"""Tests for LayerNorm, position-wise feed-forward, positional encoding, and token embeddings."""

import torch.nn.functional as F
import torch
from transformer.layers import LayerNorm, Embeddings



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