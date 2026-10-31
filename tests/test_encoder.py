"""Tests for the encoder layer and encoder stack."""

import torch
from transformer.encoder import EncoderLayer, Encoder



def test_encoder_layer():
    # M4.2: Shapes, Parameter count
    # M4.3: Identical outputs
    B, h, T, d_model = 2, 2, 3, 8
    d_ff = 16
    input = torch.randn(B, T, d_model)
    encoder = EncoderLayer(d_model, h, d_ff, dropout=0.1)
    encoder.eval()
    output1 = encoder(input)
    output2 = encoder(input)
    assert output1.shape == (B, T, d_model)
    assert torch.equal(output1, output2)
    encoder.train()
    output1 = encoder(input)
    output2 = encoder(input)
    assert not torch.equal(output1, output2)
    expected_params = ((4 * d_model * d_model) + (4 * d_model)) \
        + ((2 * d_model * d_ff) + (d_model + d_ff)) + \
        2 * 2 * d_model
    actual_params = sum(p.numel() for p in encoder.parameters())
    assert expected_params == actual_params


def test_encoder():
    # M4.2: Shapes, Parameter count
    # M4.3: Identical outputs
    N, B, T, max_len = 3, 2, 6, 6
    vocab_size, d_model, h, d_ff = 10, 8, 2, 16
    encoder = Encoder(N, vocab_size, max_len, d_model, h, d_ff, 0.1)
    input = torch.tensor([
        [3, 2, 1, 4, 5, 6],
        [6, 4, 1, 5, 6, 3]
    ])
    output = encoder(input)
    assert output.shape == (B, T, d_model)
    encoder.eval()
    output1 = encoder(input)
    output2 = encoder(input)
    assert output1.shape == (B, T, d_model)
    assert torch.equal(output1, output2)
    encoder.train()
    output1 = encoder(input)
    output2 = encoder(input)
    assert not torch.equal(output1, output2)
    expected_params = N * (((4 * d_model * d_model) + (4 * d_model)) \
        + ((2 * d_model * d_ff) + (d_model + d_ff)) + \
        (2 * 2 * d_model)) + (vocab_size * d_model)
    actual_params = sum(p.numel() for p in encoder.parameters())
    assert expected_params == actual_params