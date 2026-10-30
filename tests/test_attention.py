"""Tests for scaled dot-product attention and multi-head attention."""


import torch
from transformer.attention import attention, MultiHeadAttention
from transformer.masks import causal_mask
import torch.nn.functional as F
import torch.nn as nn



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


def test_multihead_attn_shape():
    # M2.1: Output shape equals query shape
    B, T_q, T_k = 2, 5, 7
    d_model, h = 6, 3
    query = torch.randn(B, T_q, d_model)
    key = torch.randn(B, T_k, d_model)
    value = torch.randn(B, T_k, d_model)
    multihead_attn = MultiHeadAttention(d_model, h)
    output, weights = multihead_attn(query, key, value)
    assert output.shape == (B, T_q, d_model)
    assert weights.shape == (B, h, T_q, T_k)


def test_multihead_params_count():
    # M2.2: Parameter count - 4 * square(d_model) + 4 * d_model
    d_model, h = 6, 3
    params1 = 4 * d_model * d_model + 4 * d_model
    multihead_attn1 = MultiHeadAttention(d_model, h)
    params2 = sum(p.numel() for p in multihead_attn1.parameters())
    assert params1 == params2


def test_multihead_reference_non_mask():
    # M2.3.1: Check output with torch's reference without mask
    B, T_q, T_k = 2, 5, 7
    d_model, h = 6, 3
    query = torch.randn(B, T_q, d_model)
    key = torch.randn(B, T_k, d_model)
    value = torch.randn(B, T_k, d_model)
    multihead_attn1 = MultiHeadAttention(d_model, h)
    output1, _ = multihead_attn1(query, key, value)
    multihead_attn2 = nn.MultiheadAttention(d_model, h, batch_first=True)
    weight_names = [multihead_attn1.w_q.weight, multihead_attn1.w_k.weight, multihead_attn1.w_v.weight]
    bias_names = [multihead_attn1.w_q.bias, multihead_attn1.w_k.bias, multihead_attn1.w_v.bias]
    in_weights = torch.cat(weight_names, dim=0)
    in_bias = torch.cat(bias_names, dim=0)
    with torch.no_grad():
        multihead_attn2.in_proj_weight.copy_(in_weights)
        multihead_attn2.in_proj_bias.copy_(in_bias)
        multihead_attn2.out_proj.weight.copy_(multihead_attn1.w_o.weight)
        multihead_attn2.out_proj.bias.copy_(multihead_attn1.w_o.bias)
    output2, _ = multihead_attn2(query, key, value)
    torch.testing.assert_close(output2, output1, atol=1e-5, rtol=0)


def test_multihead_reference_masked():
    # M2.3.2: Check output with torch's reference with mask
    B, T_q, T_k = 2, 5, 5
    d_model, h = 6, 3
    query = torch.randn(B, T_q, d_model)
    key = torch.randn(B, T_k, d_model)
    value = torch.randn(B, T_k, d_model)
    multihead_attn1 = MultiHeadAttention(d_model, h)
    mask = causal_mask(5)
    output1, _ = multihead_attn1(query, key, value, mask=mask)
    mask_ = ~mask
    multihead_attn2 = nn.MultiheadAttention(d_model, h, batch_first=True)
    weight_names = [multihead_attn1.w_q.weight, multihead_attn1.w_k.weight, multihead_attn1.w_v.weight]
    bias_names = [multihead_attn1.w_q.bias, multihead_attn1.w_k.bias, multihead_attn1.w_v.bias]
    in_weights = torch.cat(weight_names, dim=0)
    in_bias = torch.cat(bias_names, dim=0)
    with torch.no_grad():
        multihead_attn2.in_proj_weight.copy_(in_weights)
        multihead_attn2.in_proj_bias.copy_(in_bias)
        multihead_attn2.out_proj.weight.copy_(multihead_attn1.w_o.weight)
        multihead_attn2.out_proj.bias.copy_(multihead_attn1.w_o.bias)
    output2, _ = multihead_attn2(query, key, value, attn_mask=mask_)
    torch.testing.assert_close(output2, output1, atol=1e-5, rtol=0)


def test_multihead_leakage():
    # M2.5: No future leakage
    B, T = 2, 5
    d_model, h = 6, 3
    input_ = torch.randn(B, T, d_model)
    multihead_attn1 = MultiHeadAttention(d_model, h)
    mask = causal_mask(5)
    output1, _ = multihead_attn1(input_, input_, input_, mask=mask)
    input__ = input_.clone()
    input__[:, 4, :] = 2.4354
    output2, _ = multihead_attn1(input__, input__, input__, mask=mask)
    assert torch.all(output1[:, :4, :] == output2[:, :4, :])
    assert torch.all(output1[:, 4, :] != output2[:, 4, :])