"""Scaled dot-product attention (M1) and multi-head attention (M2)."""


import numpy as np
import torch
import math


def attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention.<br>
    Input shape - q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v).
    Output shape - (..., T_q, d_v).
    Mask convention: True indicates positions that should be attended to.
    """
    d_k = query.shape[-1]
    score = torch.matmul(query, torch.transpose(key, -1, -2))
    scaled_score = score / math.sqrt(d_k)
    weighted_score = torch.softmax(scaled_score, dim=-1)
    output = torch.matmul(weighted_score, value)
    return output, weighted_score