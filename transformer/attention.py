"""Scaled dot-product attention (M1) and multi-head attention (M2)."""


import numpy as np
import torch


def attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention.<br>
    Input shape - q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v).
    Output shape - (..., T_q, d_v).
    Mask convention: True indicates positions that should be attended to.
    """
    raise NotImplementedError("attention function is not implemented yet.")
