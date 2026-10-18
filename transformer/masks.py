"""Causal mask, padding mask, and combining them."""


import torch


def causal_mask(T, device=None):
    """
    Returns a causal mask of shape (T, T) where the upper triangular part is masked (set to False).
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

