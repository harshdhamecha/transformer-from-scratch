"""Causal mask, padding mask, and combining them."""


import torch


def causal_mask(T, device=None):
    """
    Returns a causal mask of shape (T, T) where the upper triangular part is masked (set to False).
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def padding_mask(tokens, pad_id):
    """
    Takes token ID of shape (B, T) and the ID the vocabulary assigns to <pad>.
    Returns a boolean tensor (B, 1, T), True where the token is not padding.
    """
    return torch.unsqueeze(torch.where(tokens == pad_id, False, True), dim=1)