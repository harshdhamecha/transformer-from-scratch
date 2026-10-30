"""LayerNorm, position-wise feed-forward, sinusoidal positional encoding, token embeddings (M3)."""


import torch.nn as nn



class LayerNorm(nn.Module):
    def __init__(self, d_model, eps):
        super().__init__()

    def forward(self):
        pass


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

    def forward(self):
        pass


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dropout):
        super().__init__()

    def forward(self):
        pass


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

    def forward(self):
        pass