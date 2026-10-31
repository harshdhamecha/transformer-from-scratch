"""Encoder layer and encoder stack (M4)."""

import torch.nn as nn
from transformer.layers import Embeddings, PositionalEncoding, FeedForward, LayerNorm
from transformer.attention import MultiHeadAttention



class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout, eps=1e-5):
        super().__init__()

    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
