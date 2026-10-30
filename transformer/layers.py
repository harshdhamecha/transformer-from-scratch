"""LayerNorm, position-wise feed-forward, sinusoidal positional encoding, token embeddings (M3)."""

import torch
import torch.nn as nn
import math




class LayerNorm(nn.Module):
    def __init__(self, d_model, eps):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        computed_mean = x.mean(dim=-1, keepdim=True)
        computed_var = x.var(dim=-1, keepdim=True, correction=0)
        result = ((x - computed_mean) / torch.sqrt(computed_var + self.eps) * self.gamma) + self.beta
        return result


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l2(self.dropout(self.relu(self.l1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()

    def forward(self, x):
        pass


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)