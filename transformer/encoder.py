"""Encoder layer and encoder stack (M4)."""

import torch.nn as nn
from transformer.layers import Embeddings, PositionalEncoding, FeedForward, LayerNorm
from transformer.attention import MultiHeadAttention



class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout, eps=1e-5):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = LayerNorm(d_model, eps)
        self.ln2 = LayerNorm(d_model, eps)
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        output, _ = self.mha(x, x, x, mask)
        x_new = self.ln1(self.d1(output) + x)
        return self.ln2(self.d2(self.ffn(x_new)) + x_new)


class Encoder(nn.Module):
    def __init__(self, N, vocab_size, max_len, d_model, h, d_ff, dropout, eps=1e-5):
        super().__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout, eps) for i in range(N)])

    def forward(self, tokens, mask=None):
        inputs = self.pe(self.embedding(tokens))
        output = inputs
        for i, _ in enumerate(self.encoder_layers):
            output = self.encoder_layers[i](output, mask)
        return output
