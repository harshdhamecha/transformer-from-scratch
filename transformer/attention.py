"""Scaled dot-product attention (M1) and multi-head attention (M2)."""


import torch
import math
import torch.nn as nn



def attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention.
    Input shape - q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v).
    Output shape - (..., T_q, d_v).
    Mask convention: True indicates positions that should be attended to.
    """
    d_k = query.shape[-1]
    score = torch.matmul(query, torch.transpose(key, -1, -2))
    scaled_score = score / math.sqrt(d_k)
    if mask is not None:
        fill_val = torch.finfo(scaled_score.dtype).min
        masked_score = torch.where(mask, scaled_score, fill_val)
        scaled_score = masked_score
    weighted_score = torch.softmax(scaled_score, dim=-1)
    output = torch.matmul(weighted_score, value)
    return output, weighted_score    



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.0):
        '''
        MultiHeadAttention Block.\n
        Inputs: 
        d_model - Embedding Model Dimension (int),
        h - Number of Heads (int),
        dropout - Dropout Probability (float)\n
        '''
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h

    def split_heads(self, tensor):
        B, T = tensor.shape[:2]
        reshaped_tensor = torch.reshape(tensor, (B, T, self.h, self.d_k))
        transposed_tensor = torch.transpose(reshaped_tensor, 1, 2)
        return transposed_tensor

    def merge_heads(self, tensor):
        B, T_q = tensor.shape[0], tensor.shape[2]
        transposed_tensor = torch.transpose(tensor, 1, 2)
        return torch.reshape(transposed_tensor, (B, T_q, self.d_k * self.h))

    def forward(self, query, key, value, mask=None):
        '''
        Apply forward for MultiHeadAttention\n
        Inputs:
        query - A tensor of shape (B, T_q, d_model),
        key - A tensor of shape (B, T_k, d_model),
        value - A tensor of shape (B, T_k, d_model),
        mask - A tensor of shape (B, T_q, T_k), (B, 1, T_k) or (T_q, T_k) \n
        Outputs:
        output - MultiHeadAttention Output,
        weights - MultiHeadAttention Weights\n
        Mask convention: True indicates positions that should be attended to.
        '''

        Q = self.w_q(query)   # (B, T_q, d_model)
        K = self.w_k(key)     # (B, T_k, d_model)
        V = self.w_v(value)   # (B, T_k, d_model)
        Q_ = self.split_heads(Q)    # (B, H, T_q, d_k)
        K_ = self.split_heads(K)    # (B, H, T_k, d_k)  
        V_ = self.split_heads(V)    # (B, H, T_k, d_k)
        if mask is not None:
            if mask.dim() == 3:
                # (B, T_q, T_k) or (B, 1, T_k) --> (B, 1, 1/T_q, T_k)
                mask = torch.unsqueeze(mask, 1)
        self_attn_output, weights = attention(Q_, K_, V_, mask=mask)
        self_attn_merged_output = self.merge_heads(self_attn_output)
        output = self.w_o(self_attn_merged_output)
        return output, weights