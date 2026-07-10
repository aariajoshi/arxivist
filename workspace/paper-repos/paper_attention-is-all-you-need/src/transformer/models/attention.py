"""
Attention mechanisms.
Paper Sections: 3.2.1, 3.2.2
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    Paper Reference: Section 3.2.1, Equation 1
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # query, key, value shape: [B, h, T, d_k]
        assert query.dim() == 4, f"Expected [B,h,T,d_k], got {query.shape}"
        
        d_k = query.size(-1)
        
        # QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Masking uses -1e9 instead of -inf to avoid NaN gradients (Ambiguity mitigation)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        p_attn = torch.softmax(scores, dim=-1)
        
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
            
        # softmax(QK^T / sqrt(d_k))V
        return torch.matmul(p_attn, value), p_attn

    def __repr__(self) -> str:
        return "ScaledDotProductAttention()"

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    Paper Reference: Section 3.2.2
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    """
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        
        self.w_q = nn.Linear(d_model, h * d_k)
        self.w_k = nn.Linear(d_model, h * d_k)
        self.w_v = nn.Linear(d_model, h * d_v)
        self.w_o = nn.Linear(h * d_v, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert query.dim() == 3, f"Expected [B,T,D], got {query.shape}"
        
        # mask is already shaped for broadcasting, no need to unsqueeze
            
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.w_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(nbatches, -1, self.h, self.d_v).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_v)
        
        return self.w_o(x)

    def __repr__(self) -> str:
        return f"MultiHeadAttention(h={self.h}, d_model={self.w_q.in_features})"
