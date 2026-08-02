"""
Feed-forward networks and positional encodings.
Paper Sections: 3.3, 3.5
"""

import math
import torch
import torch.nn as nn
from typing import Optional

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Paper Reference: Section 3.3, Equation 2.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected [B,T,D], got {x.shape}"
        # Eq. 2 in Section 3.3
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

    def __repr__(self) -> str:
        return f"PositionwiseFeedForward(d_model={self.w_1.in_features}, d_ff={self.w_1.out_features})"

class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.
    Paper Reference: Section 3.5
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # div_term = 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Eq. in Section 3.5: PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Eq. in Section 3.5: PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected [B,T,D], got {x.shape}"
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x

    def __repr__(self) -> str:
        return "PositionalEncoding()"
