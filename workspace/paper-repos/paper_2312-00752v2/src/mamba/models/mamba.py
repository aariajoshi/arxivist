"""
Primary Mamba language model definition.
Implements MambaLMHeadModel.
"""
import torch
import torch.nn as nn
from typing import Optional
from .blocks import MambaBlock

class MambaLayer(nn.Module):
    """
    A single layer of Mamba combining normalization, the MambaBlock, and residual connection.
    """
    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, expand, d_state, d_conv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.mamba(self.norm(x)) + x

class MambaLMHeadModel(nn.Module):
    """
    Primary Mamba language model.
    Repeats Mamba blocks with normalization and residual connections.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            MambaLayer(d_model, expand, d_state, d_conv)
            for _ in range(n_layer)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mamba LM.
        Args:
            input_ids: Input tokens of shape [B, L]
        Returns:
            Logits of shape [B, L, vocab_size]
        """
        assert input_ids.dim() == 2, f"Expected [B,L], got {input_ids.shape}"
        
        hidden_states = self.embedding(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
        
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Generate tokens sequence using the model.
        Args:
            input_ids: Initial tokens of shape [B, L]
            max_length: Maximum length to generate
        Returns:
            Generated tokens of shape [B, max_length]
        """
        self.eval()
        for _ in range(max_length - input_ids.shape[1]):
            logits = self.forward(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
