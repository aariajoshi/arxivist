"""Latent U-Net denoiser for DiscDiff (Sec 4.1.3, Fig 3 Step 2).

eps_theta(z_t, t, species) predicts the added noise in the VAE latent space
(Eq 2: E ||eps - eps_theta(z_t, t)||^2). ResNet blocks + optional self-attention +
cross-attention on (species, time). Kept minimal/configurable for CPU forward passes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ResBlock1d(nn.Module):
    def __init__(self, ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, ch)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class LatentUNet(nn.Module):
    def __init__(self, ch: int = 8, emb_dim: int = 64, n_species: int = 15,
                 use_self_attention: bool = False) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.in_conv = nn.Conv1d(ch, ch, 3, padding=1)
        self.block1 = ResBlock1d(ch, emb_dim)
        self.attn = nn.MultiheadAttention(ch, num_heads=1, batch_first=True) \
            if use_self_attention else None
        self.block2 = ResBlock1d(ch, emb_dim)
        self.out_conv = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                species: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        if species is not None:
            emb = emb + self.species_emb(species)          # cross-condition (species,time)
        h = self.in_conv(z_t)
        h = self.block1(h, emb)
        if self.attn is not None:
            a = h.transpose(1, 2)
            a, _ = self.attn(a, a, a)
            h = h + a.transpose(1, 2)
        h = self.block2(h, emb)
        return self.out_conv(h)                            # predicted eps, shape of z_t
