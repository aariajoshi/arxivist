"""Two-stage VAE for DiscDiff (Sec 4.1.1, CNN-VAE variant).

s in N_4^L  --E_phi1-->  z1 in R^{K x M}  --E_phi2-->  z in R^{C x K' x M'}
z          --D_theta2-->  z1~          --D_theta1-->  s~ (4-way categorical per position)

CE reconstruction loss + beta * KL to an isotropic Gaussian (Sec 4.1.2). This is the
CNN-VAE (best generation quality per Table 4). Kept small and configurable so a full
forward/backward runs on CPU for verification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1Encoder(nn.Module):
    """s (one-hot [B,4,L]) -> 2D latent z1 [B,K,M] via 1D convs with downsampling."""

    def __init__(self, k_channels: int = 32, downsample: int = 4) -> None:
        super().__init__()
        layers, ch = [], 4
        n_down = downsample.bit_length() - 1  # e.g. 4 -> 2 halvings
        for _ in range(n_down):
            layers += [nn.Conv1d(ch, k_channels, 3, stride=2, padding=1), nn.GELU()]
            ch = k_channels
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)  # [B, K, M]


class Stage1Decoder(nn.Module):
    """z1 [B,K,M] -> logits [B,4,L] via transposed convs (symmetric)."""

    def __init__(self, k_channels: int = 32, upsample: int = 4) -> None:
        super().__init__()
        layers = []
        n_up = upsample.bit_length() - 1
        for idx in range(n_up):
            out_ch = k_channels if idx < n_up - 1 else 4
            layers += [nn.ConvTranspose1d(k_channels, out_ch, 4, stride=2, padding=1)]
            if idx < n_up - 1:
                layers += [nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        return self.net(z1)  # [B, 4, L] logits


class Stage2Encoder(nn.Module):
    """z1 [B,K,M] -> 3D latent (mu, logvar) each [B,C,K',M']. Adds a dummy dim then
    2D-convs down. Here represented compactly as a channel/length reduction."""

    def __init__(self, k_channels: int = 32, c_channels: int = 8) -> None:
        super().__init__()
        self.to_mu = nn.Conv1d(k_channels, c_channels, 3, stride=2, padding=1)
        self.to_logvar = nn.Conv1d(k_channels, c_channels, 3, stride=2, padding=1)

    def forward(self, z1: torch.Tensor):
        return self.to_mu(z1), self.to_logvar(z1)  # each [B, C, M']


class Stage2Decoder(nn.Module):
    """z [B,C,M'] -> z1~ [B,K,M] (symmetric to Stage2Encoder)."""

    def __init__(self, k_channels: int = 32, c_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.ConvTranspose1d(c_channels, k_channels, 4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DiscDiffVAE(nn.Module):
    def __init__(self, k_channels: int = 32, c_channels: int = 8,
                 seq_len: int = 256, beta: float = 1e-4) -> None:
        super().__init__()
        self.enc1 = Stage1Encoder(k_channels)
        self.enc2 = Stage2Encoder(k_channels, c_channels)
        self.dec2 = Stage2Decoder(k_channels, c_channels)
        self.dec1 = Stage1Decoder(k_channels)
        self.beta = beta
        self.seq_len = seq_len

    def encode(self, s_onehot: torch.Tensor):
        z1 = self.enc1(s_onehot)
        mu, logvar = self.enc2(z1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z1 = self.dec2(z)
        logits = self.dec1(z1)
        # pad/crop to exact seq_len
        if logits.shape[-1] != self.seq_len:
            logits = F.interpolate(logits, size=self.seq_len, mode="linear", align_corners=False)
        return logits  # [B, 4, L]

    def forward(self, s_idx: torch.Tensor):
        """s_idx: [B, L] integer tokens in {0..3}. Returns (logits, loss, recon_acc)."""
        s_onehot = F.one_hot(s_idx, num_classes=4).permute(0, 2, 1).float()  # [B,4,L]
        mu, logvar = self.encode(s_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)                                             # [B,4,L]
        recon = F.cross_entropy(logits, s_idx)                              # CE recon loss
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())       # KL to N(0,I)
        loss = recon + self.beta * kl
        acc = (logits.argmax(1) == s_idx).float().mean()
        return logits, loss, acc
