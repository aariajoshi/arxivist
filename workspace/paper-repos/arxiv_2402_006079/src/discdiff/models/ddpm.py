"""DDPM noise schedule + latent eps-objective + sampling (Sec 4.1.3, Eq 2).

Standard DDPM in the VAE latent space: forward q(z_t | z_0) adds Gaussian noise; the
U-Net predicts eps; reverse sampling denoises z_T -> z_0, which the FROZEN VAE decoder
then maps to a DNA sequence (Fig 3, 'Locked Decoder').
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu") -> None:
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.device = device

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward: z_t = sqrt(abar_t) z0 + sqrt(1-abar_t) eps."""
        ab = self.alpha_bars[t][:, None, None]
        return torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * noise

    def loss(self, model, z0: torch.Tensor, species: torch.Tensor | None = None) -> torch.Tensor:
        """Eq 2: E_{z,t,eps} || eps - eps_theta(z_t, t) ||^2."""
        b = z0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        eps_hat = model(z_t, t, species)
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def p_sample_step(self, model, z_t: torch.Tensor, t: int,
                      species: torch.Tensor | None = None) -> torch.Tensor:
        """One reverse DDPM step z_t -> z_{t-1}."""
        tt = torch.full((z_t.shape[0],), t, device=z_t.device, dtype=torch.long)
        eps = model(z_t, tt, species)
        alpha, ab = self.alphas[t], self.alpha_bars[t]
        mean = (z_t - (1 - alpha) / torch.sqrt(1 - ab) * eps) / torch.sqrt(alpha)
        if t == 0:
            return mean
        noise = torch.randn_like(z_t)
        return mean + torch.sqrt(self.betas[t]) * noise

    @torch.no_grad()
    def sample(self, model, shape, species: torch.Tensor | None = None) -> torch.Tensor:
        """Full reverse chain z_T ~ N(0,I) -> z_0."""
        z = torch.randn(shape, device=self.device)
        for t in reversed(range(self.n_steps)):
            z = self.p_sample_step(model, z, t, species)
        return z
