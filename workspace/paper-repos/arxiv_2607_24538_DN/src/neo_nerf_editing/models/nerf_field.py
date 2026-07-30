"""
models/nerf_field.py

Implements F_Theta, the base joint scene+language NeRF from Sec. II, Eq. 1:
    (S, L) = F_Theta(I, Omega)
S is a scene field in R^4 (density + RGB), L is a language field in R^{d_l}.

SIR note: the paper never specifies the concrete backbone (MLP width/depth, positional
encoding vs hash-grid). SIR architecture confidence for this specific detail is ~0.5-0.72.
We use a plain positional-encoding MLP (the simplest faithful choice per the paper's own
citation of Mildenhall et al. [1] for the base representation) and expose width/depth via
config so a hash-grid backbone can be substituted later without touching call sites.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard NeRF positional encoding (Mildenhall et al. 2020), used because the paper
    does not restate its own encoding scheme (assumed inherited from base NeRF, SIR conf. 0.7)."""

    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)

    def output_dim(self, input_dim: int) -> int:
        return input_dim * (2 * self.num_freqs + (1 if self.include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D] -> [..., D*(2*num_freqs [+1 if include_input])]"""
        assert x.shape[-1] > 0, f"Expected non-empty last dim, got {x.shape}"
        out = [x] if self.include_input else []
        freq_bands = self.freq_bands.to(x.device)
        for freq in freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)

    def __repr__(self) -> str:
        return f"PositionalEncoding(num_freqs={self.num_freqs}, include_input={self.include_input})"


class JointNeRFField(nn.Module):
    """The base joint scene+language NeRF F_Theta (Eq. 1).

    Args:
        d_l: language feature dimensionality.
        hidden_dim: MLP hidden width.  # ASSUMED (SIR training/architecture confidence ~0.5)
        num_layers: MLP depth, including a skip connection.  # ASSUMED
        posenc_num_freqs: positional encoding frequency bands for 3D points.
        skip_layers: indices (0-based, before which layer) to concatenate the encoded input again.
    """

    def __init__(
        self,
        d_l: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 6,
        posenc_num_freqs: int = 10,
        posenc_include_input: bool = True,
        skip_layers: List[int] = (3,),
    ):
        super().__init__()
        self.d_l = d_l
        self.point_encoder = PositionalEncoding(posenc_num_freqs, posenc_include_input)
        self.dir_encoder = PositionalEncoding(4, posenc_include_input)  # fewer freqs for view dirs, standard practice
        pt_dim = self.point_encoder.output_dim(3)
        dir_dim = self.dir_encoder.output_dim(3)
        self.skip_layers = set(skip_layers)

        layers = []
        in_dim = pt_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            if i in self.skip_layers:
                in_dim = in_dim + pt_dim
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.trunk = nn.ModuleList(layers)

        self.density_head = nn.Linear(hidden_dim, 1)
        self.lang_head = nn.Linear(hidden_dim, d_l)  # language field L, queried independent of view dir
        self.rgb_trunk = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.rgb_head = nn.Linear(hidden_dim // 2, 3)

    def forward(self, points: torch.Tensor, view_dirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: [..., 3] world-space query points.
            view_dirs: [..., 3] unit view directions (broadcastable to points' leading dims).
        Returns:
            dict with 'density' [...,1], 'rgb' [...,3], 'lang_feat' [...,d_l]
        """
        assert points.shape[-1] == 3, f"Expected points [...,3], got {points.shape}"
        assert view_dirs.shape[-1] == 3, f"Expected view_dirs [...,3], got {view_dirs.shape}"

        pt_enc = self.point_encoder(points)
        h = pt_enc
        for i, layer in enumerate(self.trunk):
            if i in self.skip_layers:
                h = torch.cat([h, pt_enc], dim=-1)
            h = torch.relu(layer(h))

        density = torch.relu(self.density_head(h))  # density must be non-negative
        lang_feat = self.lang_head(h)  # language field decoded from geometry features, view-independent

        dir_enc = self.dir_encoder(view_dirs.expand_as(points))
        rgb_in = torch.cat([h, dir_enc], dim=-1)
        rgb_h = torch.relu(self.rgb_trunk(rgb_in))
        rgb = torch.sigmoid(self.rgb_head(rgb_h))

        return {"density": density, "rgb": rgb, "lang_feat": lang_feat}

    def query_density_only(self, points: torch.Tensor) -> torch.Tensor:
        """Density-only query, used by object-removal resampling (Sec. II-A) which needs
        density along a ray before committing to a full color/language pass."""
        dummy_dir = torch.zeros_like(points)
        dummy_dir[..., 2] = 1.0
        return self.forward(points, dummy_dir)["density"]

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"JointNeRFField(d_l={self.d_l}, n_params={n_params})"
