"""
spectral/fft_utils.py — 3D FFT computation, low-pass truncation, and polar resampling.

Implements the spectral preprocessing pipeline from arXiv 2506.02244v2:
  - Section 4.1:   low-pass cube truncation (ϱ=0.3, 2.7% of coefficients)
  - Appendix A.1:  separable 3D DFT with optional Hann window
  - Appendix A.6:  polar/log-radius resampling via precomputed bilinear LUT

All operations run in FP32 (autocast disabled upstream by FP32Context).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Polar Look-Up Table
# ---------------------------------------------------------------------------

class PolarLUT(nn.Module):
    """Precomputed bilinear look-up table mapping polar bins → Cartesian spectral grid.

    Paper reference: Appendix A.6 — "We pre-compute a look-up table (LUT) to map
    polar bins (ρk, θℓ) to Cartesian indices (ωx, ωy) and use bilinear interpolation
    on the spectral grid."

    The LUT is computed once at __init__ and cached as a non-parameter buffer.
    This is critical for performance: on-the-fly bilinear interpolation per batch
    would be prohibitive at training throughput.

    Args:
        H: Spatial height of the spectral grid (number of ωy frequencies).
        W: Spatial width of the spectral grid (number of ωx frequencies).
        Nr: Number of concentric rings (linear in ρ). Default 20 (App A.8).
        M:  Number of angular bins. Default 24 (App A.8).
    """

    def __init__(self, H: int, W: int, Nr: int = 20, M: int = 24) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.Nr = Nr
        self.M = M

        # Build and register the LUT as a buffer (not a trainable parameter)
        lut = self._build_lut(H, W, Nr, M)  # [Nr, M, 2]
        self.register_buffer("lut", lut)

    @staticmethod
    def _build_lut(H: int, W: int, Nr: int, M: int) -> Tensor:
        """Build normalised grid for torch.nn.functional.grid_sample.

        Returns:
            Tensor of shape [Nr, M, 2] with (x, y) coordinates in [-1, 1],
            ready for use with grid_sample (align_corners=True).
        """
        # Maximum radius in pixel units: use half the smaller dimension
        rho_max = min(H, W) / 2.0 - 0.5

        # Linear radii from near-0 to rho_max (App A.6: "linear in ρ")
        rho_vals = torch.linspace(0.5, rho_max, Nr)           # [Nr]

        # Angular bins: evenly spaced in [0, 2π)
        theta_vals = torch.linspace(0, 2 * torch.pi, M + 1)[:-1]  # [M], exclude 2π

        # Expand to [Nr, M]
        rho_grid = rho_vals.unsqueeze(1).expand(Nr, M)
        theta_grid = theta_vals.unsqueeze(0).expand(Nr, M)

        # Convert polar → Cartesian pixel coordinates (centre = H/2, W/2)
        cx, cy = W / 2.0, H / 2.0
        x_cart = cx + rho_grid * torch.cos(theta_grid)  # [Nr, M]
        y_cart = cy + rho_grid * torch.sin(theta_grid)  # [Nr, M]

        # Normalise to [-1, 1] for grid_sample (align_corners=True convention)
        x_norm = (x_cart / (W - 1)) * 2.0 - 1.0
        y_norm = (y_cart / (H - 1)) * 2.0 - 1.0

        # Stack: grid_sample expects [H_out, W_out, 2] with (x, y) order
        lut = torch.stack([x_norm, y_norm], dim=-1)  # [Nr, M, 2]
        return lut

    def resample(self, spectrum_frame: Tensor) -> Tensor:
        """Bilinear resample a 2D spatial spectrum to polar coordinates.

        Args:
            spectrum_frame: Complex or real spectrum, shape [H, W].

        Returns:
            Resampled spectrum in polar coordinates, shape [Nr, M].
            Complex input returns complex output; real input returns real.
        """
        assert spectrum_frame.dim() == 2, (
            f"Expected [H, W], got {spectrum_frame.shape}"
        )
        is_complex = spectrum_frame.is_complex()

        if is_complex:
            # Process real and imaginary parts separately through grid_sample
            real_part = self._sample_2d(spectrum_frame.real)
            imag_part = self._sample_2d(spectrum_frame.imag)
            return torch.complex(real_part, imag_part)
        else:
            return self._sample_2d(spectrum_frame)

    def _sample_2d(self, x: Tensor) -> Tensor:
        """grid_sample wrapper for a single [H, W] frame → [Nr, M]."""
        # grid_sample requires [N, C, H, W]; add batch and channel dims
        x_4d = x.unsqueeze(0).unsqueeze(0)                  # [1, 1, H, W]
        grid = self.lut.unsqueeze(0)                         # [1, Nr, M, 2]
        out = F.grid_sample(
            x_4d.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )                                                     # [1, 1, Nr, M]
        return out.squeeze(0).squeeze(0)                     # [Nr, M]

    def __repr__(self) -> str:
        return f"PolarLUT(H={self.H}, W={self.W}, Nr={self.Nr}, M={self.M})"


# ---------------------------------------------------------------------------
# Spectral Processor
# ---------------------------------------------------------------------------

class SpectralProcessor(nn.Module):
    """End-to-end spectral preprocessing: FFT → low-pass → polar/ring features.

    Paper reference:
      - Appendix A.1: separable 3D DFT with Hann window
      - Section 4.1:  low-pass cube truncation (ϱ=0.3)
      - Appendix A.6: polar resampling, ring energies, log-radius sampling

    Args:
        rho:  Low-pass fraction per dimension (default 0.3 → 2.7% coefficients).
        Nr:   Number of concentric rings (default 20).
        M:    Number of angular bins (default 24).
        Nxi:  Number of log-radius bins for scaling branch (default 24).
        window: Temporal window type, currently only 'hann' supported.
    """

    def __init__(
        self,
        rho: float = 0.3,
        Nr: int = 20,
        M: int = 24,
        Nxi: int = 24,
        window: str = "hann",
    ) -> None:
        super().__init__()
        self.rho = rho
        self.Nr = Nr
        self.M = M
        self.Nxi = Nxi
        self.window_type = window

        # PolarLUT is built lazily on first forward() since H, W are not known
        # at __init__ time (they depend on input video resolution).
        self._polar_lut: dict[Tuple[int, int], PolarLUT] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_spectrum(self, video: Tensor) -> Tensor:
        """Compute 3D spatiotemporal FFT with Hann windowing.

        Implements App A.1: "separable 2D spatial DFT per frame followed by a
        1D temporal DFT (with an optional Hann window h[t])".

        Args:
            video: Single-channel video tensor, shape [T, H, W], float32.

        Returns:
            Complex spectrum V̂(ωx, ωy, ωt), shape [T, H, W], complex64.
        """
        assert video.dim() == 3, f"Expected [T, H, W], got {video.shape}"
        T, H, W = video.shape

        # App A.1: apply Hann window along temporal axis before temporal DFT
        h = self._hann_window(T, device=video.device)  # [T]
        video_windowed = video * h.view(T, 1, 1)

        # Separable: 2D spatial DFT per frame, then 1D temporal DFT
        # torch.fft.fft2 on last two dims → spatial spectrum per frame [T, H, W]
        spatial_spec = torch.fft.fft2(video_windowed, dim=(-2, -1))

        # 1D temporal DFT along dim 0 → full 3D spectrum [T, H, W]
        spectrum = torch.fft.fft(spatial_spec, dim=0)
        return spectrum  # complex64

    def apply_lowpass_cube(self, spectrum: Tensor) -> Tensor:
        """Retain only the low-frequency cube (ϱ fraction per dimension).

        Paper reference: Section 4.1 — "We keep a per-dimension low-pass cube
        with fraction ϱ=0.3 along (ωt, ωx, ωy), so only ϱ³=2.7% of spectral
        coefficients are processed."

        Energy retained: ηcube ∈ [0.97, 0.987] under power-law κ≈1.8 (App C).

        Args:
            spectrum: Full spectrum, shape [T, H, W], complex.

        Returns:
            Low-pass cube, shape [T_lp, H_lp, W_lp], complex.
            Where T_lp=⌊ϱ·T⌋, H_lp=⌊ϱ·H⌋, W_lp=⌊ϱ·W⌋ (minimum 2 per dim).
        """
        assert spectrum.dim() == 3, f"Expected [T, H, W], got {spectrum.shape}"
        T, H, W = spectrum.shape

        # Compute low-pass cutoff indices, minimum 2 per dimension
        T_lp = max(2, int(self.rho * T))
        H_lp = max(2, int(self.rho * H))
        W_lp = max(2, int(self.rho * W))

        return spectrum[:T_lp, :H_lp, :W_lp]

    def get_ring_energies(self, lowpass_cube: Tensor) -> Tensor:
        """Compute ring energies Ek(t) for each concentric annulus over time.

        Paper reference: Appendix D.2, Section 3.6.
        Ring energies underpin both the rotation (C_ring entropy) and scaling
        (C_flow, S_trend) loss branches.

        Ek(t) = Σ_{(ωx,ωy) ∈ ring_k} |V̂(ωx,ωy,t)|²

        Args:
            lowpass_cube: Complex spectrum, shape [T_lp, H_lp, W_lp].

        Returns:
            Ring energies, shape [Nr, T_lp], float32.
        """
        assert lowpass_cube.dim() == 3, f"Expected [T_lp, H_lp, W_lp], got {lowpass_cube.shape}"
        T_lp, H_lp, W_lp = lowpass_cube.shape

        energy = lowpass_cube.abs() ** 2  # [T_lp, H_lp, W_lp], real

        # Build concentric ring masks in the spatial frequency plane
        # Coordinates centred at DC (0,0)
        wy = torch.fft.fftfreq(H_lp, device=lowpass_cube.device) * H_lp  # [H_lp]
        wx = torch.fft.fftfreq(W_lp, device=lowpass_cube.device) * W_lp  # [W_lp]
        wy_grid, wx_grid = torch.meshgrid(wy, wx, indexing="ij")          # [H_lp, W_lp]
        rho_map = torch.sqrt(wx_grid ** 2 + wy_grid ** 2)                 # [H_lp, W_lp]

        rho_max = rho_map.max().item()
        ring_edges = torch.linspace(0.0, rho_max, self.Nr + 1, device=lowpass_cube.device)

        ring_energies = torch.zeros(self.Nr, T_lp, device=lowpass_cube.device)
        for k in range(self.Nr):
            mask = (rho_map >= ring_edges[k]) & (rho_map < ring_edges[k + 1])  # [H_lp, W_lp]
            # Sum energy over spatial frequencies within this ring for each time step
            ring_energies[k] = (energy * mask.unsqueeze(0)).sum(dim=(-2, -1))  # [T_lp]

        return ring_energies  # [Nr, T_lp]

    def to_polar_sequence(self, lowpass_cube: Tensor) -> Tensor:
        """Resample spectrum to polar coordinates for each time step.

        Applies PolarLUT.resample() to each temporal slice, producing the
        angular harmonic input C_m(ρ, t) needed by the rotation loss branch.

        Args:
            lowpass_cube: Complex spectrum, shape [T_lp, H_lp, W_lp].

        Returns:
            Polar spectrum, shape [T_lp, Nr, M], complex.
        """
        assert lowpass_cube.dim() == 3
        T_lp, H_lp, W_lp = lowpass_cube.shape

        lut = self._get_polar_lut(H_lp, W_lp, lowpass_cube.device)

        frames = []
        for t in range(T_lp):
            frame_polar = lut.resample(lowpass_cube[t])  # [Nr, M], complex
            frames.append(frame_polar)

        return torch.stack(frames, dim=0)  # [T_lp, Nr, M]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_polar_lut(self, H: int, W: int, device: torch.device) -> PolarLUT:
        """Retrieve or create a cached PolarLUT for the given (H, W)."""
        key = (H, W)
        if key not in self._polar_lut:
            self._polar_lut[key] = PolarLUT(H, W, self.Nr, self.M).to(device)
        else:
            # Move to correct device if needed
            existing = self._polar_lut[key]
            if existing.lut.device != device:
                self._polar_lut[key] = existing.to(device)
        return self._polar_lut[key]

    @staticmethod
    def _hann_window(T: int, device: torch.device) -> Tensor:
        """Generate a Hann window of length T.

        App A.1: "optional Hann window h[t]" applied before temporal DFT
        to reduce spectral leakage.

        h[t] = 0.5 · (1 − cos(2πt / (T−1)))
        """
        if T == 1:
            return torch.ones(1, device=device)
        t = torch.arange(T, device=device, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(2.0 * torch.pi * t / (T - 1)))

    def __repr__(self) -> str:
        return (
            f"SpectralProcessor(rho={self.rho}, Nr={self.Nr}, "
            f"M={self.M}, Nxi={self.Nxi}, window='{self.window_type}')"
        )
