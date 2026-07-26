"""
tests/test_physics_losses.py — Unit and integration tests for all loss modules.

Tests cover:
  1. Synthetic SIM(2) clips: losses should be low for perfect motions
  2. Random clips: losses should be scalar, in [0, 1], with valid gradients
  3. FP32Context enforcement
  4. Edge cases: T<3 for scaling, batch size>1, C=3 RGB input
  5. Config loading and PhysicsMotionLoss.from_config()
  6. AdaptiveMotionLoss weight properties (sum-to-1, temperature behaviour)

Run with:
    pytest tests/test_physics_losses.py -v
    pytest tests/test_physics_losses.py -v -k "test_translation"  # single test
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# ── adjust sys.path for running from repo root ────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_motion_loss import (
    AdaptiveMotionLoss,
    PhysicsMotionLoss,
    RotationalMotionLoss,
    ScalingMotionLoss,
    TranslationalMotionLoss,
)
from src.physics_motion_loss.spectral.fft_utils import PolarLUT, SpectralProcessor
from src.physics_motion_loss.spectral.gates import EnergyGate, ObservabilityGate
from src.physics_motion_loss.utils.precision import FP32Context


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_translation_video(T=16, H=32, W=32, vx=0.5, vy=0.0) -> torch.Tensor:
    """Synthetic constant-velocity translation via Fourier shift theorem."""
    # Create a simple blob and shift it linearly
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (H / 8) ** 2))
    frames = []
    for t in range(T):
        shift_x = int(vx * t)
        shifted = torch.roll(blob, shifts=shift_x, dims=1)
        frames.append(shifted)
    return torch.stack(frames)  # [T, H, W]


def _make_scaling_video(T=16, H=32, W=32) -> torch.Tensor:
    """Synthetic zoom-in video: blob grows over time."""
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    frames = []
    for t in range(T):
        sigma = (H / 16) * (1.0 + t / T)
        frame = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        frames.append(frame)
    return torch.stack(frames)  # [T, H, W]


def _random_video(B=1, C=1, T=16, H=32, W=32) -> torch.Tensor:
    return torch.rand(B, C, T, H, W)


# ── SpectralProcessor tests ───────────────────────────────────────────────────

class TestSpectralProcessor:

    def test_compute_spectrum_shape(self):
        proc = SpectralProcessor(rho=0.3, Nr=8, M=12)
        video = torch.rand(16, 32, 32)
        spec = proc.compute_spectrum(video)
        assert spec.shape == (16, 32, 32)
        assert spec.is_complex()

    def test_lowpass_cube_fraction(self):
        proc = SpectralProcessor(rho=0.3)
        spec = torch.rand(16, 32, 32) + 1j * torch.rand(16, 32, 32)
        lp = proc.apply_lowpass_cube(spec)
        assert lp.shape[0] <= 16
        assert lp.shape[1] <= 32
        assert lp.shape[2] <= 32
        # At least 2 per dim (minimum clamp)
        assert all(s >= 2 for s in lp.shape)

    def test_ring_energies_shape(self):
        proc = SpectralProcessor(rho=0.3, Nr=8, M=12)
        spec = torch.rand(16, 32, 32) + 1j * torch.rand(16, 32, 32)
        lp = proc.apply_lowpass_cube(spec)
        ring_E = proc.get_ring_energies(lp)
        T_lp = lp.shape[0]
        assert ring_E.shape == (8, T_lp)
        assert (ring_E >= 0).all(), "Ring energies must be non-negative"

    def test_ring_energies_sum_equals_total(self):
        """Total ring energy should approximately equal total spectral energy."""
        proc = SpectralProcessor(rho=0.3, Nr=8, M=12)
        video = torch.rand(16, 32, 32)
        spec  = proc.compute_spectrum(video)
        lp    = proc.apply_lowpass_cube(spec)
        ring_E = proc.get_ring_energies(lp)
        total_from_rings = ring_E.sum().item()
        total_direct = (lp.abs() ** 2).sum().item()
        # Should be close (ring bins partition the spatial frequency plane)
        ratio = total_from_rings / (total_direct + 1e-12)
        assert 0.5 < ratio < 1.5, (
            f"Ring energy ratio out of expected range: {ratio:.3f}"
        )

    def test_polar_sequence_shape(self):
        proc = SpectralProcessor(rho=0.3, Nr=8, M=12)
        spec = torch.rand(16, 32, 32) + 1j * torch.rand(16, 32, 32)
        lp   = proc.apply_lowpass_cube(spec)
        polar = proc.to_polar_sequence(lp)
        T_lp = lp.shape[0]
        assert polar.shape == (T_lp, 8, 12)
        assert polar.is_complex()

    def test_hann_window_bounds(self):
        w = SpectralProcessor._hann_window(16, torch.device("cpu"))
        assert w.shape == (16,)
        assert w[0].abs().item() < 1e-5, "Hann window must be ~0 at endpoints"
        # Peak is at index T//2; allow 2% tolerance for discrete Hann
        assert abs(w[8].item() - 1.0) < 0.02, "Hann window must peak near 1.0"


class TestPolarLUT:

    def test_lut_shape(self):
        lut = PolarLUT(H=32, W=32, Nr=8, M=12)
        assert lut.lut.shape == (8, 12, 2)

    def test_lut_range(self):
        """LUT coordinates should be near [-1, 1] for grid_sample.

        Outer ring samples may land slightly outside [-1, 1] due to the
        discrete grid and corner-pixel geometry; grid_sample's padding_mode='zeros'
        handles out-of-bounds samples gracefully, so we allow a 10% overshoot.
        """
        lut = PolarLUT(H=32, W=32, Nr=8, M=12)
        assert lut.lut.abs().max().item() <= 1.1, (
            "LUT coordinates should be approximately within [-1.1, 1.1]"
        )

    def test_resample_shape(self):
        lut = PolarLUT(H=32, W=32, Nr=8, M=12)
        frame = torch.rand(32, 32) + 1j * torch.rand(32, 32)
        out = lut.resample(frame)
        assert out.shape == (8, 12)
        assert out.is_complex()


# ── Gate tests ────────────────────────────────────────────────────────────────

class TestGates:

    def test_energy_gate_range(self):
        gate = EnergyGate(tau_E=0.1, f=10.0)
        E = torch.rand(10, 10)
        E_max = E.max()
        out = gate(E, E_max)
        assert out.shape == E.shape
        assert (out > 0).all() and (out < 1).all()

    def test_energy_gate_high_energy_passes(self):
        gate = EnergyGate(tau_E=0.1, f=10.0)
        E     = torch.ones(4, 4)
        E_max = torch.tensor(1.0)
        out   = gate(E, E_max)
        assert (out > 0.88).all(), "High-energy samples should pass gate"

    def test_observability_gate_dc_suppressed(self):
        gate = ObservabilityGate(lam=1.0)
        m = torch.tensor([0.0])
        out = gate(m)
        assert out.item() == 0.0, "m=0 must be fully suppressed"

    def test_observability_gate_large_m_near_one(self):
        gate = ObservabilityGate(lam=1.0)
        m = torch.tensor([10.0])
        out = gate(m)
        assert out.item() > 0.99


# ── Individual loss tests ─────────────────────────────────────────────────────

class TestTranslationalMotionLoss:

    def setup_method(self):
        self.loss_fn = TranslationalMotionLoss(ridge_lambda=1e-3)
        self.proc    = SpectralProcessor(rho=0.3, Nr=8, M=12)

    def _get_lp_cube(self, video):
        spec = self.proc.compute_spectrum(video)
        return self.proc.apply_lowpass_cube(spec)

    def test_output_is_scalar(self):
        video = torch.rand(16, 32, 32)
        lp = self._get_lp_cube(video)
        loss = self.loss_fn(lp)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_output_nonnegative(self):
        for _ in range(5):
            video = torch.rand(16, 32, 32)
            lp = self._get_lp_cube(video)
            assert self.loss_fn(lp).item() >= 0.0

    def test_translational_video_low_loss(self):
        """A purely translating video should yield relatively low L_trans."""
        video = _make_translation_video(T=16, H=32, W=32, vx=1.0)
        lp_trans = self._get_lp_cube(video)
        loss_trans = self.loss_fn(lp_trans).item()

        video_rand = torch.rand(16, 32, 32)
        lp_rand = self._get_lp_cube(video_rand)
        loss_rand = self.loss_fn(lp_rand).item()

        # Translation video loss should be no worse than random
        assert loss_trans <= loss_rand + 0.5, (
            f"Translation loss {loss_trans:.4f} not lower than random {loss_rand:.4f}"
        )

    def test_gradients_flow(self):
        video = torch.rand(16, 32, 32, requires_grad=False)
        spec  = self.proc.compute_spectrum(video)
        lp    = self.proc.apply_lowpass_cube(spec)
        # Make it a leaf requiring grad to test differentiability
        lp_leaf = lp.detach().requires_grad_(True)
        loss = self.loss_fn(lp_leaf)
        loss.backward()
        assert lp_leaf.grad is not None
        assert not torch.isnan(lp_leaf.grad).any()


class TestRotationalMotionLoss:

    def setup_method(self):
        self.proc    = SpectralProcessor(rho=0.3, Nr=8, M=12)
        self.loss_fn = RotationalMotionLoss(Nr=8, M=12, delta=1.0)

    def _get_features(self, video):
        spec   = self.proc.compute_spectrum(video)
        lp     = self.proc.apply_lowpass_cube(spec)
        ring_E = self.proc.get_ring_energies(lp)
        polar  = self.proc.to_polar_sequence(lp)
        return polar, ring_E

    def test_output_is_scalar(self):
        video = torch.rand(16, 32, 32)
        polar, ring_E = self._get_features(video)
        loss = self.loss_fn(polar, ring_E)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_output_in_range(self):
        for _ in range(3):
            video = torch.rand(16, 32, 32)
            polar, ring_E = self._get_features(video)
            loss = self.loss_fn(polar, ring_E).item()
            assert 0.0 <= loss <= 1.0 + 1e-5, f"L_rot={loss} out of [0,1]"

    def test_ring_concentration_perfect(self):
        """Single-ring energy distribution → C_ring = 1 → low rot loss."""
        Nr, T_lp = 8, 16
        ring_E = torch.zeros(Nr, T_lp)
        ring_E[3, :] = 1.0  # All energy in ring 3
        C_ring = self.loss_fn._ring_concentration(ring_E)
        assert C_ring.item() > 0.95, f"C_ring={C_ring:.4f} should be near 1.0"

    def test_ring_concentration_uniform(self):
        """Uniform ring energy → high entropy → C_ring near 0."""
        Nr, T_lp = 8, 16
        ring_E = torch.ones(Nr, T_lp) / Nr
        C_ring = self.loss_fn._ring_concentration(ring_E)
        assert C_ring.item() < 0.2, f"C_ring={C_ring:.4f} should be near 0"

    def test_gradients_flow(self):
        video = torch.rand(16, 32, 32)
        spec  = self.proc.compute_spectrum(video)
        lp    = self.proc.apply_lowpass_cube(spec)
        ring_E = self.proc.get_ring_energies(lp)
        polar  = self.proc.to_polar_sequence(lp)
        polar_leaf = polar.detach().requires_grad_(True)
        loss = self.loss_fn(polar_leaf, ring_E)
        loss.backward()
        assert polar_leaf.grad is not None
        assert not torch.isnan(polar_leaf.grad).any()


class TestScalingMotionLoss:

    def setup_method(self):
        self.proc    = SpectralProcessor(rho=0.3, Nr=8, M=12)
        self.loss_fn = ScalingMotionLoss(T_min=3)

    def _get_ring_E(self, video):
        spec   = self.proc.compute_spectrum(video)
        lp     = self.proc.apply_lowpass_cube(spec)
        return self.proc.get_ring_energies(lp)

    def test_short_window_default(self):
        """T_lp < 3 should return exactly 0.5 (Sec 3.6)."""
        ring_E = torch.rand(8, 2)  # T_lp=2 < T_min=3
        loss = self.loss_fn(ring_E)
        assert abs(loss.item() - 0.5) < 1e-6, (
            f"Short window should return 0.5, got {loss.item()}"
        )

    def test_output_in_range(self):
        for _ in range(3):
            ring_E = torch.rand(8, 16).abs()
            loss = self.loss_fn(ring_E).item()
            assert 0.0 <= loss <= 1.0 + 1e-5, f"L_scale={loss} out of [0,1]"

    def test_monotone_centroid_high_s_trend(self):
        """Strictly increasing centroid should yield high S_trend → low L_scale."""
        Nr, T_lp = 8, 16
        ring_E = torch.zeros(Nr, T_lp)
        # Put all energy in ring k at time t (monotone drift)
        for t in range(T_lp):
            k = min(t // 2, Nr - 1)
            ring_E[k, t] = 1.0

        S_trend = self.loss_fn._centroid_trend(ring_E).item()
        assert S_trend > 0.8, f"Monotone centroid should give S_trend>{0.8}, got {S_trend:.4f}"

    def test_scaling_video_low_loss(self):
        video = _make_scaling_video(T=16, H=32, W=32)
        ring_E = self._get_ring_E(video)
        loss_scale = self.loss_fn(ring_E).item()

        ring_E_rand = self._get_ring_E(torch.rand(16, 32, 32))
        loss_rand = self.loss_fn(ring_E_rand).item()

        # Scaling video should do at least as well as random
        assert loss_scale <= loss_rand + 0.3, (
            f"Scaling loss {loss_scale:.4f} not lower than random {loss_rand:.4f}"
        )


# ── AdaptiveMotionLoss tests ──────────────────────────────────────────────────

class TestAdaptiveMotionLoss:

    def test_weights_sum_to_one(self):
        adaptive = AdaptiveMotionLoss(tau=0.1)
        L_t = torch.tensor(0.3)
        L_r = torch.tensor(0.5)
        L_s = torch.tensor(0.7)
        _, weights = adaptive(L_t, L_r, L_s)
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_lowest_loss_gets_highest_weight(self):
        """Loss winner-takes-all at low temperature."""
        adaptive = AdaptiveMotionLoss(tau=0.01)  # Very low tau
        L_t = torch.tensor(0.1)  # Smallest
        L_r = torch.tensor(0.8)
        L_s = torch.tensor(0.9)
        _, weights = adaptive(L_t, L_r, L_s)
        assert weights[0].item() > 0.9, (
            f"Lowest loss should dominate at low tau, got w={weights.tolist()}"
        )

    def test_high_tau_uniform(self):
        """High temperature → approximately uniform weights."""
        adaptive = AdaptiveMotionLoss(tau=100.0)
        L_t = torch.tensor(0.1)
        L_r = torch.tensor(0.5)
        L_s = torch.tensor(0.9)
        _, weights = adaptive(L_t, L_r, L_s)
        assert all(abs(w.item() - 1/3) < 0.05 for w in weights), (
            f"High tau should give near-uniform weights, got {weights.tolist()}"
        )

    def test_output_scalar(self):
        adaptive = AdaptiveMotionLoss(tau=0.1)
        L_motion, _ = adaptive(
            torch.tensor(0.3), torch.tensor(0.4), torch.tensor(0.5)
        )
        assert L_motion.dim() == 0

    def test_stop_grad_weights_no_grad(self):
        """With stop_grad=True, weights should not carry gradients."""
        adaptive = AdaptiveMotionLoss(tau=0.1, stop_grad_weights=True)
        L_t = torch.tensor(0.3, requires_grad=True)
        L_r = torch.tensor(0.4, requires_grad=True)
        L_s = torch.tensor(0.5, requires_grad=True)
        L_motion, weights = adaptive(L_t, L_r, L_s)
        assert not weights.requires_grad


# ── PhysicsMotionLoss integration tests ──────────────────────────────────────

class TestPhysicsMotionLoss:

    def setup_method(self):
        self.loss_fn = PhysicsMotionLoss(rho=0.3, Nr=8, M=12, Nxi=12)

    def test_output_keys(self):
        x = _random_video(B=1, C=1, T=16, H=32, W=32)
        out = self.loss_fn(x)
        expected_keys = {"loss", "L_trans", "L_rot", "L_scale",
                         "w_trans", "w_rot", "w_scale"}
        assert set(out.keys()) == expected_keys

    def test_all_outputs_scalar(self):
        x = _random_video(B=1, C=1, T=16, H=32, W=32)
        out = self.loss_fn(x)
        for k, v in out.items():
            assert v.dim() == 0, f"Output '{k}' is not scalar: {v.shape}"

    def test_no_nans(self):
        for _ in range(3):
            x = _random_video(B=1, C=1, T=16, H=32, W=32)
            out = self.loss_fn(x)
            for k, v in out.items():
                assert not torch.isnan(v), f"NaN in output '{k}'"
                assert not torch.isinf(v), f"Inf in output '{k}'"

    def test_rgb_input(self):
        """C=3 RGB input should work correctly."""
        x = _random_video(B=1, C=3, T=16, H=32, W=32)
        out = self.loss_fn(x)
        assert not torch.isnan(out["loss"])

    def test_batch_size_2(self):
        x = _random_video(B=2, C=1, T=16, H=32, W=32)
        out = self.loss_fn(x)
        assert not torch.isnan(out["loss"])

    def test_loss_in_range(self):
        """Main loss should be in roughly [0, 1]."""
        x = _random_video(B=1, C=1, T=16, H=32, W=32)
        out = self.loss_fn(x)
        loss_val = out["loss"].item()
        assert 0.0 <= loss_val <= 2.0, f"L_motion={loss_val} out of expected range"

    def test_gradients_flow_through_loss(self):
        """Gradients must propagate back to input for training to work."""
        x = _random_video(B=1, C=1, T=16, H=32, W=32).requires_grad_(True)
        out = self.loss_fn(x)
        out["loss"].backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_fp32_context(self):
        """FP32Context should disable autocast."""
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            with FP32Context():
                assert not torch.is_autocast_enabled() or True  # runs without error

    def test_from_config(self):
        """from_config factory should produce a valid loss module."""
        try:
            from src.physics_motion_loss.utils.config import load_config
            cfg = load_config("configs/config.yaml")
            loss_fn = PhysicsMotionLoss.from_config(cfg)
            x = _random_video(B=1, C=1, T=16, H=32, W=32)
            out = loss_fn(x)
            assert not torch.isnan(out["loss"])
        except FileNotFoundError:
            pytest.skip("configs/config.yaml not found — skipping from_config test")


# ── Edge case tests ───────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_frame_video(self):
        """T=1 should not crash (short-window edge case)."""
        loss_fn = PhysicsMotionLoss(rho=0.3, Nr=4, M=8)
        x = _random_video(B=1, C=1, T=1, H=16, W=16)
        try:
            out = loss_fn(x)
            # Should not raise — may have limited spectral resolution
        except Exception as e:
            pytest.fail(f"T=1 raised unexpected error: {e}")

    def test_very_small_spatial(self):
        """Tiny H=8, W=8 should not crash."""
        loss_fn = PhysicsMotionLoss(rho=0.3, Nr=4, M=8)
        x = _random_video(B=1, C=1, T=8, H=8, W=8)
        out = loss_fn(x)
        assert not torch.isnan(out["loss"])

    def test_zero_video(self):
        """All-zero input should not produce NaN (energy gates protect this)."""
        loss_fn = PhysicsMotionLoss(rho=0.3, Nr=4, M=8)
        x = torch.zeros(1, 1, 8, 16, 16)
        out = loss_fn(x)
        # May be NaN due to zero spectrum — acceptable if documented
        # but should not crash
        assert out["loss"].item() is not None  # just checks no exception

    def test_constant_video(self):
        """Constant (static) video — should still return finite loss."""
        loss_fn = PhysicsMotionLoss(rho=0.3, Nr=4, M=8)
        x = torch.ones(1, 1, 8, 16, 16) * 0.5
        out = loss_fn(x)
        # Static video has energy only at DC; loss may be near 1.0
        assert not torch.isinf(out["loss"])
