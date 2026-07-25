"""Mechanical verification of DiscDiff (CPU, no trained weights).

Core verified claim: **Absorb-Escape** (Sec 4.2, Table 3, Fig 4) — a deterministic
post-processing algorithm over per-position softmax confidences — provably repairs
low-confidence 'valleys' (e.g. TATT -> TATA) and is a no-op on confident sequences.
The DiscDiff LDM (two-stage VAE + latent U-Net + DDPM) is verified structurally
(forward pass, loss finiteness, one denoise step) on tiny CPU configs.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discdiff.evaluation.metrics import (
    delta_diversity,
    diversity,
    motif_correlation,
    reconstruction_accuracy,
    s_fid,
)
from discdiff.models.ddpm import DDPM
from discdiff.models.unet import LatentUNet
from discdiff.models.vae import DiscDiffVAE
from discdiff.refine.absorb_escape import ARModel, AbsorbEscapeConfig, absorb_escape
from discdiff.refine.conditions import absorb_condition, escape_natural

BASES = "ACGT"


# ---------------- Absorb-Escape: the core contribution ----------------

def test_absorb_condition_fires_below_threshold():
    assert absorb_condition(0.4, t_absorb=0.9) is True
    assert absorb_condition(0.95, t_absorb=0.9) is False


def test_escape_natural_stops_when_diffusion_reconfident():
    assert escape_natural(l_d_j=0.9, l_m_j=0.5) is True   # diffusion > AR -> escape
    assert escape_natural(l_d_j=0.3, l_m_j=0.8) is False  # still in valley


class OracleAR:
    """A mock autoregressive model that always emits the CORRECT target token with high
    confidence — the ideal local corrector (Fig 4: it knows the real motif is TATA)."""

    def __init__(self, target: list, conf: float = 0.99):
        self.target = target
        self.conf = conf

    def generate(self, prefix, pos):
        return self.target[pos], self.conf


def test_absorb_escape_repairs_valley_TATT_to_TATA():
    """Fig 4 exactly: DiscDiff emits ...TATT... but is UNconfident over the wrong 'T';
    Absorb-Escape hands that region to the AR model, which restores TATA."""
    target = [BASES.index(c) for c in "CGCGCATATACGCG"]   # the intended (correct) sequence
    generated = [BASES.index(c) for c in "CGCGCATATTCGCG"]  # DiscDiff's version (TATT valley)
    # diffusion confident everywhere EXCEPT the wrong nucleotide (index 8)
    l_d = [0.95] * len(generated)
    l_d[8] = 0.30                                            # the low-confidence 'valley'

    ar = OracleAR(target)
    refined = absorb_escape(generated, l_d, ar,
                            AbsorbEscapeConfig(t_absorb=0.9, escape="natural"))
    assert refined == target                                # TATT -> TATA repaired
    assert refined != generated


def test_absorb_escape_is_noop_when_all_confident():
    """No valleys -> nothing absorbed -> sequence returned unchanged."""
    generated = [BASES.index(c) for c in "ACGTACGTACGT"]
    l_d = [0.99] * len(generated)
    ar = OracleAR([0] * len(generated))                     # would corrupt if ever called
    refined = absorb_escape(generated, l_d, ar, AbsorbEscapeConfig(t_absorb=0.9))
    assert refined == generated


def test_absorb_escape_only_touches_low_confidence_region():
    """A single isolated valley is corrected; confident flanks are preserved."""
    target = [BASES.index(c) for c in "AAAACAAAA"]
    generated = [BASES.index(c) for c in "AAAAGAAAA"]       # wrong middle base
    l_d = [0.99] * 9
    l_d[4] = 0.2
    refined = absorb_escape(generated, l_d, OracleAR(target),
                            AbsorbEscapeConfig(t_absorb=0.9, escape="natural"))
    assert refined[:4] == generated[:4]                     # flanks untouched
    assert refined[4] == target[4]                          # valley fixed


# ---------------- DiscDiff LDM: structural verification ----------------

def test_vae_roundtrip_shapes_and_finite_loss():
    torch.manual_seed(0)
    vae = DiscDiffVAE(k_channels=16, c_channels=4, seq_len=64)
    s_idx = torch.randint(0, 4, (2, 64))
    logits, loss, acc = vae(s_idx)
    assert logits.shape == (2, 4, 64)
    assert torch.isfinite(loss)
    assert 0.0 <= acc.item() <= 1.0


def test_latent_unet_eps_prediction_and_ddpm_step():
    torch.manual_seed(0)
    unet = LatentUNet(ch=4, emb_dim=32, n_species=15)
    z = torch.randn(2, 4, 16)
    t = torch.randint(0, 1000, (2,))
    species = torch.randint(0, 15, (2,))
    eps = unet(z, t, species)
    assert eps.shape == z.shape                             # predicts noise of same shape
    ddpm = DDPM(n_steps=10)
    loss = ddpm.loss(unet, z, species)
    assert torch.isfinite(loss)
    z_prev = ddpm.p_sample_step(unet, z, t=5, species=species)
    assert z_prev.shape == z.shape                          # one reverse step runs


# ---------------- Metrics ----------------

def test_motif_correlation_and_delta_div():
    nat = ["ACGT" + "TATA" + "ACGT"] * 20
    gen_same = list(nat)
    assert motif_correlation(gen_same, nat, motif="TATA", length=12) == pytest.approx(1.0)
    assert delta_diversity(gen_same, nat, sample=20) == pytest.approx(0.0, abs=1e-9)


def test_sfid_gated_returns_none_without_sei():
    assert s_fid(["ACGT"], ["ACGT"], sei_embedder=None) is None
