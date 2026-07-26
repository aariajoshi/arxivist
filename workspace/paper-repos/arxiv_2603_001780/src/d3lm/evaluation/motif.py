"""Motif frequency distributions + correlation to natural DNA (DiscDiff/D3LM).

For a set of aligned DNA sequences, the motif distribution is the per-position
frequency of a motif (TATA-box, Initiator, GC-box, CCAAT-box). A generated set is
good if its motif distribution tracks natural DNA — quantified by the Pearson
correlation CorM between the two per-position frequency curves (DiscDiff Sec 3).
These metrics need no external model, so they are cheap REAL checks.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# Canonical consensus motifs (IUPAC-simplified to exact strings for counting).
MOTIFS = {
    "TATA": "TATAAA",
    "Initiator": "TCAGT",   # Inr consensus core (simplified)
    "GC": "GGGCGG",         # GC-box (Sp1) consensus
    "CCAAT": "CCAAT",
}


def motif_distribution(seqs: List[str], motif: str, window: int = 1) -> np.ndarray:
    """Per-position occurrence frequency (%) of ``motif`` across a sequence set."""
    if not seqs:
        return np.zeros(0)
    L = min(len(s) for s in seqs)
    m = len(motif)
    freq = np.zeros(L)
    for s in seqs:
        s = s.upper()
        for i in range(L - m + 1):
            if s[i:i + m] == motif:
                freq[i] += 1
    freq = 100.0 * freq / len(seqs)
    if window > 1:  # simple smoothing
        k = np.ones(window) / window
        freq = np.convolve(freq, k, mode="same")
    return freq


def motif_correlation(generated: List[str], natural: List[str],
                      motif_name: str = "TATA") -> float:
    """Pearson correlation between generated & natural motif distributions (CorM)."""
    motif = MOTIFS[motif_name]
    fg = motif_distribution(generated, motif)
    fn = motif_distribution(natural, motif)
    L = min(len(fg), len(fn))
    fg, fn = fg[:L], fn[:L]
    if fg.std() < 1e-9 or fn.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(fg, fn)[0, 1])


def all_motif_correlations(generated: List[str], natural: List[str]) -> Dict[str, float]:
    """CorM for every canonical motif."""
    return {name: motif_correlation(generated, natural, name) for name in MOTIFS}
