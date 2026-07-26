"""Evaluation metrics for DiscDiff (Sec 3, 'Model Evaluation').

- CorTATA / CorM: Pearson correlation of a motif's positional frequency distribution
  between generated and natural DNA (HIGHER better).
- Diversity / Delta-Div: average pairwise diversity; report |Div_gen - Div_nat| (match natural).
- S-FID: Frechet distance in the latent space of a pretrained genomic model (Sei). Gated
  on a Sei embedder; returns None otherwise (same gate as D3LM's SFID).
- reconstruction_accuracy: fraction of positions a VAE reconstructs identically (Table 4).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

BASES = "ACGT"


def motif_positional_frequency(seqs: List[str], motif: str, length: int) -> np.ndarray:
    """Fraction of sequences containing `motif` starting at each position (a positional
    frequency distribution, as in the motif-distribution plots)."""
    freq = np.zeros(length)
    m = len(motif)
    for s in seqs:
        for p in range(min(len(s), length) - m + 1):
            if s[p:p + m] == motif:
                freq[p] += 1
    return freq / max(len(seqs), 1)


def motif_correlation(gen: List[str], nat: List[str], motif: str = "TATA",
                      length: int = 256) -> float:
    """CorTATA: Pearson correlation between generated and natural positional motif freqs."""
    fg = motif_positional_frequency(gen, motif, length)
    fn = motif_positional_frequency(nat, motif, length)
    if fg.std() == 0 or fn.std() == 0:
        return 0.0
    return float(np.corrcoef(fg, fn)[0, 1])


def diversity(seqs: List[str], sample: int = 200, seed: int = 0) -> float:
    """Average pairwise Hamming diversity (fraction of differing positions)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=min(sample, len(seqs)), replace=False)
    sub = [seqs[i] for i in idx]
    if len(sub) < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            a, b = sub[i], sub[j]
            L = min(len(a), len(b))
            tot += sum(a[k] != b[k] for k in range(L)) / L
            cnt += 1
    return tot / max(cnt, 1)


def delta_diversity(gen: List[str], nat: List[str], **kw) -> float:
    """Delta-Div = |Div_gen - Div_nat| (LOWER better — match natural diversity)."""
    return abs(diversity(gen, **kw) - diversity(nat, **kw))


def reconstruction_accuracy(recon_idx: np.ndarray, true_idx: np.ndarray) -> float:
    """Fraction of positions reconstructed identically (Table 4 'Acc')."""
    return float((recon_idx == true_idx).mean())


def s_fid(gen: List[str], nat: List[str],
          sei_embedder: Optional[Callable[[List[str]], np.ndarray]] = None) -> Optional[float]:
    """S-FID: Frechet distance between generated & natural in the Sei latent space.

    Requires a Sei embedder (seqs -> [n, d] features). Returns None if unavailable
    (the cheap real metrics CorTATA / Delta-Div stay usable). Same gate as D3LM's SFID.
    """
    if sei_embedder is None:
        return None
    fg, fn = sei_embedder(gen), sei_embedder(nat)
    mu_g, mu_n = fg.mean(0), fn.mean(0)
    cov_g, cov_n = np.cov(fg, rowvar=False), np.cov(fn, rowvar=False)
    from scipy.linalg import sqrtm
    covmean = sqrtm(cov_g @ cov_n)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu_g - mu_n) ** 2) + np.trace(cov_g + cov_n - 2 * covmean))
