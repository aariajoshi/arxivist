"""Generation-quality metrics for D3LM (Sec 3.2, Appendix B.2).

* **GC ratio** — #G / #C, a compositional-validity check (Chargaff's parity rule;
  natural DNA ~ 1.0). Cheap and discriminative (paper: D3LM-R 1.07 vs Truth 1.06;
  collapsed models Evo 0.86, P2 12.7).
* **Diversity** — average pairwise Levenshtein distance among generated sequences.
* **Novelty** — average min edit distance from each generated sequence to the
  training set (guards against memorization).
* **SFID** — Sei-based Frechet Inception Distance: embed sequences with the
  pretrained Sei genomic CNN (20k chromatin profiles), then Frechet distance
  between generated and real feature distributions. Requires Sei; returns None
  with a clear message if unavailable.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np


def gc_ratio(seqs: List[str]) -> float:
    """Mean #G/#C over sequences (Chargaff parity; ~1.0 for natural DNA)."""
    ratios = []
    for s in seqs:
        s = s.upper()
        g, c = s.count("G"), s.count("C")
        if c > 0:
            ratios.append(g / c)
    return float(np.mean(ratios)) if ratios else 0.0


def _edit(a: str, b: str) -> int:
    try:
        import Levenshtein

        return Levenshtein.distance(a, b)
    except Exception:
        # O(len) Hamming fallback for equal-length DNA (fast, adequate proxy)
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b))
        # tiny DP fallback
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[n]


def diversity(seqs: List[str], max_pairs: int = 2000, seed: int = 42) -> float:
    """Average pairwise Levenshtein distance (sampled for large sets)."""
    rng = random.Random(seed)
    n = len(seqs)
    if n < 2:
        return 0.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    return float(np.mean([_edit(seqs[i], seqs[j]) for i, j in pairs]))


def novelty(generated: List[str], train: List[str],
            max_train: int = 500, seed: int = 42) -> float:
    """Average min edit distance from each generated seq to the training set."""
    rng = random.Random(seed)
    ref = train if len(train) <= max_train else rng.sample(train, max_train)
    if not ref:
        return 0.0
    return float(np.mean([min(_edit(g, r) for r in ref) for g in generated]))


def _frechet(mu1, s1, mu2, s2) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def sfid(generated: List[str], reference: List[str],
         sei_embed=None) -> Optional[float]:
    """Sei-based Frechet Inception Distance (lower is better).

    ``sei_embed`` must be a callable seqs -> np.ndarray[N, D] using the pretrained
    Sei model. If None (Sei unavailable), returns None with a message — the cheap
    metrics above still validate generation.
    """
    if sei_embed is None:
        print("[sfid] Sei model not provided -> SFID skipped. "
              "Use gc_ratio/diversity/novelty (cheap, real) or supply a Sei embedder.")
        return None
    fg = np.asarray(sei_embed(generated))
    fr = np.asarray(sei_embed(reference))
    return _frechet(fg.mean(0), np.cov(fg, rowvar=False),
                    fr.mean(0), np.cov(fr, rowvar=False))
