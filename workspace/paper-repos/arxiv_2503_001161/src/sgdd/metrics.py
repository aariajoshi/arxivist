"""Distribution-distance metrics (Sec 4.2): Hellinger distance and total variation."""
from __future__ import annotations

import numpy as np


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > 0 else p


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """H(p,q) = (1/sqrt2) * || sqrt(p) - sqrt(q) ||_2 in [0,1]."""
    p, q = _normalize(p).ravel(), _normalize(q).ravel()
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV(p,q) = 0.5 * ||p - q||_1 in [0,1]."""
    p, q = _normalize(p).ravel(), _normalize(q).ravel()
    return float(0.5 * np.sum(np.abs(p - q)))


def empirical_marginal(samples: np.ndarray, N: int, dims=(0, 1)) -> np.ndarray:
    """Histogram of `samples` [n, D] over the chosen `dims` -> normalized [N,...,N]."""
    shape = tuple(N for _ in dims)
    hist = np.zeros(shape)
    for s in samples:
        key = tuple(int(s[d]) for d in dims)
        hist[key] += 1
    return _normalize(hist)
