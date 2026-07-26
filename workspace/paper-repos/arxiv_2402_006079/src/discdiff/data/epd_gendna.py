"""EPD-GenDNA loader (the dataset introduced by DiscDiff).

Tries the HuggingFace release (Zehui127/*); on failure falls back to a LOUD synthetic
smoke-test set so the pipeline runs without the multi-GB download — but clearly flags
that metrics are then not paper-comparable (same pattern as the D3LM repo).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

BASES = "ACGT"
USING_SYNTHETIC = False


def _synthetic(n: int, length: int, seed: int = 0) -> List[str]:
    """Random DNA with a mild TATA-box enrichment near a fixed position, so motif
    correlation is non-degenerate for smoke tests (NOT biologically meaningful)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        arr = rng.integers(0, 4, size=length)
        if rng.random() < 0.6:  # plant a TATA near position length//4
            p = length // 4
            for k, b in enumerate("TATA"):
                if p + k < length:
                    arr[p + k] = BASES.index(b)
        seqs.append("".join(BASES[i] for i in arr))
    return seqs


def load_sequences(split: str = "test", length: int = 256, n: int = 200,
                   hf_name: str = "Zehui127/EPD-GenDNA") -> List[str]:
    """Load EPD-GenDNA sequences; fall back to synthetic with a loud banner."""
    global USING_SYNTHETIC
    try:
        from datasets import load_dataset  # noqa: F401
        ds = load_dataset(hf_name, split=split)
        seqs = [r["sequence"][:length] for r in ds][:n]
        if seqs:
            USING_SYNTHETIC = False
            return seqs
        raise RuntimeError("empty split")
    except Exception:
        USING_SYNTHETIC = True
        print("=" * 72)
        print(f"[epd] no real EPD-GenDNA ({hf_name}) -> SYNTHETIC test (n={n}).")
        print("[epd] !! SMOKE TEST ONLY — metrics here are NOT comparable to the paper.")
        print("[epd] !! Provide real EPD-GenDNA for S-FID/CorTATA evaluation.")
        print("=" * 72)
        return _synthetic(n, length)


def one_hot_indices(seq: str) -> np.ndarray:
    """DNA string -> integer token array in {0..3}."""
    return np.array([BASES.index(c) for c in seq if c in BASES], dtype=np.int64)
