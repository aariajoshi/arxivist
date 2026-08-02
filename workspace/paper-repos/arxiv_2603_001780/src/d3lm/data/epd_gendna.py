"""EPD-GenDNA dataset loader (Sec 3.1).

D3LM trains/evaluates on EPD-GenDNA (Li et al. 2024, the DiscDiff dataset): 160k
DNA sequences across 15 species, 2048 or 256 bp, centered on the TSS. D3LM uses
the **mammalian subset** (~80k sequences).

Reads a local FASTA/CSV first; else tries the released HuggingFace dataset; else
falls back to a clearly-labeled SYNTHETIC promoter-like generator so the pipeline
and tests run without the (large) download. Synthetic sequences are NOT a
stand-in for real evaluation.
"""
from __future__ import annotations

import os
import random
from typing import List

from torch.utils.data import Dataset

_BASES = "ACGT"

#: True whenever sequences came from the synthetic fallback.
USING_SYNTHETIC = False


class EPDGenDNADataset(Dataset):
    """A dataset of DNA strings for D3LM training/eval."""

    def __init__(self, sequences: List[str]) -> None:
        self.sequences = sequences

    def __repr__(self) -> str:  # noqa: D105
        return f"EPDGenDNADataset(n={len(self)})"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def _read_fasta(path: str) -> List[str]:
    seqs, cur = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def _synthetic_promoters(n: int, length: int, seed: int) -> List[str]:
    """Toy promoter-like sequences with a TATA-box motif near the center.

    Purely a smoke-test generator — the motif structure is crude and NOT a
    substitute for real EPD-GenDNA. Metrics on these are not comparable to paper.
    """
    rng = random.Random(seed)
    tss = length // 2
    seqs = []
    for _ in range(n):
        s = [rng.choice(_BASES) for _ in range(length)]
        # plant a weak TATA box ~30bp upstream of TSS
        if tss - 30 >= 0:
            for j, b in enumerate("TATAAA"):
                if rng.random() < 0.7:
                    s[tss - 30 + j] = b
        seqs.append("".join(s))
    return seqs


def load_sequences(dataset: str, split: str, data_dir: str,
                   length: int = 2048, synthetic_n: int = 200) -> List[str]:
    """Load EPD-GenDNA sequences for a split, else synthesize (clearly flagged)."""
    global USING_SYNTHETIC
    # 1) local FASTA
    fasta = os.path.join(data_dir, dataset, f"{split}.fasta")
    if os.path.isfile(fasta):
        return _read_fasta(fasta)
    # 2) released HF dataset
    try:
        from datasets import load_dataset

        ds = load_dataset("Zehui127/EPD-GenDNA", split=split)  # DiscDiff release
        col = "sequence" if "sequence" in ds.column_names else ds.column_names[0]
        return [s[:length] for s in ds[col]]
    except Exception:
        pass
    # 3) synthetic fallback
    USING_SYNTHETIC = True
    seed = {"train": 0, "val": 1, "test": 2}.get(split, 3)
    n = synthetic_n
    print("=" * 72)
    print(f"[epd] no real EPD-GenDNA at {fasta} (and HF load failed) -> SYNTHETIC {split} (n={n}).")
    print("[epd] !! SMOKE TEST ONLY — metrics here are NOT comparable to the paper.")
    print("[epd] !! Provide real EPD-GenDNA (data/README_data.md) for SFID/GC evaluation.")
    print("=" * 72)
    return _synthetic_promoters(n, length, seed)
