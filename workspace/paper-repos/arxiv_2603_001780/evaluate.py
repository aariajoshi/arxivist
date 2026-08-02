#!/usr/bin/env python
"""Evaluate generated DNA sequences (Sec 3.2).

Computes cheap REAL metrics that need no extra model — GC ratio (Chargaff parity),
Diversity, Novelty, and motif correlations (CorTATA etc.) — and SFID if a Sei
embedder is supplied. Compares to the paper's targets.
"""
from __future__ import annotations

import argparse

from src.d3lm.data.epd_gendna import _read_fasta, load_sequences
from src.d3lm.evaluation.metrics import diversity, gc_ratio, novelty, sfid
from src.d3lm.evaluation.motif import all_motif_correlations
from src.d3lm.utils.config import load_config, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate D3LM-generated DNA")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--generated", required=True, help="generated FASTA")
    p.add_argument("--reference", default=None, help="reference (real DNA) FASTA")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.hardware.get("seed", 42))

    generated = _read_fasta(args.generated)
    if args.reference:
        reference = _read_fasta(args.reference)
    else:
        length = cfg.data.get("length", 2048)
        reference = load_sequences(cfg.data.get("dataset", "epd_gendna"), "test",
                                   cfg.data.get("data_dir", "data/"), length=length)

    print(f"[eval] generated={len(generated)} reference={len(reference)}")
    results = {
        "gc_ratio": round(gc_ratio(generated), 4),
        "diversity": round(diversity(generated), 2),
        "novelty": round(novelty(generated, reference), 2),
    }
    results["motif_corr"] = {k: round(v, 4) for k, v in
                             all_motif_correlations(generated, reference).items()}
    s = sfid(generated, reference, sei_embed=None)  # supply a Sei embedder to enable
    results["sfid"] = None if s is None else round(s, 3)

    print("[results]", results)
    print("[paper]   D3LM-R 2048bp: SFID 10.92 | GC ratio 1.07 (Truth GC 1.06, SFID 7.85)")


if __name__ == "__main__":
    main()
