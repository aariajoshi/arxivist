"""Evaluate generated DNA against EPD-GenDNA (Sec 5.2 metrics: CorTATA, Delta-Div, S-FID).

Works on any FASTA of generated sequences. CorTATA + Delta-Div need no extra model; S-FID
is computed only if a Sei embedder is supplied (else reported as None, like D3LM's SFID).

    python evaluate.py --config configs/config.yaml --generated results/generated.fasta
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from discdiff.data.epd_gendna import load_sequences
from discdiff.evaluation.metrics import delta_diversity, motif_correlation, s_fid


def read_fasta(path: str):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--generated", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    L = cfg["data"]["seq_len"]
    motif = cfg["eval"]["motif"]

    gen = read_fasta(args.generated)
    nat = load_sequences(split="test", length=L, n=cfg["data"]["n_eval"])
    print(f"[eval] generated={len(gen)} reference={len(nat)}")

    cortata = motif_correlation(gen, nat, motif=motif, length=L)
    ddiv = delta_diversity(gen, nat)
    sfid = s_fid(gen, nat, sei_embedder=None)  # supply a Sei embedder for a real S-FID
    if sfid is None:
        print("[sfid] Sei model not provided -> S-FID skipped. Use CorTATA/Delta-Div "
              "(cheap, real) or supply a Sei embedder.")
    print(f"[results] {{'CorTATA({motif})': {cortata:.4f}, 'delta_div': {ddiv:.4f}, "
          f"'sfid': {sfid}}}")
    print("[paper]   Absorb-Escape small(256bp): S-FID 3.21 | CorTATA 0.975 | dDiv 5.70% "
          "|| DiscDiff: S-FID 57.4 | CorTATA 0.973")


if __name__ == "__main__":
    main()
