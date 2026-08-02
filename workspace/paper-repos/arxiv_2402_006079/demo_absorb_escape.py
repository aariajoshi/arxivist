"""Absorb-Escape demo (Sec 4.2, Fig 4) — runs on CPU, NO trained weights.

Shows the algorithm correcting a diffusion 'valley': a sequence where DiscDiff emitted a
wrong nucleotide with low confidence is repaired by a local autoregressive re-generation.
This is the paper's most portable contribution and is fully reproducible here.

    python demo_absorb_escape.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from discdiff.refine.absorb_escape import AbsorbEscapeConfig, absorb_escape

BASES = "ACGT"


class OracleAR:
    """Mock AR model (stands in for fine-tuned Hyena) that emits the correct local motif."""

    def __init__(self, target, conf=0.99):
        self.target, self.conf = target, conf

    def generate(self, prefix, pos):
        return self.target[pos], self.conf


def show(tokens):
    return "".join(BASES[t] for t in tokens)


def main():
    intended = [BASES.index(c) for c in "CGCGCATATACGCGATATAAA"]
    generated = [BASES.index(c) for c in "CGCGCATATTCGCGATCTAAA"]  # two valleys
    l_d = [0.96] * len(generated)
    l_d[8] = 0.28    # TATT valley (should be TATA)
    l_d[16] = 0.31   # ATCT valley (should be ATAT)

    print("DiscDiff (raw)   :", show(generated))
    print("low-conf valleys :", "".join("^" if c < 0.9 else " " for c in l_d))

    refined = absorb_escape(generated, l_d, OracleAR(intended),
                            AbsorbEscapeConfig(t_absorb=0.9, escape="natural"))
    print("Absorb-Escape    :", show(refined))
    print("intended (truth) :", show(intended))
    fixed = sum(a != b for a, b in zip(generated, refined))
    ok = refined == intended
    print(f"\n[result] {fixed} nucleotide(s) corrected; matches intended: {ok}")
    print("[note] With real weights, the AR model is a fine-tuned Hyena and the")
    print("       confidences L(D,.) come from the DiscDiff decoder softmax.")


if __name__ == "__main__":
    main()
