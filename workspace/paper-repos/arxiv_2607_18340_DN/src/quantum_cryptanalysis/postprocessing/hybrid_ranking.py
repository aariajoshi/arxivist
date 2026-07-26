"""
Top-K hybrid candidate ranking.

IMPORTANT: this is a best-effort SUBSTITUTE for the paper's withheld
"hardware-aware circuit-conditioning and readout post-selection technique"
(Section 5, Supplementary S4; SIR ambiguities[0], confidence 0.05 for the
real method). It is NOT a reproduction of the paper's actual method, which
is not disclosed. It exists so users can experiment with a noisy backend
and see *some* rank-narrowing behavior, clearly labeled as not
paper-faithful.

Approach: given a (possibly noisy) list of measured bitstrings from
repeated Simon-circuit runs, score every candidate key by how well it
satisfies the collected linear constraints (y . candidate = 0 mod 2, for
each measured y), and rank candidates by that score. The true key's
position in this ranking is reported (rank 1 = best).
"""

from __future__ import annotations

import numpy as np


class TopKHybridRanker:
    """Ranks candidate keys by constraint-satisfaction against noisy Simon measurements."""

    def rank_candidates(
        self, measurements: list[str], n: int, true_key: int, top_k: int | None = None
    ) -> dict:
        """Rank all 2^n candidate keys by how well they satisfy the measured constraints.

        Args:
            measurements: list of n-bit measured strings (possibly containing
                noise-induced violations of the true period's constraints).
            n: bit-length of the candidate key / period.
            true_key: the actual hidden period, for rank reporting (as an
                integer) -- known to the experimenter for validation
                purposes, not used by the ranking algorithm itself.
            top_k: if given, only the top_k candidates are returned in
                'top_k_candidates'; the full rank is still computed and
                returned in 'rank'.

        Returns:
            {'rank': int (1-indexed position of true_key in the ranking,
             lower is better), 'total_candidates': int, 'top_k_candidates': list[int]}
        """
        vectors = [
            np.array([int(c) for c in m], dtype=np.int64) for m in measurements if any(c == "1" for c in m)
        ]

        scores = np.zeros(2 ** n, dtype=np.int64)
        for candidate in range(2 ** n):
            cand_vec = np.array([int(c) for c in format(candidate, f"0{n}b")], dtype=np.int64)
            violations = sum(int((v @ cand_vec) % 2) for v in vectors)
            scores[candidate] = violations  # 0 violations = perfectly consistent candidate

        # Exclude the trivial all-zero candidate (period 0 is never the answer).
        scores[0] = np.iinfo(np.int64).max

        order = np.argsort(scores, kind="stable")  # ascending: fewest violations first
        rank = int(np.where(order == true_key)[0][0]) + 1  # 1-indexed

        top_k = top_k or len(order)
        return {
            "rank": rank,
            "total_candidates": 2 ** n - 1,
            "top_k_candidates": order[:top_k].tolist(),
        }
