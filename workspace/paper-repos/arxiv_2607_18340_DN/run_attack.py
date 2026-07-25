#!/usr/bin/env python
"""
Run a single quantum cryptanalysis attack.

Usage:
    python run_attack.py --attack bv --n 8
    python run_attack.py --attack simon_em --n 5
    python run_attack.py --attack simon_cbcmac --n 4
    python run_attack.py --attack simon_feistel --n 3   # n here is the half-block size m
    python run_attack.py --attack grover --n 6
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantum_cryptanalysis.algorithms.bv_circuit import BernsteinVazirani  # noqa: E402
from quantum_cryptanalysis.algorithms.grover_circuit import GroverSearch  # noqa: E402
from quantum_cryptanalysis.algorithms.simon_circuit import SimonAlgorithm  # noqa: E402
from quantum_cryptanalysis.backend.execution import BackendFactory  # noqa: E402
from quantum_cryptanalysis.oracles.bernstein_vazirani import BVOracle  # noqa: E402
from quantum_cryptanalysis.oracles.grover_spn import GroverSPNOracle, ToySPNCipher  # noqa: E402
from quantum_cryptanalysis.oracles.simon_cbc_mac import CBCMACForgeryOracle  # noqa: E402
from quantum_cryptanalysis.oracles.simon_even_mansour import (  # noqa: E402
    EvenMansourOracle,
    find_valid_permutation,
)
from quantum_cryptanalysis.oracles.simon_feistel import FeistelOracle  # noqa: E402
from quantum_cryptanalysis.postprocessing.hybrid_ranking import TopKHybridRanker  # noqa: E402
from quantum_cryptanalysis.utils.config import load_config  # noqa: E402

ATTACKS = ["bv", "grover", "simon_em", "simon_cbcmac", "simon_feistel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quantum cryptanalysis attack")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--attack", type=str, required=True, choices=ATTACKS)
    parser.add_argument("--n", type=int, default=None, help="Override security parameter / block size")
    parser.add_argument("--backend-mode", type=str, default=None)
    return parser.parse_args()


def run_bv(config: dict, n: int, backend) -> dict:
    rng = random.Random(config["experiment"]["seed"])
    secret = "".join(rng.choice("01") for _ in range(n))
    oracle = BVOracle().build_circuit(secret)
    recovered = BernsteinVazirani().run(oracle, n=n, backend=backend, shots=1)
    return {"attack": "bv", "n": n, "true_secret": secret, "recovered": recovered, "match": recovered == secret}


def run_grover(config: dict, n: int, backend) -> dict:
    rng = random.Random(config["experiment"]["seed"])
    cipher = ToySPNCipher(n)
    true_key = rng.randint(1, 2 ** n - 1)
    # Pick a plaintext that gives a UNIQUE matching key for this ciphertext
    # (key -> ciphertext at fixed plaintext need not be injective in
    # general, so some (plaintext, ciphertext) pairs admit multiple valid
    # keys -- a real cryptographic property of the toy cipher, not a Grover
    # bug; see oracles/grover_spn.py). We search for an unambiguous instance
    # so this demo's success rate reflects Grover's own correctness.
    plaintext = None
    for candidate_pt in range(2 ** n):
        ciphertext = cipher.encrypt(true_key, candidate_pt)
        num_matches = sum(1 for k in range(2 ** n) if cipher.encrypt(k, candidate_pt) == ciphertext)
        if num_matches == 1:
            plaintext = candidate_pt
            break
    if plaintext is None:
        plaintext = rng.randint(0, 2 ** n - 1)  # fall back; oracle's M-correction still applies
    ciphertext = cipher.encrypt(true_key, plaintext)
    oracle = GroverSPNOracle().build_circuit(plaintext, ciphertext, n)
    recovered = GroverSearch().run(oracle, n=n, backend=backend, shots=config["experiment"]["shots"])
    return {
        "attack": "grover", "n": n, "true_key": true_key, "recovered": recovered,
        "match": recovered == true_key, "num_marked": oracle.metadata.get("num_marked"),
    }


def run_simon_em(config: dict, n: int, backend) -> dict:
    rng = random.Random(config["experiment"]["seed"])
    k1 = rng.randint(1, 2 ** n - 1)
    perm = find_valid_permutation(n, k1, rng)
    oracle = EvenMansourOracle().build_circuit(k1=k1, permutation=perm, n=n)
    simon = SimonAlgorithm()
    measurements = simon.collect_measurements(oracle, n=n, backend=backend, shots=config["experiment"]["shots"])
    s = simon.solve_period(measurements, n=n)
    true_s = format(k1, f"0{n}b")
    result = {"attack": "simon_em", "n": n, "true_k1": true_s, "recovered": s, "match": s == true_s}

    if s != true_s:
        top_k = config["even_mansour"]["top_k_by_n"].get(n, 16)
        hybrid = TopKHybridRanker().rank_candidates(measurements, n=n, true_key=k1, top_k=top_k)
        result["hybrid_fallback"] = hybrid
        result["note"] = "Clean recovery failed; hybrid ranker is a labeled SUBSTITUTE for the withheld technique."
    return result


def run_simon_cbcmac(config: dict, n: int, backend) -> dict:
    from quantum_cryptanalysis.oracles.simon_even_mansour import find_valid_permutation

    rng = random.Random(config["experiment"]["seed"])
    # Fix b=0: f(x) = Ek(x^a) XOR Ek(x^0) = Ek(x^a) XOR Ek(x), period = a^0 = a.
    # This is mathematically identical in form to the Even-Mansour reduction
    # (g(x)=P(x^k1)^P(x), period k1), so we reuse the same proven
    # deterministic permutation construction rather than an unreliable
    # random search over (a,b) pairs (same birthday-paradox-like scaling
    # issue as Even-Mansour -- see oracles/simon_even_mansour.py).
    a = rng.randint(1, 2 ** n - 1)
    b = 0
    key = 0
    perm = find_valid_permutation(n, a, rng)

    def block_cipher(k: int, pt: int) -> int:
        return perm[pt]

    true_s = a ^ b  # = a; verified discrepancy vs. paper's stated Ek(a)^Ek(b) -- see oracles/simon_cbc_mac.py

    oracle = CBCMACForgeryOracle().build_circuit(block_cipher, key, a, b, n)
    simon = SimonAlgorithm()
    measurements = simon.collect_measurements(oracle, n=n, backend=backend, shots=config["experiment"]["shots"])
    s = simon.solve_period(measurements, n=n)
    true_s_str = format(true_s, f"0{n}b")
    return {"attack": "simon_cbcmac", "n": n, "true_s": true_s_str, "recovered": s, "match": s == true_s_str}


def run_simon_feistel(config: dict, m: int, backend) -> dict:
    rng = random.Random(config["experiment"]["seed"])
    perm1 = list(range(2 ** m)); rng.shuffle(perm1)
    perm2 = list(range(2 ** m)); rng.shuffle(perm2)  # F2 must be a permutation for a clean period
    f1, f2 = (lambda a: perm1[a]), (lambda a: perm2[a])
    alpha0 = rng.randint(0, 2 ** m - 1)
    alpha1 = rng.randint(0, 2 ** m - 1)
    while alpha1 == alpha0:
        alpha1 = rng.randint(0, 2 ** m - 1)
    gamma = f1(alpha0) ^ f1(alpha1)
    true_s = format(1 | (gamma << 1), f"0{1+m}b")

    oracle = FeistelOracle().build_circuit(f1, f2, alpha0, alpha1, m)
    simon = SimonAlgorithm()
    measurements = simon.collect_measurements(oracle, n=1 + m, backend=backend, shots=config["experiment"]["shots"])
    s = simon.solve_period(measurements, n=1 + m)
    block_size = 2 * m
    return {
        "attack": "simon_feistel", "half_block_m": m, "block_size": block_size,
        "true_s": true_s, "recovered": s, "match": s == true_s,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.backend_mode:
        config["experiment"]["backend_mode"] = args.backend_mode

    backend = BackendFactory(config["experiment"]["backend_mode"]).get_backend()

    n_defaults = {
        "bv": config["bernstein_vazirani"]["n_values"][0],
        "grover": config["grover_spn"]["n_values"][0],
        "simon_em": config["even_mansour"]["n_values"][0],
        "simon_cbcmac": config["cbc_mac"]["n_values"][0],
        "simon_feistel": config["feistel"]["half_block_sizes"][0],
    }
    n = args.n if args.n is not None else n_defaults[args.attack]

    runners = {
        "bv": run_bv, "grover": run_grover, "simon_em": run_simon_em,
        "simon_cbcmac": run_simon_cbcmac, "simon_feistel": run_simon_feistel,
    }
    result = runners[args.attack](config, n, backend)

    print(f"\n=== {args.attack} (n={n}) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
