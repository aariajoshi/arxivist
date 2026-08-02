#!/usr/bin/env python
"""
Run all five attacks across their validated sizes and print/save a
Table 1/Table 2-style summary.

Usage:
    python run_benchmark.py --config configs/config.yaml --output results/benchmark_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from run_attack import (  # noqa: E402
    run_bv,
    run_grover,
    run_simon_cbcmac,
    run_simon_em,
    run_simon_feistel,
)
from quantum_cryptanalysis.backend.execution import BackendFactory  # noqa: E402
from quantum_cryptanalysis.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full attack benchmark")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="results/benchmark_table.json")
    parser.add_argument("--quick", action="store_true", help="Only run the smallest size per attack (fast smoke test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    backend = BackendFactory(config["experiment"]["backend_mode"]).get_backend()

    rows = []

    def add_row(fn, sizes, label):
        for n in sizes:
            t0 = time.time()
            result = fn(config, n, backend)
            result["label"] = label
            result["runtime_s"] = round(time.time() - t0, 3)
            rows.append(result)
            print(f"  {label} n={n}: match={result.get('match')} ({result['runtime_s']}s)")

    print("Running Bernstein-Vazirani...")
    sizes = config["bernstein_vazirani"]["n_values"][:1] if args.quick else config["bernstein_vazirani"]["n_values"]
    add_row(run_bv, sizes, "bv")

    print("Running Grover SPN...")
    sizes = config["grover_spn"]["n_values"][:1] if args.quick else config["grover_spn"]["n_values"]
    add_row(run_grover, sizes, "grover")

    print("Running Simon - Even-Mansour...")
    sizes = config["even_mansour"]["n_values"][:1] if args.quick else config["even_mansour"]["n_values"]
    add_row(run_simon_em, sizes, "simon_em")

    print("Running Simon - CBC-MAC forgery...")
    sizes = config["cbc_mac"]["n_values"][:1] if args.quick else config["cbc_mac"]["n_values"]
    add_row(run_simon_cbcmac, sizes, "simon_cbcmac")

    print("Running Simon - 3-round Feistel...")
    sizes = config["feistel"]["half_block_sizes"][:1] if args.quick else config["feistel"]["half_block_sizes"]
    add_row(run_simon_feistel, sizes, "simon_feistel")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)

    total = len(rows)
    matches = sum(1 for r in rows if r.get("match"))
    print(f"\n=== Summary: {matches}/{total} clean (rank-1) recoveries ===")
    print(f"Saved full results to {output_path}")
    print(
        "\nNOTE: this benchmark runs on a NOISELESS simulator by default, where the "
        "disclosed algorithms succeed by construction. It does NOT reproduce the "
        "paper's noisy real-hardware N=6-10 Even-Mansour result, which depends on "
        "a withheld technique (see README.md)."
    )


if __name__ == "__main__":
    main()
