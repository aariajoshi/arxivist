#!/usr/bin/env python3
"""train.py -- runs one scheme through one benchmark to a time horizon T.

There is no learned model in this paper (it's a numerical PDE scheme), so this entrypoint
is kept as the ArXivist repo-template's 'training run' analogue: it time-marches a chosen
scheme on a chosen benchmark and saves the resulting density + diagnostics to results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fcdf_diagonal_frog.benchmarks.factory import build_benchmark
from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.schemes.factory import (
    TWO_STAGE_SCHEMES,
    build_backward_euler_stepper,
    build_two_stage_stepper,
)
from fcdf_diagonal_frog.utils.config import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-march a FCDF-family scheme (no ML training exists in this paper)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scheme", type=str, default=None, help="Override config.model.scheme")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="n=51, T=0.02 fast smoke run")
    parser.add_argument("--dry-run", action="store_true", help="Assemble operators only, no time marching")
    parser.add_argument("--out-dir", type=str, default="results/")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    scheme = args.scheme or cfg["model"]["scheme"]
    n = 51 if args.debug else cfg["data"]["mesh_sizes"][-1]
    T = 0.02 if args.debug else cfg["training"]["T_horizon"]
    dt = cfg["training"]["dt"]

    grid, mu, D, p0, exact = build_benchmark(cfg["data"], n=n, t_final=T)
    op = DFOperator.assemble(grid, mu, D)

    print(f"[train] scheme={scheme} benchmark={cfg['data']['benchmark']} n={n} dt={dt} T={T}")
    print(f"[train] grid: {grid}")
    print(f"[train] param count (n/a, no learned params): mesh nodes = {n}")

    if args.dry_run:
        print("[train] --dry-run: operators assembled successfully, exiting without integrating.")
        return

    n_steps = max(1, int(round(T / dt)))
    p = p0.copy()
    t0 = time.time()

    if scheme in TWO_STAGE_SCHEMES:
        stepper = build_two_stage_stepper(cfg["model"])
        for _ in range(n_steps):
            out = stepper(op, p, dt)
            p = out["p_next"]
    else:
        stepper = build_backward_euler_stepper(scheme, cfg["model"])
        for _ in range(n_steps):
            p = stepper(op, p, dt)

    elapsed = time.time() - t0
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"train_{scheme}_{cfg['data']['benchmark']}_n{n}.json")
    result = {
        "scheme": scheme,
        "benchmark": cfg["data"]["benchmark"],
        "n": n,
        "dt": dt,
        "T": T,
        "n_steps": n_steps,
        "elapsed_seconds": elapsed,
        "min_nodal_value": float(np.min(p)),
        "mass": float(np.sum(p) * grid.h),
        "mass_defect": float(abs(np.sum(p) - np.sum(p0))),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[train] done in {elapsed:.2f}s. min(p)={result['min_nodal_value']:.3e}  "
          f"mass_defect={result['mass_defect']:.3e}")
    print(f"[train] wrote {out_path}")


if __name__ == "__main__":
    main()
