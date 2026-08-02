#!/usr/bin/env python3
"""inference.py -- single implicit step (or single full-horizon integration) of a chosen
scheme given a benchmark's initial density and drift/diffusion. The PDE-solver analogue of
'single-sample inference'."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fcdf_diagonal_frog.benchmarks.factory import build_benchmark
from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.schemes.factory import (
    TWO_STAGE_SCHEMES,
    build_backward_euler_stepper,
    build_two_stage_stepper,
)
from fcdf_diagonal_frog.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Single implicit step demo")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scheme", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Override the implicit step parameter")
    parser.add_argument("--n", type=int, default=101)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scheme = args.scheme or cfg["model"]["scheme"]
    gamma = args.gamma if args.gamma is not None else cfg["training"]["dt"]

    grid, mu, D, p0, _ = build_benchmark(cfg["data"], n=args.n)
    op = DFOperator.assemble(grid, mu, D)

    if scheme in TWO_STAGE_SCHEMES:
        stepper = build_two_stage_stepper(cfg["model"])
        out = stepper(op, p0, gamma)
        p1 = out["p_next"]
        print(f"[inference] {scheme}: predictor sweeps={out['sweeps_predictor']}, "
              f"corrector sweeps={out['sweeps_corrector']}")
    else:
        stepper = build_backward_euler_stepper(scheme, cfg["model"])
        p1 = stepper(op, p0, gamma)

    print(f"[inference] scheme={scheme} gamma={gamma} n={args.n}")
    print(f"[inference] min(p_before)={np.min(p0):.4e}  min(p_after)={np.min(p1):.4e}")
    print(f"[inference] mass_before={np.sum(p0) * grid.h:.6f}  mass_after={np.sum(p1) * grid.h:.6f}")


if __name__ == "__main__":
    main()
