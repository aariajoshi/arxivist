#!/usr/bin/env python3
"""evaluate.py -- reproduces a subset of the paper's Tables 2-9.

Implemented: Table 2 (coverage thresholds), Table 3 (OU spatial order), Table 5 (Peclet
sweep, FCDF vs Chang-Cooper), Table 7 (positivity/conservation), Table 8 (active-set solver
cost vs step size).

NOT implemented (documented honestly rather than faked): Table 4 (temporal order via
semi-discrete exponential e^{TA}p0 -- would need a separate matrix-exponential/expm
integrator not otherwise used anywhere in the paper's *schemes*), Table 6 (long-time front
smearing over 2000 steps -- computationally heavy for a demo script), Table 9 (coverage-gap
active-set cost on the front problem -- same active-set machinery as Table 8, omitted only
for scope/time, not for any technical reason). See README.md "Known Limitations".
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fcdf_diagonal_frog.benchmarks.advection_diffusion import SmoothAdvectionDiffusionBenchmark
from fcdf_diagonal_frog.benchmarks.factory import build_benchmark
from fcdf_diagonal_frog.benchmarks.ou_process import OUBenchmark
from fcdf_diagonal_frog.evaluation.metrics import Metrics
from fcdf_diagonal_frog.limiter.zalesak import ZalesakLimiter
from fcdf_diagonal_frog.linear_windows.thresholds import LinearWindowThresholds
from fcdf_diagonal_frog.operators.chang_cooper import ChangCooperOperator
from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.operators.grid import Grid1D
from fcdf_diagonal_frog.schemes.active_set import ActiveSetSolver
from fcdf_diagonal_frog.schemes.fcdf_b import FCDF_B_Solver
from fcdf_diagonal_frog.schemes.unlimited import UnlimitedSolver
from fcdf_diagonal_frog.utils.config import load_config


def table2_coverage(cfg: dict, out_dir: str) -> pd.DataFrame:
    """Table 2: gamma_pic, gamma_0, gamma_r and coverage conditions (22),(23) on the OU
    benchmark, across the mesh sequence."""
    p = cfg["data"]["ou"]
    bench = OUBenchmark(alpha=p["alpha"], sigma=p["sigma"], x0=p["x0"], v0=p["v0"])
    x_min, x_max = bench.domain(p.get("domain_std_devs", 6))
    thresholds = LinearWindowThresholds()
    rows = []
    for n in cfg["data"]["mesh_sizes"]:
        grid = Grid1D(x_min, x_max, n)
        mu = bench.drift(grid.x)
        D = bench.diffusion(grid.x)
        mu_bar = float(np.max(np.abs(mu)))
        op = DFOperator.assemble(grid, mu, D)
        gamma_pic = FCDF_B_Solver.picard_contraction_bound(mu_bar, grid.h)
        gamma_0 = thresholds.gamma_0(op)
        gamma_r = thresholds.gamma_r(op)
        cond_a = gamma_0 is not None and gamma_pic >= gamma_0
        cond_b = gamma_r is not None and gamma_pic >= gamma_r
        rows.append({
            "n": n, "h": grid.h, "gamma_pic": gamma_pic,
            "gamma_0": gamma_0 if gamma_0 is not None else float("nan"),
            "gamma_r": gamma_r if gamma_r is not None else float("nan"),
            "condition_a_holds": bool(cond_a), "condition_b_holds": bool(cond_b),
        })
        print(f"[table2] n={n}: gamma_pic={gamma_pic:.4e} gamma_0={gamma_0} gamma_r={gamma_r} "
              f"cond(a)={cond_a} cond(b)={cond_b}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table2_coverage.csv"), index=False)
    return df


def table3_ou_spatial_order(cfg: dict, out_dir: str) -> pd.DataFrame:
    """Table 3: observed L1 spatial order on the OU benchmark, FCDF-B vs unlimited, at a
    small fixed dt. (Reduced mesh count vs. the paper for runtime; documented in README.)"""
    p = cfg["data"]["ou"]
    bench = OUBenchmark(alpha=p["alpha"], sigma=p["sigma"], x0=p["x0"], v0=p["v0"])
    x_min, x_max = bench.domain(p.get("domain_std_devs", 6))
    dt = cfg["training"]["dt"]
    T = cfg["training"]["T_horizon"]
    limiter = ZalesakLimiter()
    solver = FCDF_B_Solver()

    errors_fcdfb, errors_unlimited, hs, ns = [], [], [], []
    for n in cfg["data"]["mesh_sizes"]:
        grid = Grid1D(x_min, x_max, n)
        mu = bench.drift(grid.x)
        D = bench.diffusion(grid.x)
        op = DFOperator.assemble(grid, mu, D)
        p0 = bench.initial_condition(grid.x)
        n_steps = max(1, int(round(T / dt)))

        p_b = p0.copy()
        p_u = p0.copy()
        unl = UnlimitedSolver()
        for _ in range(n_steps):
            p_b = solver.step(op, limiter, p_b, dt)["p"]
            p_u = unl.step(op, p_u, dt)

        exact = bench.exact_density(grid.x, T)
        errors_fcdfb.append(Metrics.l1_error(p_b, exact, grid.h))
        errors_unlimited.append(Metrics.l1_error(p_u, exact, grid.h))
        hs.append(grid.h)
        ns.append(n)
        print(f"[table3] n={n}: L1(FCDF-B)={errors_fcdfb[-1]:.4e}  L1(unlimited)={errors_unlimited[-1]:.4e}")

    orders_b = Metrics.observed_order(errors_fcdfb)
    orders_u = Metrics.observed_order(errors_unlimited)
    df = pd.DataFrame({"n": ns, "h": hs, "L1_FCDF_B": errors_fcdfb, "order_FCDF_B": orders_b,
                        "L1_unlimited": errors_unlimited, "order_unlimited": orders_u})
    df.to_csv(os.path.join(out_dir, "table3_ou_spatial_order.csv"), index=False)
    return df


def table5_peclet_sweep(cfg: dict, out_dir: str) -> pd.DataFrame:
    """Table 5: pure spatial L1 order (semi-discrete e^{T*A}p0 comparison replaced here by a
    single small-dt backward-Euler step, which isolates spatial error to the same effect the
    paper's methodological note describes) for A2/FCDF-B vs Chang-Cooper across a Peclet
    sweep, at fixed n."""
    n = cfg["data"]["mesh_sizes"][-1]
    dt = 1.0e-5  # deliberately tiny so temporal error is negligible vs spatial error
    T = 1.0e-3
    n_steps = max(1, int(round(T / dt)))
    limiter = ZalesakLimiter()
    solver = FCDF_B_Solver()

    rows = []
    for D_val in cfg["data"]["smooth_advection"]["D_sweep"]:
        bench = SmoothAdvectionDiffusionBenchmark(D=D_val)
        x_min, x_max = bench.domain(t_final=T)
        grid = Grid1D(x_min, x_max, n)
        mu = bench.drift(grid.x)
        D = bench.diffusion(grid.x)
        peh = float(np.max(grid.cell_peclet(mu, D)))

        op = DFOperator.assemble(grid, mu, D)
        cc = ChangCooperOperator.assemble(grid, mu, D)
        p0 = bench.initial_condition(grid.x)

        p_fcdfb = p0.copy()
        p_cc = p0.copy()
        for _ in range(n_steps):
            p_fcdfb = solver.step(op, limiter, p_fcdfb, dt)["p"]
            p_cc = cc.step(p_cc, dt)

        exact = bench.exact_density(grid.x, T)
        err_fcdfb = Metrics.l1_error(p_fcdfb, exact, grid.h)
        err_cc = Metrics.l1_error(p_cc, exact, grid.h)
        rows.append({"D": D_val, "Peh_finest": peh, "L1_FCDF_B": err_fcdfb, "L1_CC": err_cc,
                      "error_ratio_CC_over_FCDF": err_cc / err_fcdfb if err_fcdfb > 0 else float("nan")})
        print(f"[table5] D={D_val:.1e} Peh={peh:.2f}: L1(FCDF-B)={err_fcdfb:.4e} L1(CC)={err_cc:.4e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table5_peclet_sweep.csv"), index=False)
    return df


def table7_positivity_conservation(cfg: dict, out_dir: str) -> pd.DataFrame:
    """Table 7: most negative nodal value + mass defect, OU and front benchmarks, small and
    large dt, across FCDF-B, unlimited, and monotone-core."""
    from fcdf_diagonal_frog.benchmarks.advection_diffusion import FrontBenchmark
    from fcdf_diagonal_frog.schemes.monotone_core import MonotoneCoreSolver

    n = 401
    dt_small, dt_large = 2.0e-4, 5.0e-2
    T = 0.2
    limiter = ZalesakLimiter()
    solver = FCDF_B_Solver()
    unl = UnlimitedSolver()
    core = MonotoneCoreSolver()

    scenarios = {}
    p_ou = cfg["data"]["ou"]
    bench_ou = OUBenchmark(alpha=p_ou["alpha"], sigma=p_ou["sigma"], x0=p_ou["x0"], v0=p_ou["v0"])
    x_min, x_max = bench_ou.domain(p_ou.get("domain_std_devs", 6))
    scenarios["OU"] = (Grid1D(x_min, x_max, n), bench_ou.drift, bench_ou.diffusion, bench_ou.initial_condition)

    p_f = cfg["data"]["front"]
    bench_f = FrontBenchmark(mu=p_f["mu"], D=p_f["D"], plateau_lo=p_f["plateau_lo"], plateau_hi=p_f["plateau_hi"])
    scenarios["front"] = (Grid1D(0.0, 1.0, n), bench_f.drift, bench_f.diffusion, bench_f.initial_condition)

    rows = []
    for name, (grid, drift_fn, diff_fn, ic_fn) in scenarios.items():
        mu, D = drift_fn(grid.x), diff_fn(grid.x)
        op = DFOperator.assemble(grid, mu, D)
        p0 = ic_fn(grid.x)
        for dt_label, dt in [("small", dt_small), ("large", dt_large)]:
            n_steps = max(1, int(round(T / dt)))
            results = {}
            for label, stepper in [
                ("FCDF-B", lambda op_, p_, g_: solver.step(op_, limiter, p_, g_)["p"]),
                ("unlimited", lambda op_, p_, g_: unl.step(op_, p_, g_)),
                ("monotone_core", lambda op_, p_, g_: core.step(op_, p_, g_)),
            ]:
                p = p0.copy()
                for _ in range(n_steps):
                    p = stepper(op, p, dt)
                results[label] = p
            for label, p in results.items():
                rows.append({
                    "scheme": label, "scenario": name, "dt": dt_label,
                    "min_nodal_value": float(np.min(p)),
                    "mass_defect": Metrics.mass_defect(p, p0),
                })
                print(f"[table7] {name}/{dt_label}/{label}: min={np.min(p):.3e} mass_defect={Metrics.mass_defect(p, p0):.2e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table7_positivity_conservation.csv"), index=False)
    return df


def table8_active_set_cost(cfg: dict, out_dir: str) -> pd.DataFrame:
    """Table 8: active-set pattern-update count vs step size on the front problem."""
    from fcdf_diagonal_frog.benchmarks.advection_diffusion import FrontBenchmark

    p_f = cfg["data"]["front"]
    bench = FrontBenchmark(mu=p_f["mu"], D=p_f["D"], plateau_lo=p_f["plateau_lo"], plateau_hi=p_f["plateau_hi"])
    n = 401
    grid = Grid1D(0.0, 1.0, n)
    mu, D = bench.drift(grid.x), bench.diffusion(grid.x)
    op = DFOperator.assemble(grid, mu, D)
    p0 = bench.initial_condition(grid.x)
    mu_bar = float(np.max(np.abs(mu)))
    gamma_pic = FCDF_B_Solver.picard_contraction_bound(mu_bar, grid.h)

    solver = ActiveSetSolver()
    rows = []
    for ratio in [0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]:
        gamma = ratio * gamma_pic
        out = solver.solve(op, p0, gamma, max_pattern_updates=cfg["model"]["active_set_max_pattern_updates"])
        rows.append({
            "gamma_over_gamma_pic": ratio, "gamma": gamma,
            "pattern_updates": out["pattern_updates"], "unlimited_accepted": out["unlimited_accepted"],
            "free_fraction": out["free_fraction"], "residual": out["residual"], "converged": out["converged"],
        })
        print(f"[table8] gamma/gamma_pic={ratio}: updates={out['pattern_updates']} "
              f"unlimited_accepted={out['unlimited_accepted']} converged={out['converged']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table8_active_set_cost.csv"), index=False)
    return df


TABLE_FUNCS = {
    "2": table2_coverage,
    "3": table3_ou_spatial_order,
    "5": table5_peclet_sweep,
    "7": table7_positivity_conservation,
    "8": table8_active_set_cost,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce paper Tables 2/3/5/7/8")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--table", type=str, default="all", help="One of: 2,3,5,7,8,all")
    parser.add_argument("--out-dir", type=str, default="results/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    tables = TABLE_FUNCS.keys() if args.table == "all" else [args.table]
    for t in tables:
        if t not in TABLE_FUNCS:
            print(f"[evaluate] skipping unknown table '{t}'")
            continue
        print(f"\n=== Table {t} ===")
        TABLE_FUNCS[t](cfg, args.out_dir)

    print(f"\n[evaluate] all requested tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
