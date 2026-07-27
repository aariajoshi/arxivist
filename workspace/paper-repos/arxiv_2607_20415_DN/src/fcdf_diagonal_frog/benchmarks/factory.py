"""Builds a (Grid1D, mu_array, D_array, initial_condition, exact_density_or_None) tuple
from a config dict's `data` section. Factored out to avoid duplicating benchmark-selection
logic across entrypoints."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from fcdf_diagonal_frog.benchmarks.advection_diffusion import (
    FrontBenchmark,
    SmoothAdvectionDiffusionBenchmark,
)
from fcdf_diagonal_frog.benchmarks.ou_process import OUBenchmark
from fcdf_diagonal_frog.operators.grid import Grid1D


def build_benchmark(cfg_data: dict, n: int, t_final: Optional[float] = None):
    name = cfg_data["benchmark"]
    if name == "ou_process":
        p = cfg_data["ou"]
        bench = OUBenchmark(alpha=p["alpha"], sigma=p["sigma"], x0=p["x0"], v0=p["v0"])
        x_min, x_max = bench.domain(p.get("domain_std_devs", 6))
        grid = Grid1D(x_min, x_max, n)
        mu = bench.drift(grid.x)
        D = bench.diffusion(grid.x)
        ic = bench.initial_condition(grid.x)
        exact: Optional[Callable[[np.ndarray, float], np.ndarray]] = (
            lambda x, t: bench.exact_density(x, t)
        )
        return grid, mu, D, ic, exact
    if name == "smooth_advection_diffusion":
        D_val = cfg_data.get("_smooth_D", cfg_data["smooth_advection"]["D_sweep"][0])
        bench = SmoothAdvectionDiffusionBenchmark(D=D_val)
        x_min, x_max = bench.domain(t_final=t_final or 0.1)
        grid = Grid1D(x_min, x_max, n)
        mu = bench.drift(grid.x)
        D = bench.diffusion(grid.x)
        ic = bench.initial_condition(grid.x)
        exact = lambda x, t: bench.exact_density(x, t)
        return grid, mu, D, ic, exact
    if name == "front":
        p = cfg_data["front"]
        bench = FrontBenchmark(mu=p["mu"], D=p["D"], plateau_lo=p["plateau_lo"], plateau_hi=p["plateau_hi"])
        grid = Grid1D(0.0, 1.0, n)
        mu = bench.drift(grid.x)
        D = bench.diffusion(grid.x)
        ic = bench.initial_condition(grid.x)
        return grid, mu, D, ic, None
    raise ValueError(f"Unknown benchmark '{name}'")
