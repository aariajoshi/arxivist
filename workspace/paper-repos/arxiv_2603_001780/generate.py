#!/usr/bin/env python
"""Generate DNA sequences with D3LM (unconditional, masked diffusion).

Reproduction entrypoint for D3LM (arXiv:2603.01780). Loads the official weights
(D3LM-R for generation) and produces sequences via the masked-diffusion sampler
(T=50 steps, temperature 1.1, random unmask order), writing a FASTA.
"""
from __future__ import annotations

import argparse
import os

from src.d3lm.models.d3lm import D3LMGenerator
from src.d3lm.utils.config import load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DNA with D3LM")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--n", type=int, default=None, help="number of sequences")
    p.add_argument("--length", type=int, default=None, help="sequence length (bp)")
    p.add_argument("--steps", type=int, default=None, help="denoising steps T")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--out", default="results/generated.fasta")
    p.add_argument("--dry-run", action="store_true", help="build only, skip generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.hardware.get("seed", 42), cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    variant = cfg.model.get("variant", "D3LM-R")
    n = args.n or cfg.generation.get("n_samples", 1000)
    length = args.length or cfg.data.get("length", 2048)
    steps = args.steps or cfg.generation.get("steps", 50)
    temperature = args.temperature or cfg.generation.get("temperature", 1.1)

    if args.dry_run:
        print(f"[dry-run] would load {variant}, generate n={n} length={length} "
              f"steps={steps} temp={temperature} on {device}")
        return

    gen = D3LMGenerator.from_pretrained(variant, device=str(device))
    print(f"[generate] {variant}: n={n} length={length} steps={steps} temp={temperature}")
    seqs = gen.generate(n=n, length=length, steps=steps, temperature=temperature)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">d3lm_{variant}_{i}\n{s}\n")
    print(f"[done] wrote {len(seqs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
