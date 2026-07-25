#!/usr/bin/env python
"""Download D3LM weights (HuggingFace) and/or the EPD-GenDNA dataset.

D3LM weights are public on the HF Hub (no token needed):
    python data/download.py --weights D3LM-R    # generation model (SFID 10.92)
    python data/download.py --weights D3LM       # understanding model

EPD-GenDNA (DiscDiff dataset) is fetched via HF datasets if available:
    python data/download.py --data epd_gendna
"""
from __future__ import annotations

import argparse

MODELS = {
    "D3LM": "Hengchang-Liu/D3LM-from-nt",
    "D3LM-R": "Hengchang-Liu/D3LM-scratch",
}


def download_weights(variant: str) -> None:
    name = MODELS.get(variant, variant)
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        print(f"[weights] fetching {name} ...")
        AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)
        print(f"[weights] cached {name} (HF cache).")
    except Exception as exc:  # noqa: BLE001
        print(f"[weights] failed for {name}: {exc}")


def download_data(data_dir: str) -> None:
    try:
        from datasets import load_dataset

        print("[data] fetching EPD-GenDNA (Zehui127/EPD-GenDNA) ...")
        ds = load_dataset("Zehui127/EPD-GenDNA")
        print(f"[data] loaded splits: {list(ds.keys())}")
    except Exception as exc:  # noqa: BLE001
        print(f"[data] EPD-GenDNA HF load failed ({exc}).")
        print("[data] The pipeline falls back to a clearly-labeled synthetic set;")
        print("[data] see data/README_data.md for manual EPD-GenDNA setup.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download D3LM weights / data")
    p.add_argument("--weights", default=None, choices=list(MODELS.keys()))
    p.add_argument("--data", default=None, choices=["epd_gendna"])
    p.add_argument("--data-dir", default="data/")
    args = p.parse_args()
    if args.weights:
        download_weights(args.weights)
    if args.data:
        download_data(args.data_dir)
    if not args.weights and not args.data:
        download_weights("D3LM-R")


if __name__ == "__main__":
    main()
