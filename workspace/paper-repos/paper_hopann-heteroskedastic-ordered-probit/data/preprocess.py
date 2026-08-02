"""
data/preprocess.py — Preprocessing script for Amazon Software Product Reviews.

Applies the paper's filtering and feature engineering steps to produce the
CSV used by the training pipeline.

Paper: Jeong (2024) Heteroskedastic Ordered Probit Models with an ANN.
       Section 3.1 (Empirical Application — Data)

Usage:
    # Process real data:
    python data/preprocess.py \\
        --raw-path data/Software_5.json.gz \\
        --output-path data/amazon_software_reviews.csv \\
        --start-date 2018-01-01 --end-date 2018-09-26

    # Generate synthetic data for testing:
    python data/preprocess.py --synthetic --n-samples 500 \\
        --output-path data/amazon_software_reviews.csv
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Amazon Software Reviews dataset.")
    parser.add_argument("--raw-path", type=str, default=None,
                        help="Path to raw Software_5.json.gz download.")
    parser.add_argument("--output-path", type=str, default="data/amazon_software_reviews.csv",
                        help="Output CSV path.")
    parser.add_argument("--start-date", type=str, default="2018-01-01",
                        help="Filter reviews from this date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2018-09-26",
                        help="Filter reviews up to this date (YYYY-MM-DD).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for testing (no real data needed).")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of synthetic samples to generate.")
    return parser.parse_args()


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load raw Amazon review JSON (gzipped)."""
    print(f"Loading raw data from: {raw_path}")
    records = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    df = pd.json_normalize(records)
    print(f"  Loaded {len(df):,} records.")
    return df


def filter_and_engineer_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Apply paper's filtering and feature engineering.
    Section 3.1: Digital footprint mining with time-ordered structure.
    """
    # Convert unix timestamp to datetime
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["unixReviewTime"], unit="s", errors="coerce")
    df = df.dropna(subset=["review_date"])

    # Filter date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    df = df[(df["review_date"] >= start) & (df["review_date"] <= end)].copy()
    print(f"  After date filter ({start_date} to {end_date}): {len(df):,} records.")

    # Keep first review per reviewer (one review per consumer)
    df = df.sort_values("review_date")
    df = df.drop_duplicates(subset="reviewerID", keep="first")
    print(f"  After deduplication (1 per reviewer): {len(df):,} records.")

    # Sort by timestamp
    df = df.sort_values("review_date").reset_index(drop=True)

    # Compute time-indexed features (paper Section 3.1)
    # For each review at time t_i:
    #   asin_n_rev: number of prior reviews for that product before t_i
    #   asin_n_reviewers: number of prior unique reviewers for that product before t_i
    #   asin_verified_share: share of verified purchases in prior reviews
    df["asin_n_rev"] = (
        df.groupby("asin").cumcount()  # reviews before current
    )
    df["asin_n_reviewers"] = df["asin_n_rev"]  # proxy; exact same as n_rev if 1/reviewer
    df["asin_verified_share"] = (
        df.groupby("asin")["verified"].transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    )

    # User prior review count
    df["user_n_rev"] = df.groupby("reviewerID").cumcount()
    df["user_reviewed_asin"] = df["user_n_rev"]  # same product category

    # Day-of-week dummies (Saturday = base group, as in paper)
    # Days: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    df["day_of_week"] = df["review_date"].dt.dayofweek
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sun"]):
        df[f"day_{name}"] = (df["day_of_week"] == d).astype(int)
    # Saturday (5) is the base group — excluded

    # Price (handle missing)
    df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(df.get("price", pd.Series([0])).median())

    # Select and rename columns
    day_cols = [c for c in df.columns if c.startswith("day_") and c != "day_of_week"]
    feature_cols = [
        "overall", "price", "asin_n_rev", "asin_n_reviewers", "asin_verified_share",
        "user_n_rev", "user_reviewed_asin",
    ] + day_cols + ["reviewerID", "asin", "review_date"]

    available = [c for c in feature_cols if c in df.columns]
    out = df[available].copy()
    out.columns = [c.lower() for c in out.columns]
    return out


def generate_synthetic_data(n_samples: int) -> pd.DataFrame:
    """
    Generate synthetic data matching the expected schema.
    Used for smoke tests only — NOT for paper reproduction.
    """
    print(f"Generating {n_samples} synthetic samples...")
    np.random.seed(42)
    n = n_samples
    dates = pd.date_range("2018-01-01", "2018-09-26", periods=n)
    data = {
        "overall": np.random.randint(1, 6, n),
        "price": np.random.exponential(60, n).clip(0.01, 1900),
        "asin_n_rev": np.random.poisson(650, n),
        "asin_n_reviewers": np.random.poisson(648, n),
        "asin_verified_share": np.random.beta(5, 1.2, n).clip(0, 1),
        "user_n_rev": np.random.poisson(0.265, n).clip(0, 24),
        "user_reviewed_asin": np.random.poisson(0.261, n).clip(0, 23),
        "day_mon": np.random.randint(0, 2, n),
        "day_tue": np.random.randint(0, 2, n),
        "day_wed": np.random.randint(0, 2, n),
        "day_thu": np.random.randint(0, 2, n),
        "day_fri": np.random.randint(0, 2, n),
        "day_sun": np.random.randint(0, 2, n),
        "reviewerid": [f"R{i:05d}" for i in range(n)],
        "asin": [f"B{np.random.randint(10000, 99999):05d}" for _ in range(n)],
        "review_date": dates,
    }
    return pd.DataFrame(data)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        df = generate_synthetic_data(args.n_samples)
    else:
        if not args.raw_path or not Path(args.raw_path).exists():
            print("ERROR: --raw-path is required for real data processing.")
            print("See data/README_data.md for download instructions.")
            print("Or use --synthetic to generate test data.")
            sys.exit(1)
        df_raw = load_raw_data(args.raw_path)
        df = filter_and_engineer_features(df_raw, args.start_date, args.end_date)

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} records to: {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
