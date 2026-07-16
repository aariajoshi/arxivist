"""
run_baselines.py — Train and evaluate all 5 baseline models for a single
experiment configuration, producing a comparison metrics table.

Baselines: Ordered Probit (Statsmodels), ANN (PyTorch), SVM, RF (scikit-learn), XGBoost.

Paper: Jeong (2024) Heteroskedastic Ordered Probit Models with an ANN.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hopann.utils.config import Config
from hopann.utils.reproducibility import ReproducibilityManager
from hopann.data.dataset import ExperimentSplitter
from hopann.data.transforms import SelectiveStandardScaler
from hopann.models.baselines import (
    OrderedProbitBaseline, ANNBaseline, SVMBaseline,
    RandomForestBaseline, XGBoostBaseline,
)
from hopann.evaluation.metrics import OrdinalMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate all 5 baseline models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=int, default=1, help="Experiment ID 1-4.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.data.experiment_id = args.experiment
    if args.data_path:
        cfg.data.data_path = args.data_path
    cfg.data.seed = args.seed

    ReproducibilityManager.seed_everything(args.seed)

    print(f"\n{'='*60}")
    print(f"  Baseline Models — Experiment {args.experiment}")
    print(f"{'='*60}\n")

    # Load splits
    splitter = ExperimentSplitter(cfg.data.data_path, cfg.data.experiment_id, cfg.data.seed)
    train_ds, val_ds, test_ds = splitter.get_splits()

    X_train, y_train = train_ds.to_numpy()
    X_val, y_val = val_ds.to_numpy()
    X_test, y_test = test_ds.to_numpy()

    # Scaler fit on train only
    scaler = SelectiveStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, exclude_col_names=cfg.data.exclude_standardisation_cols)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Ordered Probit uses total training (train+val) — no hyperparameters
    import numpy as np
    X_total_train = np.vstack([X_train_scaled, X_val_scaled])
    y_total_train = np.concatenate([y_train, y_val])

    num_classes = cfg.model.num_classes
    metrics_calc = OrdinalMetrics()
    results_all = {}

    baselines = {
        "Ordered Probit": OrderedProbitBaseline(num_classes=num_classes),
        "ANN": ANNBaseline(num_classes=num_classes, seed=args.seed),
        "SVM": SVMBaseline(num_classes=num_classes, seed=args.seed),
        "Random Forest": RandomForestBaseline(num_classes=num_classes, seed=args.seed),
        "XGBoost": XGBoostBaseline(num_classes=num_classes, seed=args.seed),
    }

    for name, model in baselines.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_total_train, y_total_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            results = metrics_calc.compute(y_test, y_pred, y_prob)
            results_all[name] = results
            print(f"    F1-macro: {results['f1_macro']:.4f}  Accuracy: {results['accuracy']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results_all[name] = {"error": str(e)}

    # Print summary table
    print(f"\n{'='*60}")
    print("  Summary Table")
    print(f"{'='*60}")
    df = pd.DataFrame(results_all).T
    print(df.to_string())

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"baselines_exp{args.experiment}.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
