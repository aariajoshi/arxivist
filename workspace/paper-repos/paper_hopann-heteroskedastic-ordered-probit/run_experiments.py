"""
run_experiments.py — Full paper reproduction script.

Runs all 4 experiments × all 7 models (OPANN, HOPANN, Ordered Probit, ANN,
SVM, RF, XGB) and aggregates results into a single CSV/JSON table matching
the paper's reported results.

Paper: Jeong (2024) Heteroskedastic Ordered Probit Models with an ANN.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.utils.data as td

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hopann.utils.config import Config
from hopann.utils.reproducibility import ReproducibilityManager
from hopann.data.dataset import ExperimentSplitter
from hopann.data.transforms import SelectiveStandardScaler
from hopann.models.opann import OPANN
from hopann.models.hopann import HOPANN
from hopann.models.baselines import (
    OrderedProbitBaseline, ANNBaseline, SVMBaseline,
    RandomForestBaseline, XGBoostBaseline,
)
from hopann.training.losses import OrderedProbitNLL
from hopann.training.trainer import Trainer, HyperparameterSearcher
from hopann.evaluation.metrics import OrdinalMetrics


EXPERIMENT_NAMES = {
    1: "5-class Imbalanced",
    2: "3-class Imbalanced",
    3: "Modified 3-class Imbalanced",
    4: "Modified 3-class Balanced",
}
EXPERIMENT_NUM_CLASSES = {1: 5, 2: 3, 3: 3, 4: 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full paper reproduction: all experiments x all models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline models; only run OPANN and HOPANN.")
    parser.add_argument("--experiments", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Which experiment IDs to run (default: all 4).")
    return parser.parse_args()


def run_neural_model(model_type, cfg, train_ds, val_ds, test_ds,
                     num_classes, device, output_dir, exp_id, seed):
    """Train and evaluate OPANN or HOPANN for one experiment."""
    searcher = HyperparameterSearcher(
        model_type=model_type,
        num_classes=num_classes,
        variance_network_type=cfg.model.variance_network_type,
        search_grid=cfg.training.search_grid,
        device=device,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.early_stopping_patience,
        batch_size=cfg.training.batch_size,
    )
    best_hp = searcher.run(train_ds, val_ds)

    input_dim = train_ds.num_features
    if model_type == "opann":
        model = OPANN(input_dim, best_hp["hidden_size"], num_classes, best_hp["activation"]).to(device)
    else:
        model = HOPANN(
            input_dim, best_hp["hidden_size"], best_hp["hidden_size"],
            num_classes, best_hp["activation"], cfg.model.variance_network_type,
        ).to(device)

    full_ds = td.ConcatDataset([train_ds, val_ds])
    full_loader = td.DataLoader(full_ds, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = td.DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    loss_fn = OrderedProbitNLL()
    trainer = Trainer(device=device, loss_fn=loss_fn)
    trainer.retrain_on_full(model, full_loader, n_epochs=cfg.training.max_epochs, config=cfg.training)

    y_true, y_pred, y_prob = trainer.predict(model, test_loader)
    metrics = OrdinalMetrics().compute(y_true, y_pred, y_prob)

    # Save checkpoint
    ckpt_path = output_dir / f"{model_type}_exp{exp_id}.pt"
    torch.save({"model_state_dict": model.state_dict(), "best_hp": best_hp,
                "input_dim": input_dim, "num_classes": num_classes}, ckpt_path)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.data_path:
        cfg.data.data_path = args.data_path
    cfg.data.seed = args.seed

    ReproducibilityManager.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for exp_id in args.experiments:
        num_classes = EXPERIMENT_NUM_CLASSES[exp_id]
        exp_name = EXPERIMENT_NAMES[exp_id]
        print(f"\n{'='*60}")
        print(f"  Experiment {exp_id}: {exp_name}  (J={num_classes})")
        print(f"{'='*60}")

        cfg.model.num_classes = num_classes
        cfg.data.experiment_id = exp_id

        splitter = ExperimentSplitter(cfg.data.data_path, exp_id, args.seed)
        train_ds, val_ds, test_ds = splitter.get_splits()

        X_train, _ = train_ds.to_numpy()
        scaler = SelectiveStandardScaler()
        scaler.fit(X_train, exclude_col_names=cfg.data.exclude_standardisation_cols)
        train_ds.apply_scaler(scaler)
        val_ds.apply_scaler(scaler)
        test_ds.apply_scaler(scaler)

        exp_results = {}

        # OPANN
        print("\n  [OPANN]")
        exp_results["OPANN"] = run_neural_model(
            "opann", cfg, train_ds, val_ds, test_ds, num_classes, device, output_dir, exp_id, args.seed)
        print(f"    F1-macro: {exp_results['OPANN']['f1_macro']:.4f}")

        # HOPANN
        print("\n  [HOPANN]")
        exp_results["HOPANN"] = run_neural_model(
            "hopann", cfg, train_ds, val_ds, test_ds, num_classes, device, output_dir, exp_id, args.seed)
        print(f"    F1-macro: {exp_results['HOPANN']['f1_macro']:.4f}")

        if not args.skip_baselines:
            X_train_np, y_train_np = train_ds.to_numpy()
            X_val_np, y_val_np = val_ds.to_numpy()
            X_test_np, y_test_np = test_ds.to_numpy()
            X_total = np.vstack([X_train_np, X_val_np])
            y_total = np.concatenate([y_train_np, y_val_np])

            baselines = {
                "Ordered Probit": OrderedProbitBaseline(num_classes=num_classes),
                "ANN": ANNBaseline(num_classes=num_classes, seed=args.seed),
                "SVM": SVMBaseline(num_classes=num_classes, seed=args.seed),
                "RF": RandomForestBaseline(num_classes=num_classes, seed=args.seed),
                "XGB": XGBoostBaseline(num_classes=num_classes, seed=args.seed),
            }
            metrics_calc = OrdinalMetrics()
            for name, bmodel in baselines.items():
                print(f"\n  [{name}]")
                try:
                    bmodel.fit(X_total, y_total)
                    y_pred = bmodel.predict(X_test_np)
                    y_prob = bmodel.predict_proba(X_test_np)
                    m = metrics_calc.compute(y_test_np, y_pred, y_prob)
                    exp_results[name] = m
                    print(f"    F1-macro: {m['f1_macro']:.4f}")
                except Exception as e:
                    exp_results[name] = {"error": str(e)}
                    print(f"    ERROR: {e}")

        all_results[f"Experiment_{exp_id}"] = exp_results

    # Aggregate and save
    print(f"\n\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for exp_key, exp_res in all_results.items():
        print(f"\n{exp_key}:")
        df = pd.DataFrame(exp_res).T
        print(df[["f1_macro", "accuracy", "pr_auc", "mse", "mae"]].to_string())

    out_json = output_dir / "all_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    out_csv = output_dir / "all_results.csv"
    rows = []
    for exp_key, exp_res in all_results.items():
        for model_name, metrics in exp_res.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                row = {"experiment": exp_key, "model": model_name}
                row.update(metrics)
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
