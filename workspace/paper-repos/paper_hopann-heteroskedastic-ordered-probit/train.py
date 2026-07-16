"""
train.py — Main training entrypoint for OPANN and HOPANN.

Implements: hyperparameter grid search on validation set, final retraining on
full training data, checkpoint saving.

Paper: Jeong (2024) Heteroskedastic Ordered Probit Models with an ANN.
       Computational Economics, Springer.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure src/ is on path when run from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hopann.utils.config import Config
from hopann.utils.reproducibility import ReproducibilityManager
from hopann.data.dataset import ExperimentSplitter
from hopann.data.transforms import SelectiveStandardScaler
from hopann.models.opann import OPANN
from hopann.models.hopann import HOPANN
from hopann.training.trainer import Trainer, HyperparameterSearcher
from hopann.training.losses import OrderedProbitNLL
from hopann.evaluation.metrics import OrdinalMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OPANN or HOPANN on Amazon Software Product Reviews."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model type: opann or hopann.")
    parser.add_argument("--experiment", type=int, default=None,
                        help="Override experiment ID (1-4).")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override path to dataset CSV.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: cuda or cpu.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip hyperparameter search; use config values directly.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: reduce dataset and steps for quick testing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build all components but skip training.")
    return parser.parse_args()


def build_model(cfg: Config, input_dim: int, device: torch.device) -> torch.nn.Module:
    """Instantiate the configured model type."""
    model_type = cfg.model.model_type.lower()
    if model_type == "opann":
        model = OPANN(
            input_dim=input_dim,
            hidden_size=cfg.model.hidden_size_mean,
            num_classes=cfg.model.num_classes,
            activation=cfg.model.activation,
        )
    elif model_type == "hopann":
        model = HOPANN(
            input_dim=input_dim,
            hidden_size_mean=cfg.model.hidden_size_mean,
            hidden_size_variance=cfg.model.hidden_size_variance,
            num_classes=cfg.model.num_classes,
            activation=cfg.model.activation,
            variance_network_type=cfg.model.variance_network_type,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'opann' or 'hopann'.")
    return model.to(device)


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.model:
        cfg.model.model_type = args.model
    if args.experiment:
        cfg.data.experiment_id = args.experiment
    if args.data_path:
        cfg.data.data_path = args.data_path
    if args.output_dir:
        cfg.evaluation.output_dir = args.output_dir
    if args.seed:
        cfg.data.seed = args.seed
    if args.device:
        cfg.hardware.device = args.device

    # Reproducibility
    ReproducibilityManager.seed_everything(
        seed=cfg.data.seed,
        deterministic=getattr(cfg.hardware, "deterministic", False),
    )

    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HOPANN Training — {cfg.model.model_type.upper()}")
    print(f"  Experiment: {cfg.data.experiment_id}  |  Device: {device}")
    print(f"  Seed: {cfg.data.seed}  |  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load data
    splitter = ExperimentSplitter(
        data_path=cfg.data.data_path,
        experiment_id=cfg.data.experiment_id,
        seed=cfg.data.seed,
    )
    train_ds, val_ds, test_ds = splitter.get_splits()

    # Fit scaler on training data only (Section 3.1 of paper)
    X_train_raw, _ = train_ds.to_numpy()
    scaler = SelectiveStandardScaler()
    scaler.fit(X_train_raw, exclude_col_names=cfg.data.exclude_standardisation_cols)
    train_ds.apply_scaler(scaler)
    val_ds.apply_scaler(scaler)
    test_ds.apply_scaler(scaler)

    input_dim = train_ds.num_features
    num_classes = cfg.model.num_classes
    print(f"  Input dim K={input_dim}  |  Classes J={num_classes}")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}\n")

    if args.dry_run:
        print("[DRY RUN] Model build successful. Skipping training.")
        model = build_model(cfg, input_dim, device)
        print(f"  Model: {model}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        return

    # Hyperparameter search
    if not args.no_search:
        print("--- Hyperparameter Search ---")
        searcher = HyperparameterSearcher(
            model_type=cfg.model.model_type,
            num_classes=num_classes,
            variance_network_type=cfg.model.variance_network_type,
            search_grid=cfg.training.search_grid,
            device=device,
            max_epochs=cfg.training.max_epochs,
            patience=cfg.training.early_stopping_patience,
            batch_size=cfg.training.batch_size,  # ASSUMED: 64
        )
        best_hp = searcher.run(train_ds, val_ds)
        print(f"\n  Best hyperparameters: {best_hp}")
        cfg.model.hidden_size_mean = best_hp["hidden_size"]
        cfg.model.hidden_size_variance = best_hp.get("hidden_size", cfg.model.hidden_size_variance)
        cfg.model.activation = best_hp["activation"]
        cfg.training.learning_rate = best_hp["learning_rate"]

    # Build final model
    model = build_model(cfg, input_dim, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Final model: {model}")
    print(f"  Parameters: {n_params:,}\n")

    # Retrain on full training data (train + val) per paper methodology
    print("--- Final Training on Full Training Set ---")
    import torch.utils.data as td
    full_train_ds = td.ConcatDataset([train_ds, val_ds])
    full_loader = td.DataLoader(
        full_train_ds,
        batch_size=cfg.training.batch_size,  # ASSUMED: 64
        shuffle=True,
        num_workers=cfg.hardware.num_workers,
    )
    test_loader = td.DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.hardware.num_workers,
    )

    loss_fn = OrderedProbitNLL()
    trainer = Trainer(device=device, loss_fn=loss_fn)
    trainer.retrain_on_full(
        model=model,
        full_loader=full_loader,
        n_epochs=cfg.training.max_epochs,
        config=cfg.training,
    )

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    metrics_calc = OrdinalMetrics()
    y_true, y_pred, y_prob = trainer.predict(model, test_loader)
    results = metrics_calc.compute(y_true, y_pred, y_prob)
    print(f"\n  Results:")
    for k, v in results.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Save checkpoint
    ckpt_path = output_dir / f"{cfg.model.model_type}_exp{cfg.data.experiment_id}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "results": results,
        "input_dim": input_dim,
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # Save results JSON
    results_path = output_dir / f"results_{cfg.model.model_type}_exp{cfg.data.experiment_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
