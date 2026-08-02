"""
evaluate.py — Evaluation entrypoint. Loads a saved checkpoint and computes
all 7 metrics (F1-macro, Accuracy, PR AUC, MSE, MAE, Cohen's Kappa, ROC AUC)
on the test set.

Paper: Jeong (2024) Heteroskedastic Ordered Probit Models with an ANN.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hopann.utils.config import Config
from hopann.utils.reproducibility import ReproducibilityManager
from hopann.data.dataset import ExperimentSplitter
from hopann.data.transforms import SelectiveStandardScaler
from hopann.models.opann import OPANN
from hopann.models.hopann import HOPANN
from hopann.training.losses import OrderedProbitNLL
from hopann.training.trainer import Trainer
from hopann.evaluation.metrics import OrdinalMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved OPANN/HOPANN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .pt checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=int, default=None, help="Experiment ID 1-4.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.experiment:
        cfg.data.experiment_id = args.experiment
    if args.data_path:
        cfg.data.data_path = args.data_path
    if args.output_dir:
        cfg.evaluation.output_dir = args.output_dir
    if args.device:
        cfg.hardware.device = args.device

    ReproducibilityManager.seed_everything(cfg.data.seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_cfg_dict = ckpt.get("config", {})
    input_dim = ckpt["input_dim"]
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Saved config: {saved_cfg_dict.get('model', {})}\n")

    # Load data
    splitter = ExperimentSplitter(
        data_path=cfg.data.data_path,
        experiment_id=cfg.data.experiment_id,
        seed=cfg.data.seed,
    )
    train_ds, val_ds, test_ds = splitter.get_splits()
    X_train_raw, _ = train_ds.to_numpy()
    scaler = SelectiveStandardScaler()
    scaler.fit(X_train_raw, exclude_col_names=cfg.data.exclude_standardisation_cols)
    test_ds.apply_scaler(scaler)

    import torch.utils.data as td
    test_loader = td.DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    # Rebuild model from saved config
    model_type = saved_cfg_dict.get("model", {}).get("model_type", cfg.model.model_type)
    num_classes = saved_cfg_dict.get("model", {}).get("num_classes", cfg.model.num_classes)
    hidden_mean = saved_cfg_dict.get("model", {}).get("hidden_size_mean", cfg.model.hidden_size_mean)
    hidden_var = saved_cfg_dict.get("model", {}).get("hidden_size_variance", cfg.model.hidden_size_variance)
    activation = saved_cfg_dict.get("model", {}).get("activation", cfg.model.activation)
    var_type = saved_cfg_dict.get("model", {}).get("variance_network_type", cfg.model.variance_network_type)

    if model_type == "opann":
        model = OPANN(input_dim, hidden_mean, num_classes, activation).to(device)
    else:
        model = HOPANN(input_dim, hidden_mean, hidden_var, num_classes, activation, var_type).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate
    loss_fn = OrderedProbitNLL()
    trainer = Trainer(device=device, loss_fn=loss_fn)
    y_true, y_pred, y_prob = trainer.predict(model, test_loader)

    metrics_calc = OrdinalMetrics()
    results = metrics_calc.compute(y_true, y_pred, y_prob)

    print("Test Set Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(args.checkpoint).stem
    results_path = output_dir / f"eval_{ckpt_stem}_exp{cfg.data.experiment_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
