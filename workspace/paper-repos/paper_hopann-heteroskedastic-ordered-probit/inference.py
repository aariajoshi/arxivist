"""
inference.py — Single-sample or small-batch inference from a saved OPANN or
HOPANN checkpoint. Prints predicted ordinal class and class probabilities.

Paper: Jeong (2024) Heteroskedastic Ordered Probit Models with an ANN.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hopann.utils.config import Config
from hopann.models.opann import OPANN
from hopann.models.hopann import HOPANN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference from a saved OPANN/HOPANN checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .pt checkpoint.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV with feature columns (one or more rows).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cuda or cpu (default cpu for inference).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    input_dim = ckpt["input_dim"]
    num_classes = ckpt.get("num_classes", cfg.model.num_classes)
    saved_cfg = ckpt.get("config", {}).get("model", {})

    model_type = saved_cfg.get("model_type", cfg.model.model_type)
    hidden_mean = saved_cfg.get("hidden_size_mean", cfg.model.hidden_size_mean)
    hidden_var = saved_cfg.get("hidden_size_variance", cfg.model.hidden_size_variance)
    activation = saved_cfg.get("activation", cfg.model.activation)
    var_type = saved_cfg.get("variance_network_type", cfg.model.variance_network_type)

    best_hp = ckpt.get("best_hp", {})
    if best_hp:
        hidden_mean = best_hp.get("hidden_size", hidden_mean)
        activation = best_hp.get("activation", activation)

    # Build and load model
    if model_type == "opann":
        model = OPANN(input_dim, hidden_mean, num_classes, activation).to(device)
    else:
        model = HOPANN(input_dim, hidden_mean, hidden_var, num_classes, activation, var_type).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load input features
    df = pd.read_csv(args.input)
    X = df.values.astype(np.float32)
    assert X.shape[1] == input_dim, (
        f"Input has {X.shape[1]} features, model expects {input_dim}."
    )

    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        if model_type == "opann":
            probs = model(x_tensor)
        else:
            probs = model(x_tensor, x_tensor)  # z = x by default

    probs_np = probs.cpu().numpy()
    preds_np = probs_np.argmax(axis=1) + 1  # Convert 0-indexed to 1-indexed class

    print(f"\nInference results ({len(preds_np)} sample(s)):")
    print(f"{'Sample':<8} {'Predicted Class':<18} {'Class Probabilities'}")
    print("-" * 70)
    for i, (pred, prob) in enumerate(zip(preds_np, probs_np)):
        prob_str = "  ".join([f"P(y={j+1})={p:.3f}" for j, p in enumerate(prob)])
        print(f"{i+1:<8} {pred:<18} {prob_str}")


if __name__ == "__main__":
    main()
