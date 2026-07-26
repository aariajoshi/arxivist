"""
Evaluation entrypoint.
"""
import argparse
import torch
from improving_rf_smoothing.utils.config import load_config, set_seed

def main():
    parser = argparse.ArgumentParser(description="Evaluate smoothed random forest")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Evaluating with checkpoint {args.checkpoint}")
    
    # TODO: Load model, load test data, evaluate using metrics from config.evaluation.metrics
    print("STUB: Evaluation loop not fully implemented.")

if __name__ == "__main__":
    main()
