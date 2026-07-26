"""
Inference entrypoint for a single sample or batch without labels.
"""
import argparse
import torch
from improving_rf_smoothing.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to input features CSV/JSON")
    args = parser.parse_args()

    print("STUB: Inference not fully implemented.")

if __name__ == "__main__":
    main()
