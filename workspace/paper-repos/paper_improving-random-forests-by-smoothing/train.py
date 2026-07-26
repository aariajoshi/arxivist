"""
Training entrypoint for Improving Random Forests by Smoothing.
"""
import argparse
import os
import torch
from improving_rf_smoothing.utils.config import load_config, set_seed
from improving_rf_smoothing.models.random_forest import TreeEnsemble
from improving_rf_smoothing.models.smoothed_rf import SmoothedRandomForest
from improving_rf_smoothing.training.trainer import SmoothingTrainer
from improving_rf_smoothing.data.dataset import TabularDataset

def main():
    parser = argparse.ArgumentParser(description="Train smoothed random forest")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer steps, small data)")
    parser.add_argument("--dry-run", action="store_true", help="Build all components but don't train")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed, config.hardware.get('deterministic', False))
    
    print(f"Loaded config from {args.config}")
    print(f"Model variant: {config.model.variant}")
    
    if args.dry_run:
        print("Dry run: Skipping data loading.")
    else:
        pass
        # dataset = TabularDataset(config.data.data_dir + "/train.csv")
    
    # 2. Train Base Random Forest
    tree_ensemble = TreeEnsemble(config.model)
    print("Initialized base TreeEnsemble")
    
    # 3. Initialize Smoothed Random Forest (needs fitted tree_ensemble)
    # smoothed_rf = SmoothedRandomForest(config.model, tree_ensemble)
    
    if args.dry_run:
        print("Dry run completed successfully.")
        return
        
    print("WARNING: Full training loop requires data and is not fully implemented in this stub.")
    
    print("Done.")

if __name__ == "__main__":
    main()
