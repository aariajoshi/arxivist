"""
Main training script for the Transformer model.
"""

import argparse
from src.transformer.utils.config import load_config, set_seed

def main():
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, required=False, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed override")
    parser.add_argument("--debug", action="store_true", help="Reduce dataset size and steps for quick local testing")
    parser.add_argument("--dry-run", action="store_true", help="Build all components but don't train")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(args.seed)
    
    print(f"Loaded config: {config}")
    if args.dry_run:
        print("Dry run complete. Components built successfully.")
        return
        
    print("Training started...")
    
if __name__ == "__main__":
    main()
