"""
Compute BLEU and perplexity on test sets using beam search.
"""

import argparse
from src.transformer.utils.config import load_config, set_seed

def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed()
    print("Evaluating...")

if __name__ == "__main__":
    main()
