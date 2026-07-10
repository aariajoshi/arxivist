"""
Translate single sentences from the command line.
"""

import argparse
from src.transformer.utils.config import load_config, set_seed

def main():
    parser = argparse.ArgumentParser(description="Inference with Transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Input sentence to translate")
    
    args = parser.parse_args()
    
    set_seed()
    print(f"Translating: {args.text}")

if __name__ == "__main__":
    main()
