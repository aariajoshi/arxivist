#!/usr/bin/env python3
"""
Single-sample generation script for Mamba.
"""
import argparse
import torch

from mamba.utils.config import load_config
from mamba.models.mamba import MambaLMHeadModel

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Mamba model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config.get("hardware", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    model_config = config["model"]
    vocab_size = 50257
    model = MambaLMHeadModel(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        n_layer=model_config["n_layer"],
        expand=model_config["expansion_factor"],
        d_state=model_config["state_dimension_N"],
        d_conv=model_config["conv_kernel_size"]
    )
    
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # STUB: Tokenizer not implemented, using fake tokenization
    print("WARNING: Tokenizer not implemented. Using dummy tokens for prompt length.")
    prompt_len = len(args.prompt.split())
    input_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    
    print("Generating...")
    generated_ids = model.generate(input_ids, max_length=args.max_length)
    
    # STUB: Decode
    print("Generation complete. (Output decoding not implemented)")
    print("Generated token sequence shape:", generated_ids.shape)
    print("Generated IDs:", generated_ids[0].cpu().tolist())

if __name__ == "__main__":
    main()
