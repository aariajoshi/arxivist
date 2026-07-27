#!/usr/bin/env python3
"""
Evaluation script for computing Perplexity and other metrics.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from mamba.utils.config import load_config
from mamba.models.mamba import MambaLMHeadModel
from mamba.data.dataset import MambaDataset
from mamba.data.transforms import collate_fn
from mamba.evaluation.metrics import compute_perplexity, compute_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mamba model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
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
    
    data_config = config["data"]
    val_dataset = MambaDataset(
        data_path=data_config.get("data_path", "data/raw"),
        seq_length=data_config["seq_length"],
        vocab_size=vocab_size
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            acc = compute_accuracy(logits, labels)
            
            total_loss += loss.item()
            total_acc += acc
            count += 1
            if count >= 20: # Limit for stub
                break
                
    avg_loss = total_loss / count
    perplexity = compute_perplexity(avg_loss)
    avg_acc = total_acc / count
    
    print(f"Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Accuracy: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
