#!/usr/bin/env python3
"""
Main training script for Mamba.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from mamba.utils.config import load_config, set_seed
from mamba.models.mamba import MambaLMHeadModel
from mamba.data.dataset import MambaDataset
from mamba.data.transforms import collate_fn
from mamba.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba model")
    parser.add_argument("--config", type=str, required=True, default="configs/config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed override")
    parser.add_argument("--debug", action="store_true",
                        help="Reduce dataset size and steps for quick local testing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build all components but do not train (validates setup)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup reproducibility
    set_seed(args.seed, deterministic=config.get("training", {}).get("deterministic", False))
    
    device_name = config.get("hardware", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model_config = config["model"]
    # Vocabulary size 50257 (GPT-2 tokenizer size as common standard)
    vocab_size = 50257
    model = MambaLMHeadModel(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        n_layer=model_config["n_layer"],
        expand=model_config["expansion_factor"],
        d_state=model_config["state_dimension_N"],
        d_conv=model_config["conv_kernel_size"]
    )
    
    # Data
    data_config = config["data"]
    train_dataset = MambaDataset(
        data_path=data_config.get("data_path", "data/raw"),
        seq_length=data_config["seq_length"],
        vocab_size=vocab_size
    )
    val_dataset = MambaDataset(
        data_path=data_config.get("data_path", "data/raw"),
        seq_length=data_config["seq_length"],
        vocab_size=vocab_size
    )
    
    # Reduce size for debug
    if args.debug:
        train_dataset._length = 100
        val_dataset._length = 10
        config["training"]["training_steps"] = 10
        config["training"]["log_every_n_steps"] = 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"]
    )
    
    # Optimizer
    train_config = config["training"]
    # Defaulting to AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(train_config["beta1"], train_config["beta2"]),
        weight_decay=train_config["weight_decay"]
    )
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
        print(f"Resumed from step {start_step}")
        
    if args.dry_run:
        print("Dry run complete. Components built successfully.")
        return
        
    trainer.train(start_step=start_step)

if __name__ == "__main__":
    main()
