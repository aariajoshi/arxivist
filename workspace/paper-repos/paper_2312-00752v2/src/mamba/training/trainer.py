"""
Training loop and optimization for Mamba.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class Trainer:
    """
    Handles the training loop, checkpointing, and logging.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 train_loader: Any, val_loader: Any, config: Dict[str, Any],
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.training_config = config.get("training", {})
        self.max_steps = self.training_config.get("training_steps", 200000)
        self.grad_clip = self.training_config.get("gradient_clipping", 0.1)
        self.log_every_n_steps = self.training_config.get("log_every_n_steps", 100)
        self.save_every_n_steps = self.training_config.get("save_every_n_steps", 1000)
        self.checkpoint_dir = "checkpoints"
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, step: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pt")
            torch.save(state, best_path)
            
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return starting step."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["step"]
        
    def evaluate(self) -> float:
        """Run evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                count += 1
                if count >= 50: # Limit eval for stub
                    break
        return total_loss / max(1, count)
        
    def train(self, start_step: int = 0):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps...")
        # Summaries
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params / 1e6:.2f} M")
        
        self.model.to(self.device)
        step = start_step
        best_val_loss = float("inf")
        
        # Determine mixed precision type
        amp_dtype = torch.bfloat16 if self.training_config.get("mixed_precision") == "bfloat16" else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
        
        train_iter = iter(self.train_loader)
        
        while step < self.max_steps:
            self.model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.device.type == "cuda"):
                logits = self.model(input_ids)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
            scaler.scale(loss).backward()
            
            if self.grad_clip > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            scaler.step(self.optimizer)
            scaler.update()
            
            step += 1
            
            if step % self.log_every_n_steps == 0:
                print(f"Step {step}/{self.max_steps} | Loss: {loss.item():.4f}")
                
            if step % self.save_every_n_steps == 0:
                val_loss = self.evaluate()
                print(f"Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {torch.exp(torch.tensor(val_loss)):.4f}")
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                self.save_checkpoint(step, val_loss, is_best)
