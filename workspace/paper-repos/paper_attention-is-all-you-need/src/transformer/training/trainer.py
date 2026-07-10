"""
Training loop and custom learning rate scheduler.
Paper Section: 5.3
"""

import torch
import torch.nn as nn

class NoamLRScheduler:
    """
    Custom learning rate scheduler from Section 5.3.
    
    lrate = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})
    """
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self) -> None:
        self.step_num += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
            
    def get_lr(self) -> float:
        return (self.d_model ** (-0.5)) * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5)))

class Trainer:
    """
    STUB: This component implements the training loop.
    SIR ambiguity: Handling variable sequence lengths and batching by token count.
    Replace this stub before training.
    """
    def __init__(self, model, optimizer, scheduler, criterion, train_dataloader, val_dataloader, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
    def train(self):
        raise NotImplementedError("See docstring — component requires manual implementation")
        
    def train_epoch(self, epoch: int) -> float:
        raise NotImplementedError("See docstring — component requires manual implementation")
