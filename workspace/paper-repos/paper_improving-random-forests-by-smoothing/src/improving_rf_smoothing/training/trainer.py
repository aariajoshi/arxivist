"""
Trainer for optimizing smoothing parameters.
"""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Any
import numpy as np

from ..models.smoothed_rf import SmoothedRandomForest
from .losses import OOBLoss

class SmoothingTrainer:
    """
    Trainer for optimizing smoothing parameters using OOB loss.
    """
    def __init__(self, model: SmoothedRandomForest, config: Any):
        self.model = model
        self.config = config
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        self.criterion = OOBLoss()
        
    def train(self) -> None:
        """
        Run optimization on OOB samples.
        """
        epochs = self.config.epochs
        self.model.train()
        
        print(f"Starting training for {epochs} epochs...")
        # WARNING: low-confidence implementation for exact OOB extraction and batching
        
        # TODO: Extract OOB samples for each tree t and optimize
        # For each epoch:
        #   optimizer.zero_grad()
        #   loss = 0
        #   For each tree t:
        #     get OOB samples X_oob, Y_oob for tree t
        #     f_tilde_t = model.compute_smoothed_tree(X_oob, t)
        #     y_pred = model.beta_1 * f_tilde_t + model.beta_0
        #     loss += criterion(y_pred, Y_oob)
        #   loss.backward()
        #   optimizer.step()
        
        print("Training completed.")
