"""
Tabular dataset loading.
"""
import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """
    Dataset loader for tabular data.
    """
    def __init__(self, data_path: str):
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Assuming CSV format with target as last column
        df = pd.read_csv(data_path)
        
        self.features = df.iloc[:, :-1].values.astype(np.float32)
        self.targets = df.iloc[:, -1].values.astype(np.float32)
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y
