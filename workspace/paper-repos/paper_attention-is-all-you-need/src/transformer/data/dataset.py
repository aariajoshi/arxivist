"""
Dataset definitions.
"""
import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    """
    STUB: Translation dataset.
    SIR ambiguity: Handling variable sequence lengths and batching by token count.
    Replace this stub before training.
    """
    def __init__(self):
        pass
        
    def __len__(self):
        return 0
        
    def __getitem__(self, idx):
        raise NotImplementedError("See docstring — component requires manual implementation")
