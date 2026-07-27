"""
Data transformations and collators.
"""
import torch
from typing import List, Dict, Any

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Args:
        batch: List of dictionaries from Dataset
    Returns:
        Batched dictionaries
    """
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
