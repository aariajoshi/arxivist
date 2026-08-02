# Data Requirements for Mamba

The Mamba model is evaluated on several datasets as described in the paper.

## The Pile
- **Split**: Validation
- **Usage**: Primary pre-training and evaluation
- **Instructions**: The Pile is publicly available but very large. You can use the HuggingFace datasets library to download it.

```python
from datasets import load_dataset
dataset = load_dataset("EleutherAI/pile")
```

## HG38, YouTubeMix, SC09
These are also publicly available for specific modalities (DNA, Audio). 

Place downloaded files into `data/raw/` or configure the path in `configs/config.yaml`.
