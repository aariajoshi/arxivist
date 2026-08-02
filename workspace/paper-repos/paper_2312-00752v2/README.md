# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Authors**: Albert Gu, Tri Dao
**arXiv**: [2312.00752v2](https://arxiv.org/abs/2312.00752v2)

## Overview
This repository is an unofficial implementation of the Mamba architecture. Foundation models are almost universally based on the Transformer architecture. Mamba introduces a subquadratic-time architecture using structured state space models (SSMs) with parameters that are functions of the input. This allows content-based reasoning. Mamba combines SSM and MLP blocks into a simplified block without attention, using a hardware-aware parallel algorithm.

## Quick Start
```bash
git clone <this-repo>
cd mamba
pip install -e .
python train.py --config configs/config.yaml --dry-run
python train.py --config configs/config.yaml
```

## Installation

### pip
```bash
pip install -r requirements.txt
pip install -e .
```

### conda
```bash
conda env create -f environment.yaml
conda activate mamba_env
```

## Usage

### Training
```bash
python train.py --config configs/config.yaml
```
For debugging on a small subset:
```bash
python train.py --config configs/config.yaml --debug
```

### Evaluation
```bash
python evaluate.py --checkpoint checkpoints/model_best.pt
```

## Expected Results
Based on the paper's reported results on The Pile (Validation):
| Metric | Value |
|--------|-------|
| Perplexity | 6.22 |

## Implementation Assumptions
- The exact batch size for language modeling wasn't fully specified; we default to 1024, configurable in `config.yaml`.
- 1D Depthwise Convolution kernel size is assumed to be 4 (standard practice).

## Reproducibility Notes
- **Hardware constraints**: Uses `mamba-ssm` and `causal-conv1d` which require CUDA. A pure PyTorch fallback is provided in `SelectiveSSM` but is very slow.
- Random seeds are tied across PyTorch, NumPy, and Python natively via `mamba.utils.config.set_seed`.
- Tokenizer implementations are stubbed as standard huggingface tokenizers can be swapped in.

## Citation
```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752v2},
  year={2023}
}
```
