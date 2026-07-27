# Improving Random Forests by Smoothing

Implementation of the paper "Improving Random Forests by Smoothing" (Ziyi Liu, Phuc Luong, Mario Boley, Daniel F. Schmidt).

DOI: 10.1007/s10994-026-07077-z

## Description
Random forest regression is a powerful non-parametric method that adapts to local data characteristics through data-driven partitioning, making it effective across diverse application domains. However, the piecewise constant nature of random forest predictions means each partition is predicted independently, ignoring potential smoothness in the underlying function. Particularly in the small data regime, this lack of information sharing across the input space can lead to suboptimal performance. This repository introduces probabilistic kernel smoothing as a general approach to equip any learned prediction function with an arbitrary degree of smoothness.

## Quick Start

```bash
git clone <repository-url>
cd <repository>
pip install -e .
python train.py --dry-run
```

## Installation

### Pip
```bash
pip install -r requirements.txt
pip install -e .
```

### Conda
```bash
conda env create -f environment.yaml
conda activate improving_rf_smoothing
pip install -e .
```

## Training

```bash
python train.py --config configs/config.yaml
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/model.pt
```

## Expected Results (from paper)

| Dataset | Metric | Value |
|---------|--------|-------|
| CCPP    | MSE    | 30.88 |
| CCPP    | PIMSE  | 12.39 |

## Reproducibility Notes
- Grid search used for `max_depth` optimization is assumed as it wasn't explicitly stated.
- Integrating over piecewise constant tree regions is non-trivial and may require batched processing of trees and leaves.
- Fallback to scikit-learn defaults for unspecified hyperparameters.

## Citation
```bibtex
@article{Liu2025,
  title={Improving Random Forests by Smoothing},
  author={Ziyi Liu and Phuc Luong and Mario Boley and Daniel F. Schmidt},
  journal={Machine Learning},
  year={2025},
  doi={10.1007/s10994-026-07077-z}
}
```
