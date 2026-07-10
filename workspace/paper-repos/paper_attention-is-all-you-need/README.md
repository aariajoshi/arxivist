# Attention Is All You Need (Transformer)

**Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

This repository is a reproduction of the "Attention Is All You Need" paper, implementing the Transformer architecture from scratch in PyTorch. The Transformer is the first sequence transduction model based entirely on attention, dispensing with recurrence and convolutions to achieve high parallelization and state-of-the-art translation quality.

## Quick Start
```bash
git clone https://github.com/your-username/transformer.git
cd transformer
pip install -e .
python data/download.py
python train.py --config configs/config.yaml
```

## Installation
```bash
# Pip
pip install -r requirements.txt

# Conda
conda env create -f environment.yaml
conda activate transformer-env
```

## Training
To train the model:
```bash
python train.py --config configs/config.yaml
```

## Evaluation
To evaluate the model (computes BLEU and perplexity):
```bash
python evaluate.py --config configs/config.yaml --checkpoint path/to/checkpoint.pt
```

## Expected Results
| Metric | Dataset | Split | Value |
|--------|---------|-------|-------|
| BLEU | WMT 2014 English-German | test | 28.4 |
| BLEU | WMT 2014 English-French | test | 41.0 |

## Reproducibility Notes
- **Attention Masking**: Masking uses `-1e9` instead of `-inf` to avoid NaN gradients in half-precision or when exponentiated.
- **Custom LR Schedule**: Custom `NoamLRScheduler` is implemented as it is coupled strictly with `d_model`.
- **Token-based Batching**: The paper batches by token count rather than sequence count, requiring a custom sampler.
- **Vocabulary Size**: The vocabulary size is assumed to be 37000 for subwords.

## Citation
```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```
