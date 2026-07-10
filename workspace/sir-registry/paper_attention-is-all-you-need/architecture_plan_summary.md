# Architecture Plan Summary: Attention Is All You Need

This document outlines the software architecture required to implement the "Attention Is All You Need" (Transformer) model, based on its Scientific Intermediate Representation (SIR).

## 1. Framework & Environment
- **Primary Framework**: PyTorch (Python 3.10+)
- **Hardware**: CUDA (min 11.8) recommended. The paper trained on 8 P100 GPUs.
- **Rationale**: PyTorch is the standard choice for NLP and offers excellent transparency for implementing explicit custom attention ops without relying on heavy abstractions like HuggingFace's `transformers` (which we explicitly avoid to ensure an exact scratch-built paper reproduction).

## 2. Module Hierarchy
The codebase is structured under `src/transformer/`:
- `models/transformer.py`: Contains the overarching `Transformer`, `Encoder`, and `Decoder` modules.
- `models/attention.py`: Contains the `MultiHeadAttention` and `ScaledDotProductAttention` mechanisms.
- `models/layers.py`: Contains `PositionwiseFeedForward` and `PositionalEncoding`.
- `training/trainer.py`: Encapsulates the training loop and the paper-specific `NoamLRScheduler`.

## 3. Data Flow
**Full Forward Pass**:
1. Source tokens `[B, T_src]` are embedded and combined with sinusoidal positional encodings.
2. Passed through `N=6` Encoder layers with self-attention to produce `memory` `[B, T_src, d_model]`.
3. Target tokens `[B, T_tgt]` are embedded, position-encoded, and passed through `N=6` Decoder layers.
4. The decoder applies masked self-attention, then cross-attention over `memory`.
5. Outputs are projected via a linear layer to `[B, T_tgt, vocab_size]` logits.

## 4. Configuration Schema
```yaml
# config.yaml (Template)
model:
  N: 6 
  d_model: 512 
  d_ff: 2048 
  h: 8 
  d_k: 64 
  d_v: 64 
  P_drop: 0.1 
  vocab_size: 37000 # ASSUMED: Approximate size for WMT En-De subword vocabulary (Confidence < 0.7)

training:
  optimizer: Adam 
  beta1: 0.9 
  beta2: 0.98 
  eps: 1e-9 
  warmup_steps: 4000 
  train_steps: 100000 
  label_smoothing: 0.1 # ASSUMED: Mentioned in paper section 5.4, standard for this architecture (Confidence < 0.7)

data:
  dataset: "WMT 2014 English-German"
  batch_size_tokens: 4096

evaluation:
  metrics: ["BLEU", "perplexity"]
  beam_size: 4
  length_penalty_alpha: 0.6
```

## 5. Entrypoints
- `train.py`: Main CLI for training the model using the config YAML.
- `evaluate.py`: Generates translations on the test set and calculates corpus BLEU/Perplexity via Beam Search.
- `inference.py`: Helper script to translate a single text string using a trained checkpoint.

## 6. Implementation Risks & Mitigations
- **Attention Masking (Low Risk)**: The SIR notes masking illegal connections to $-\infty$. We will use `-1e9` to prevent NaN errors, particularly if mixed precision is used.
- **Custom LR Schedule (Medium Risk)**: The learning rate formula is tightly coupled to $d_{model}$ and warmup steps. We will implement a custom `NoamLRScheduler` class.
- **Token-based Batching (Medium Risk)**: The paper batches data by ~4096 tokens rather than raw sequence count. We will need a custom `BatchSampler` in the PyTorch `DataLoader` to group similar length sequences.
