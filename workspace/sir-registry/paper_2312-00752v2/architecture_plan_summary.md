# Architecture Plan Summary: Mamba-3B (paper_2312-00752v2)

## 1. Framework Selection
**Primary Framework:** PyTorch  
**Python Version:** 3.10+  
**CUDA:** Required (Min 11.8)  
**Reasoning:** PyTorch is the standard framework for foundation models. Mamba's core innovation (the hardware-aware S6 scan) typically relies on custom CUDA kernels (`mamba-ssm`, `causal-conv1d`) that are explicitly built for PyTorch. HuggingFace integration will be used for evaluating on standard NLP tasks.

## 2. Module Hierarchy
- `src/mamba/__init__.py`: Root module.
- `src/mamba/models/mamba.py`: Contains `MambaLMHeadModel` (base: `nn.Module`) for language modeling.
- `src/mamba/models/blocks.py`: Core components like `MambaBlock` and `MambaMixer`. Maps to SIR modules: Mamba Block, Input Projections, 1D Convolution, Selective SSM (S6), Nonlinearity, Output Projection.
- `src/mamba/data/dataset.py`: Contains `MambaDataset` for loading The Pile.
- `src/mamba/training/trainer.py`: Contains `Trainer` for managing the optimization loop.

## 3. Tensor Flow Specification
**Mamba Block Forward Pass**
- `x`: `[B, L, D]` (dtype: bfloat16) — input tensor
- `x_expanded = Input_Projections(x)` → `[B, L, E]`
- `conv_out = 1D_Convolution(x_expanded)` → `[B, L, E]`
- `ssm_out = Selective_SSM(conv_out)` → `[B, L, E]`
- `activation_out = Nonlinearity(ssm_out)` → `[B, L, E]`
- `out = Output_Projection(activation_out)` → `[B, L, D]`

## 4. Configuration Schema (config.yaml template)
```yaml
model:
  d_model: 2560
  n_layer: 64
  expansion_factor: 2
  state_dimension_N: 16
  conv_kernel_size: 4  # ASSUMED: Standard practice in H3/Mamba for sequence modeling

training:
  optimizer: AdamW
  learning_rate: 0.002
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1
  batch_size: 1024  # ASSUMED: Batch size scales similarly to GPT-3 / DNA models
  training_steps: 200000
  mixed_precision: bfloat16
  gradient_clipping: 0.1

data:
  dataset_name: The Pile
  seq_length: 2048

evaluation:
  metrics:
    - Perplexity
    - Accuracy

hardware:
  device: cuda
  num_workers: 8
```

## 5. Dependencies Manifest
- **Runtime:** `torch>=2.1.0`, `transformers>=4.36.0`, `mamba-ssm>=1.0.1`, `causal-conv1d>=1.1.0`, `pyyaml>=6.0`
- **Development:** `pytest>=7.0.0`, `black>=23.0.0`

## 6. Entrypoints
- `train.py`: Main training script for Mamba (Requires `--config`).
- `evaluate.py`: Evaluation script for computing metrics like Perplexity (Requires `--checkpoint`).
- `inference.py`: Single-sample generation script (Requires `--prompt`).

## 7. Docker Specification
- **Base Image:** `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel` (needs `devel` for building CUDA extensions).
- **System Dependencies:** `git`, `build-essential`, `ninja-build`.
- **Working Directory:** `/workspace/mamba`
- **Default Command:** `python train.py --config config.yaml`

## 8. Risk Assessment
1. **High Severity:** Hardware-aware parallel algorithm (S6) requires custom CUDA kernels (`mamba-ssm`, `causal-conv1d`). 
   *Mitigation:* Use the official pre-built Python packages instead of implementing the kernels from scratch. Provide a fallback pure PyTorch scan for debugging.
2. **Medium Severity:** Batch size for language modeling is ambiguous in the paper.
   *Mitigation:* Expose batch size and gradient accumulation in the config to allow flexible scaling.
3. **Low Severity:** 1D Depthwise Convolution kernel size is assumed (4).
   *Mitigation:* Parameterize kernel size in the configuration file.
