# Architecture Plan — Physics-Guided Motion Loss
**Paper**: arXiv 2506.02244v2  
**Plan Version**: 1  
**Generated**: 2025-07-25

---

## Framework

**PyTorch ≥ 2.1.0 + CUDA 11.8+**  
All three backbone models (Open-Sora, MVDIT, Hunyuan) are PyTorch-native. LoRA via HuggingFace PEFT. Spectral blocks forced to FP32 regardless of backbone precision (autocast disabled). Multi-GPU via `accelerate`.

---

## Module Map

```
src/physics_motion_loss/
├── __init__.py                         ← exports PhysicsMotionLoss
│
├── spectral/
│   ├── fft_utils.py                    ← SpectralProcessor, PolarLUT
│   └── gates.py                        ← EnergyGate, ObservabilityGate
│
├── losses/
│   ├── translation_loss.py             ← TranslationalMotionLoss (WLS plane fit)
│   ├── rotation_loss.py                ← RotationalMotionLoss (ring entropy + tilted-line)
│   ├── scaling_loss.py                 ← ScalingMotionLoss (flow alignment + centroid trend)
│   ├── adaptive_composite.py           ← AdaptiveMotionLoss (softmax weighting)
│   ├── physics_motion_loss.py          ← PhysicsMotionLoss (top-level orchestrator)
│   └── flow_baseline.py                ← FlowConsistencyLoss (RAFT ablation baseline)
│
├── training/
│   ├── trainer.py                      ← PhysicsGuidedTrainer
│   └── lora_wrapper.py                 ← LoRAWrapper (Hunyuan PEFT)
│
├── data/
│   ├── dataset.py                      ← OpenVIDDataset, MotionComplexityFilter
│   └── transforms.py                   ← video preprocessing
│
├── evaluation/
│   └── metrics.py                      ← EvalCrafterMetrics wrapper
│
└── utils/
    ├── config.py                        ← PhysicsLossConfig (OmegaConf schema)
    └── precision.py                     ← FP32Context manager
```

---

## Critical Design Decisions

### 1. FP32 Enforcement
All spectral processing and WLS solvers run in FP32 even when backbone uses BF16. Implemented via `FP32Context` context manager that disables `torch.autocast` for the physics loss forward pass.

### 2. Fully Differentiable WLS
The ridge WLS solver uses `torch.linalg.solve` (not `numpy`) so gradients flow back through `β̂` into the spectrum and ultimately into model weights. Pseudo-inverse fallback for ill-conditioned cases.

### 3. PolarLUT Precomputation
The polar coordinate LUT mapping `(ρk, θℓ) → (ωx, ωy)` is precomputed once at `__init__` and cached. This is critical for performance — bilinear interpolation on-the-fly per batch would be prohibitive.

### 4. Channel-Wise + Batch Loop
RGB channels processed independently; energies summed (per App A.1). Inner loop over batch items (not fully batched) because spectral shapes depend on T,H,W which may vary.

### 5. LoRA Swappability
`LoRAWrapper` is isolated from the physics loss entirely — it wraps the backbone before training begins. The physics loss is backbone-agnostic.

---

## Key Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input video | `[B, 3, T, H, W]` | x̂₀ from diffusion |
| Per-channel | `[T, H, W]` | Inner loop |
| Spectrum | `[T, H, W]` complex | After 3D FFT + Hann |
| Low-pass cube | `[⌊0.3T⌋, ⌊0.3H⌋, ⌊0.3W⌋]` | 2.7% coefficients |
| Polar | `[Nr=20, M=24, T_lp]` | Angular harmonics |
| Ring energies | `[Nr=20, T_lp]` | Summed over θ |
| Angular harmonics C̃ₘ | `[Nr, M, T_lp]` complex | Angular+temporal DFT |
| Losses | `[3]` scalars | L_trans, L_rot, L_scale |
| Weights | `[3]` | Softmax over -L/τ |
| ℒ_motion | scalar | Added to denoising loss |

---

## Config Template (abbreviated)

```yaml
spectral:
  low_pass_rho: 0.3          # keeps 2.7% coefficients
  Nr_rings: 20
  M_angular_bins: 24
  Nxi_log_radius_bins: 24
  delta_tolerance: 1         # temporal-freq bins
  temporal_window: hann

losses:
  softmax_temperature: 0.1
  physics_loss_weight: 0.1
  ridge_lambda: 1.0e-3
  numeric_epsilon: 1.0e-8

training:
  learning_rate: 2.0e-5      # cosine annealed
  epochs: 4
  backbone: open_sora        # {open_sora, mvdit, hunyuan}
  mixed_precision_backbone: bf16
  mixed_precision_spectral: fp32   # always FP32

# ASSUMED: AdamW — not stated in paper (confidence 0.65)
# ASSUMED: T=16 window — range 12-16 given (confidence 0.72)  
# ASSUMED: physics_loss_weight constant (confidence 0.78)
```

---

## Entrypoints

| Script | Purpose |
|--------|---------|
| `train.py` | Main fine-tuning (all backbones) |
| `evaluate.py` | EvalCrafter metrics on checkpoint |
| `inference.py` | Single-prompt generation |
| `scripts/ablation.py` | No-trans/no-rot/no-scale/flow-only runs |
| `scripts/visualize_spectrum.py` | Debug: plot FFT, rings, loss branches |

---

## Risk Summary

| ID | Severity | Issue |
|----|----------|-------|
| R1 | 🔴 High | LoRA rank/target modules unspecified |
| R2 | 🔴 High | Optimizer not named (AdamW assumed) |
| R3 | 🟡 Medium | Polar resampling εinterp uncalibrated |
| R4 | 🟡 Medium | T window ambiguous (12–16) |
| R5 | 🟡 Medium | Stop-grad on adaptive weights unspecified |
| R6 | 🟡 Medium | Motion complexity LLM rubric not published |
| R7 | 🟢 Low | EvalCrafter separate install required |
| R8 | 🟢 Low | RAFT checkpoint needed for flow baseline |
| R9 | 🟢 Low | OpenVID-1M multi-TB dataset size |

**Mitigations**: All High/Medium risks addressed via config flags, abstract base classes, or ablation scripts. No blocking issues — proceed to Stage 4.

---

*Next: Stage 4 — Code Generator will implement all modules above as production-quality Python.*
