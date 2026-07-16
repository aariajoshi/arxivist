# Architecture Plan Summary
## Paper: Heteroskedastic Ordered Probit Models with an Artificial Neural Network (HOPANN)
**paper_id:** `paper_hopann-heteroskedastic-ordered-probit`
**Plan Version:** 1 | **Created:** 2026-07-16

---

## 1. Framework Selection

| Decision | Choice | Reasoning |
|---|---|---|
| Primary framework | **PyTorch 2.1+** | Explicitly used in the paper for OPANN and HOPANN |
| Python version | **3.10+** | Modern typing support, stable ecosystem |
| CUDA required | **Yes (11.8+)** | Paper trained on T4 GPU (Google Colab) |
| HuggingFace | No | No Transformer architectures |
| Config management | **Plain YAML** | Small tabular experiment |
| Baselines | Statsmodels (Ordered Probit), scikit-learn (SVM, RF), XGBoost | Explicitly named in paper |

---

## 2. Module Hierarchy

```
project_root/
├── src/
│   └── hopann/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── mean_network.py        ← MeanNetwork (shared OPANN + HOPANN)
│       │   ├── variance_network.py    ← VarianceNetworkBase + ANN + Linear variants
│       │   ├── cutting_points.py      ← CuttingPoints (ordered reparameterisation)
│       │   ├── opann.py               ← OPANN model
│       │   ├── hopann.py              ← HOPANN model
│       │   └── baselines.py           ← All 5 baseline wrappers (unified interface)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py             ← AmazonReviewDataset + ExperimentSplitter
│       │   └── transforms.py          ← SelectiveStandardScaler
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py              ← OrderedProbitNLL
│       │   └── trainer.py             ← EarlyStopping + HyperparameterSearcher + Trainer
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py             ← OrdinalMetrics (7 metrics)
│       └── utils/
│           ├── __init__.py
│           ├── config.py              ← YAML config loader
│           └── reproducibility.py     ← Seed management
├── train.py
├── evaluate.py
├── run_baselines.py
├── run_experiments.py
├── inference.py
├── config.yaml
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
└── Dockerfile
```

---

## 3. Key Design Decisions

### Variance Network — Swappable Strategy (Risk Mitigation)
SIR has **low confidence (0.62)** on whether the HOPANN variance network is a linear layer or a full ANN:
- `VarianceNetworkBase` is an abstract base class
- `VarianceNetworkANN` — hidden-layer version (default)
- `VarianceNetworkLinear` — linear fallback
- `config.yaml` exposes `variance_network_type: ann | linear` to switch without code changes

### CuttingPoints Ordering
Enforced via **cumulative sum + softplus** reparameterisation:
- `c_1` is unconstrained
- `c_k = c_1 + cumsum(softplus(delta_k))` for k > 1
- Avoids sorting (non-differentiable); guarantees strict ordering

### Selective Standardisation
Per the paper: **Asin_verified_share** and **Day dummy columns** are NOT standardised. `SelectiveStandardScaler` fits only on numeric training columns.

### Unified Baseline Interface
All 5 baselines implement the same `BaselineModel` ABC with `fit / predict / predict_proba`, allowing the evaluation loop to treat all models uniformly.

---

## 4. Tensor Flow Summary

| Flow | Input | Output |
|---|---|---|
| OPANN Forward | `[B, K] float32` | `[B, J] float32` class probabilities |
| HOPANN Forward | `[B, K] x [B, K] float32` | `[B, J] float32` heteroskedastic probabilities |
| OrderedProbitNLL | `[B, J] probs + [B] targets` | `[] float32` scalar NLL |
| CuttingPoints | `[J-1] raw params` | `[J-1] ordered cuts` |
| SelectiveStandardScaler | `[N, K] raw` | `[N, K] scaled` |

Key difference OPANN vs HOPANN: dividing `(c_j - f(x))` by `sigma_i = exp(variance_ANN(z_i))`.

---

## 5. Configuration — Key Parameters

| Parameter | Value | SIR Confidence | Note |
|---|---|---|---|
| `hidden_size_mean` (Q) | 16 | ASSUMED | Hyperparameter-searched on val set |
| `hidden_size_variance` (Q_z) | 16 | ASSUMED (0.62) | TODO: verify from paper |
| `activation` | relu | ASSUMED | Searched over {sigmoid, tanh, relu, elu, gelu} |
| `variance_network_type` | ann | ASSUMED (0.62) | Swappable to linear |
| `learning_rate` | 1e-3 | ASSUMED (0.52) | Adam default; searched |
| `batch_size` | 64 | ASSUMED (0.52) | Unspecified in paper |
| `early_stopping_patience` | 15 | ASSUMED (0.52) | Prechelt 2002 |
| `num_classes` (J) | 5 or 3 | HIGH (0.88) | Per experiment config |
| `optimizer` | Adam | HIGH | Explicit in paper |

---

## 6. Entrypoints

| Script | Purpose |
|---|---|
| `train.py` | Train OPANN/HOPANN with hyperparameter search |
| `evaluate.py` | Evaluate checkpoint on test set (all 7 metrics) |
| `run_baselines.py` | Train + evaluate all 5 baselines for one experiment |
| `run_experiments.py` | Full paper reproduction: all 4 experiments × all 7 models |
| `inference.py` | Single-sample or batch inference from checkpoint |

---

## 7. Dependencies

**Runtime:** `torch>=2.1`, `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `xgboost>=2.0`, `statsmodels>=0.14`, `scipy>=1.11`, `pyyaml>=6.0`, `tqdm>=4.65`, `matplotlib>=3.7`, `seaborn>=0.12`

**Dev:** `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy`, `jupyter`, `ipykernel`

**Docker base image:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`

---

## 8. Risk Assessment

| Severity | Risk | Mitigation |
|---|---|---|
| **High** | Training hyperparameters all assumed (batch size, LR, patience) — conf. 0.52 | All in config.yaml with ASSUMED comments |
| **High** | Variance network architecture ambiguous (linear vs ANN) — conf. 0.62 | Both implementations, switchable via config |
| **High** | Dataset exact subset may differ from public data | preprocessing script + data README + synthetic fallback |
| **Medium** | Feature count K unknown (12-19) | Input dim inferred dynamically at runtime |
| **Medium** | Cutting-point initialisation unspecified | Evenly spaced init in [-2, 2]; softplus ordering; NLL clamp |
| **Medium** | Prechelt early stopping variant unspecified | Generalisation-loss variant; configurable patience |
| **Low** | statsmodels API version compatibility | Pinned >=0.14.0; version check in baseline wrapper |
| **Low** | Multi-class ROC/PR AUC averaging unspecified | sklearn OvR macro; documented in metrics.py |

---

## 9. Notes for Code Generator (Stage 04)

1. **Never hardcode K** — infer from `dataset.num_features` at model init time
2. **sigma_i clamping** — clamp sigma_i to `[1e-6, inf]` before dividing to prevent div-by-zero
3. **Boundary probabilities** — handle `Phi(-inf)=0` and `Phi(+inf)=1` explicitly in OPANN/HOPANN forward
4. **Class indexing** — paper uses 1-indexed classes (1 to J); convert to 0-indexed internally
5. **Ordered Probit baseline** — use `statsmodels.miscmodels.ordinal_model.OrderedModel` with `distr='probit'`
