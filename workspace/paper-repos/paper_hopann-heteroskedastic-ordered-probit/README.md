# HOPANN — Heteroskedastic Ordered Probit Models with an Artificial Neural Network

**Paper:** Jeong, J. (2024). Heteroskedastic Ordered Probit Models with an Artificial Neural
Network. *Computational Economics*, Springer.

**Reproduced by:** ArXivist automated pipeline.

---

## What This Paper Does

This paper introduces **HOPANN** (Heteroskedastic Ordered Probit model with an Artificial Neural
Network), a model for predicting ordinal responses (e.g., 1-5 star ratings) that combines two
key ideas:

1. **Ordered Probit structure** — instead of a softmax output, the ANN feeds into an ordered
   probit log-likelihood, preserving the ordinal nature of the labels and producing calibrated
   class probabilities.

2. **Heteroskedasticity** — unlike the base OPANN model, HOPANN allows each observation to have
   its own variance `sigma_i`, learned by a second ANN. This captures the fact that some reviews
   are inherently harder to predict than others.

The models are evaluated on Amazon Software Product Reviews (star ratings 1-5), with a focus on
**imbalanced datasets**, where the minority class is rarely predicted correctly by standard methods.
HOPANN and OPANN achieve the best F1-macro scores on all three imbalanced experiments.

---

## Quick Start

```bash
# 1. Install
conda env create -f environment.yaml
conda activate hopann
# OR: pip install -r requirements.txt

# 2. Get data
python data/preprocess.py --synthetic --n-samples 500 \
    --output-path data/amazon_software_reviews.csv

# 3. Train HOPANN (Experiment 1: 5-class imbalanced)
python train.py --model hopann --experiment 1

# 4. Evaluate
python evaluate.py --checkpoint outputs/hopann_exp1.pt --experiment 1

# 5. Full reproduction (all 4 experiments x all 7 models)
python run_experiments.py --data-path data/amazon_software_reviews.csv
```

---

## Installation

### pip
```bash
pip install -r requirements.txt
pip install -e .
```

### conda
```bash
conda env create -f environment.yaml
conda activate hopann
```

### Docker
```bash
cd docker
docker-compose build
docker-compose run train
```

---

## Training

```bash
# HOPANN, Experiment 1 (5-class imbalanced), with hyperparameter search
python train.py --model hopann --experiment 1 --data-path data/amazon_software_reviews.csv

# OPANN, Experiment 4 (balanced), skip hyperparameter search
python train.py --model opann --experiment 4 --no-search

# Resume from checkpoint
python train.py --model hopann --experiment 1 --resume outputs/hopann_exp1.pt

# Debug mode (reduced data, fast)
python train.py --model hopann --experiment 1 --debug
```

**Experiment IDs:**
| ID | Name | Classes | Balance |
|----|------|---------|---------|
| 1 | 5-class | 1-2-3-4-5 star | Imbalanced |
| 2 | 3-class | pos(4+5) / neutral(3) / neg(1+2) | Imbalanced |
| 3 | Modified 3-class | pos(5) / mild(2+3+4) / neg(1) | Imbalanced |
| 4 | Modified 3-class (balanced) | Same as 3 | Balanced (undersampled) |

---

## Evaluation

```bash
python evaluate.py --checkpoint outputs/hopann_exp1.pt --experiment 1
```

**Metrics computed:** F1-macro, Accuracy, PR AUC, MSE, MAE, Cohen's Kappa, ROC AUC.

---

## Baselines

```bash
# Run all 5 baselines for Experiment 1
python run_baselines.py --experiment 1

# Full paper reproduction
python run_experiments.py --data-path data/amazon_software_reviews.csv
```

---

## Expected Results

From the paper (qualitative summary):

| Metric | Imbalanced (Exp 1-3) | Balanced (Exp 4) |
|--------|----------------------|------------------|
| Best F1-macro | **HOPANN / OPANN** | Ordered Probit / ANN |
| Best Accuracy | RF / XGB | **RF** |
| Best PR AUC / MSE / MAE | **XGB** | XGB |
| Minority class prediction | Only HOPANN & OPANN succeed | All models comparable |

> **Note:** Exact numerical values from the paper's tables are not reproduced here because
> the original results tables were not fully parseable from the PDF. See `sir-registry/sir.json`
> for the SIR's qualitative result annotations.

---

## Model Architecture

### OPANN
```
x [B, K] → Hidden Layer [B, Q] (activation: σ) → Output [B, 1] → f(x, θ)
                                                                        ↓
                                               Cutting Points [J-1]: c₁ < c₂ < ... < c_{J-1}
                                                                        ↓
P(y=j|x) = Φ(cⱼ - f(x)) - Φ(c_{j-1} - f(x))     [standard normal CDF]
```

### HOPANN (extends OPANN)
```
x [B, K] → Mean ANN → f(x, θ) [B, 1]
z [B, K] → Variance ANN → log σᵢ [B, 1] → σᵢ = exp(log σᵢ)

P(y=j|x,z) = Φ((cⱼ - f(x))/σᵢ) - Φ((c_{j-1} - f(x))/σᵢ)
```

**Key equations** (LaTeX, from SIR):
- Latent index: `y*_i = f(x_i, θ) + ε_i`, `ε_i ~ N(0, σᵢ²)`
- Hidden layer: `h_q = σ(Σ_k w_{qk} x_{ik} + b_q)`
- Latent output: `f(x_i, θ) = Σ_q v_q h_q + c`
- HOPANN variance: `σ_i = exp(z_i γ)` (or ANN variant)
- Training objective: `ln LL(θ) = Σ_i Σ_j 𝟙(y_i=j) ln P(y_i=j|x_i)`

---

## Implementation Assumptions

The following values were **not explicitly stated** in the paper and were assumed:

| Parameter | Assumed Value | Confidence | Note |
|-----------|---------------|------------|------|
| `batch_size` | 64 | Low (0.52) | Unspecified in paper |
| `learning_rate` | 1e-3 | Low (0.52) | Adam default; searched |
| `early_stopping_patience` | 15 | Low (0.52) | Prechelt (2002) criterion |
| `hidden_size` (Q) | 16 | Low | Hyperparameter-searched |
| `variance_network_type` | ann | Medium (0.62) | Ambiguous in paper |
| `z_i = x_i` | True | Medium (0.55) | Not distinguished from x_i |

All assumptions are marked with `# ASSUMED:` comments in `configs/config.yaml` and in the
relevant source files.

---

## Reproducibility Notes

1. **Variance network ambiguity:** The paper states `σ_i = exp(z_i γ)` but also describes
   a "separate ANN" for the variance. This is implemented with a configurable flag:
   `configs/config.yaml → model.variance_network_type: ann | linear`. Default is `ann`.

2. **Cutting point ordering:** Enforced via cumulative sum + softplus reparameterisation
   (not described in paper). Initialised at evenly-spaced values in [-2, 2].

3. **Dataset:** The exact 6,173-review subset used in the paper requires downloading the
   Amazon Software category from the McAuley Lab and applying the paper's time filtering.
   See `data/README_data.md` for full instructions.

4. **Day dummies:** Interpreted as day-of-week (6 dummies, Saturday = base). The paper
   mentions "day" as a time-fixed effect but does not specify granularity.

---

## Citation

```bibtex
@article{jeong2024hopann,
  title   = {Heteroskedastic Ordered Probit Models with an Artificial Neural Network},
  author  = {Jeong, Jikhan},
  journal = {Computational Economics},
  year    = {2024},
  publisher = {Springer}
}
```

---

## Project Structure

```
paper-repos/
├── src/hopann/
│   ├── models/
│   │   ├── mean_network.py        MeanNetwork (shared OPANN + HOPANN)
│   │   ├── variance_network.py    VarianceNetworkANN / Linear (swappable)
│   │   ├── cutting_points.py      Ordered cutting points (softplus+cumsum)
│   │   ├── opann.py               OPANN model
│   │   ├── hopann.py              HOPANN model
│   │   └── baselines.py           5 baselines (unified interface)
│   ├── data/
│   │   ├── dataset.py             AmazonReviewDataset + ExperimentSplitter
│   │   └── transforms.py          SelectiveStandardScaler
│   ├── training/
│   │   ├── losses.py              OrderedProbitNLL
│   │   └── trainer.py             EarlyStopping + HyperparameterSearcher + Trainer
│   ├── evaluation/metrics.py      OrdinalMetrics (7 metrics)
│   └── utils/
│       ├── config.py              YAML config loader
│       └── reproducibility.py     Seed management
├── train.py                       Main training entrypoint
├── evaluate.py                    Evaluation entrypoint
├── run_baselines.py               Run all 5 baselines
├── run_experiments.py             Full paper reproduction
├── inference.py                   Single-sample inference
├── configs/config.yaml            Hyperparameters (with confidence annotations)
├── data/
│   ├── README_data.md             Dataset download instructions
│   └── preprocess.py              Feature engineering script
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```
