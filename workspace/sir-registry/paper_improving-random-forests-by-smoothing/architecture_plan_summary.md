# Architecture Plan Summary: Improving Random Forests by Smoothing

## 1. Framework Selection
**Primary Framework:** PyTorch & scikit-learn
**Reasoning:** PyTorch is selected to optimize the smoothing parameters (`\Sigma, \beta_0, \beta_1`) via gradient descent with AdamW using the out-of-bag loss. The base Random Forest is integrated via scikit-learn, with tree structures parsed into PyTorch tensors for efficient parallel evaluation and integration over leaf hyper-rectangles.

## 2. Module Hierarchy
- `src/improving_rf_smoothing/models/random_forest.py`: Base Random Forest wrapper extracting tree structures.
- `src/improving_rf_smoothing/models/kernels.py`: Kernel functions (Gaussian, Hyperbolic Secant) and their CDF implementations.
- `src/improving_rf_smoothing/models/smoothed_rf.py`: Main `SmoothedRandomForest` module with probabilistic smoothing logic.
- `src/improving_rf_smoothing/data/dataset.py`: Tabular dataset loaders.
- `src/improving_rf_smoothing/training/losses.py`: Out-of-bag loss formulation.
- `src/improving_rf_smoothing/training/trainer.py`: AdamW training loop for the smoothing parameters.

## 3. Tensor Flows
**Probabilistic Smoothing Forward Pass**
- **Input:** `x [B, p]` (float32)
- **Output:** `y_tilde [B, 1]` (float32)
- **Steps:**
  1. Initialize output tensor `f_tilde_sum = 0` for aggregating smoothed tree predictions.
  2. For each tree `t` in 1 to `T`:
  3.   Retrieve the tree's leaf regions (bounding boxes) and corresponding prediction values.
  4.   For each leaf region:
  5.     Compute the probability mass of the kernel `k(z|x, Sigma_t)` within the leaf region using the kernel's CDF.
  6.     Multiply the probability mass by the leaf's prediction value.
  7.   Sum these weighted values to get the smoothed tree prediction `f_tilde_t(x)`.
  8.   Add `f_tilde_t(x)` to `f_tilde_sum`.
  9. Compute ensemble average: `f_tilde(x) = f_tilde_sum / T`.
  10. Apply re-calibration: `y_tilde = beta_1 * f_tilde(x) + beta_0`.
  11. Return `y_tilde`.

## 4. Configuration Template (`config.yaml`)
```yaml
model:
  variant: "EST-PD"
  n_estimators: 100
  max_features: 0.8
  max_depth: 15 # ASSUMED: Grid used for CV optimization of max_depth is not specified. Assuming 15 as typical default. (Confidence 0.7)
  kernel: "gaussian"

training:
  optimizer: "AdamW"
  learning_rate: 0.01
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.001
  lr_schedule: "OneCycleLR"
  epochs: 100

data:
  dataset_name: "CCPP"
  split_ratio: 0.8

evaluation:
  metrics: ["MSE", "PIMSE", "MAD", "MaxAbsoluteError"]

hardware:
  device: "cuda"
  num_workers: 4
```

## 5. Dependencies Manifest
- **Runtime:** `torch >= 2.0.0`, `scikit-learn >= 1.2.0`, `pandas >= 2.0.0`, `omegaconf >= 2.3.0`
- **Development:** `pytest >= 7.0.0`, `black >= 23.0.0`

## 6. Entrypoints
- `train.py`: CLI for training base RF and optimizing smoothing parameters.
- `evaluate.py`: CLI for evaluating the trained smoothed RF on test sets.

## 7. Docker Specification
- **Base Image:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- **System Dependencies:** `build-essential`
- **Default Command:** `python train.py`

## 8. Risk Assessment
1. **High Severity:** Evaluating the integral of the kernel over piecewise constant tree regions is non-trivial. **Mitigation:** Extract tree structures into PyTorch tensors for leaf bounding boxes. Use factorized kernels to compute integrals as products of 1D CDF differences.
2. **Medium Severity:** Ambiguity in `max_depth` optimization grid. **Mitigation:** Expose `max_depth` grid in config for hyperparameter search.
3. **Medium Severity:** Memory and computational cost of OOB loss optimization for EST-PD variant. **Mitigation:** Implement batched processing of OOB samples and trees to prevent OOM errors on GPUs.
