# Benchmark Comparison Report
**Paper**: Heteroskedastic Ordered Probit Models with an Artificial Neural Network
**Paper ID**: paper_hopann-heteroskedastic-ordered-probit
**Comparison Date**: 2026-07-16
**Reproducibility Score**: 0.0 (low confidence)

## Metric Comparison

> [!WARNING]
> **Synthetic Data Used**: The user ran the verification on a **synthetic dataset** of N=200 samples rather than the Amazon Software Review dataset used in the paper. Therefore, empirical metric comparison is not applicable.

| Metric | Dataset | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------------|------------|-----------|----------|
| Accuracy | Synthetic | N/A | 0.2500 | N/A | UNMATCHED |
| NLL Loss | Synthetic | N/A | 1.5469 | N/A | UNMATCHED |

## Summary

The pipeline structure, components, and forward/backward passes have been successfully reproduced and validated on synthetic data. Both the OPANN and HOPANN models compile, accept inputs, strictly enforce cutting-point ordering, properly parameterize the variance network, and exhibit decreasing training loss (1.7290 → 1.6309). Empirical validation against the paper's claims (predicting minority classes on imbalanced data) requires training on the full Amazon Software Reviews dataset.

## Root Cause Analysis

N/A — Differences are due to using random synthetic data.

## Recommended Actions

1. Download the Amazon Software Review dataset and run the full `run_experiments.py` pipeline to get true empirical comparisons.
