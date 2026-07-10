# Benchmark Comparison Report
**Paper**: Attention Is All You Need
**Paper ID**: paper_attention-is-all-you-need
**Comparison Date**: 2026-07-11
**Reproducibility Score**: 0.0 (low confidence)

## Metric Comparison

| Metric | Dataset | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------------|------------|-----------|----------|
| BLEU | WMT 2014 English-German test | 28.4 | UNMATCHED | N/A | Critical |
| BLEU | WMT 2014 English-French test | 41.0 | UNMATCHED | N/A | Critical |

## Summary

The provided user results were from a mini-training loop on a toy dataset, completing 5 epochs and achieving a final loss of 3.4514. Since the full training was not run on the WMT datasets and no BLEU scores were provided, there are no matched metrics to compare against the paper's reported values. Consequently, reproducibility cannot be assessed, resulting in an unmatched penalty and a base reproducibility score of 0.0.

## Root Cause Analysis

- **BLEU En-De (Critical)**:
  - Cause 1 (High): **Training convergence**. The user only ran a mini-training loop on a toy dataset for 5 epochs. The paper trains for 100,000 steps on a massive corpus (WMT 2014).
    - Suggested Fix: Run the full training loop on the WMT 2014 English-German dataset for the specified 100,000 steps.
- **BLEU En-Fr (Critical)**:
  - Cause 1 (High): **Training convergence**. Full evaluation was bypassed because the required dataset was not used.
    - Suggested Fix: Same as above; ensure the WMT 2014 English-French dataset is used and run the complete training.

## Recommended Actions

1. Acquire and preprocess the WMT 2014 English-German and/or English-French datasets.
2. Run the full training loop (`train.py`) for the specified 100,000 steps as detailed in the configuration.
3. Perform evaluation (`evaluate.py`) using beam search (size 4, length penalty 0.6) to generate test set BLEU scores.
4. Resubmit the BLEU scores for a complete comparison.
