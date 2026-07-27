# Benchmark Comparison Report
**Paper**: Physics-Guided Motion Loss for Video Generation Model  
**Paper ID**: arxiv_2506_02244  
**arXiv**: https://arxiv.org/abs/2506.02244  
**Comparison Date**: 2025-07-25  
**ArXivist SIR Version**: 1 (confidence: 0.89)  
**Stage**: Pre-training baseline (no user experimental results submitted yet)

---

## Status: Awaiting User Experimental Results

No experimental results have been submitted to this comparator yet. This report
documents the **full comparison scaffold** — paper targets, deviation thresholds,
root-cause decision trees, and the pre-computed hallucination audit — so that
when you run training and paste your metrics, the comparator can produce an
instantaneous, rigorous reproducibility score.

**To use this comparator:** run your training, then open an ArXivist session and
paste your metric outputs in one of these formats:

```json
{
  "backbone": "open_sora",
  "metrics": {
    "Action_Recognition_Score": 68.5,
    "Motion_Accuracy_Score": 47.0,
    "Warping_Error": 0.0061,
    "Text_Video_Alignment": 59.8
  },
  "training_steps": 40000,
  "config_changes": "none"
}
```

---

## Paper Target Metrics (Ground Truth from SIR)

All values are from Tables 1a, 1b, 2 of arXiv 2506.02244v2.
Evaluation protocol: EvalCrafter on OpenVID-1M (official implementation).

### Open-Sora Backbone

| Metric | Baseline | **Paper (Ours)** | Improvement | Direction |
|--------|----------|-----------------|-------------|-----------|
| Action Recognition Score ↑ | 60.77 | **69.71** | +14.7% | Higher better |
| Motion Accuracy Score ↑ | 44.00 | **49.00** | +11.4% | Higher better |
| Warping Error ↓ | 0.0089 | **0.0056** | −37.1% | Lower better |
| Text-Video Alignment ↑ | 54.02 | **61.05** | +13.0% | Higher better |
| VQA A ↑ | 65.15 | 69.20 | +6.2% | Higher better |
| VQA T ↑ | 59.57 | 69.71 | +17.0% | Higher better |
| CLIP Temporal Score ↑ | 99.80 | 99.85 | +0.05 | Higher better |
| Temporal Consistency ↑ | 61.45 | 63.82 | +3.9% | Higher better |
| Flow Score ↑ | 1.15 | 1.18 | +2.6% | Higher better |
| BLIP-BLEU ↑ | 23.73 | 24.52 | +3.3% | Higher better |

### MVDIT Backbone

| Metric | Baseline | **Paper (Ours)** | Improvement | Direction |
|--------|----------|-----------------|-------------|-----------|
| Action Recognition Score ↑ | 62.34 | **69.70** | +11.8% | Higher better |
| Motion Accuracy Score ↑ | 44.00 | **51.00** | +15.9% | Higher better |
| Warping Error ↓ | 0.0080 | **0.0062** | −22.5% | Lower better |
| Flow Score ↑ | 1.01 | **1.22** | +20.8% | Higher better |
| CLIP Temporal Score ↑ | 99.83 | 99.89 | +0.06 | Higher better |
| Text-Video Alignment ↑ | 61.04 | 63.62 | +4.2% | Higher better |

### Hunyuan (LoRA) Backbone

| Metric | Baseline | **Paper (Ours)** | Improvement | Direction |
|--------|----------|-----------------|-------------|-----------|
| Action Recognition Score ↑ | 68.93 | **73.15** | +6.1% | Higher better |
| Motion Accuracy Score ↑ | 56.00 | **59.00** | +5.4% | Higher better |
| Warping Error ↓ | 0.0024 | **0.0016** | −33.3% | Lower better |
| Text-Video Alignment ↑ | 59.60 | **65.34** | +9.6% | Higher better |
| VQA T ↑ | 85.14 | 87.37 | +2.6% | Higher better |

### User Study (2AFC, N=106, 15 trials/participant)

| Backbone | Quality/Motion Preference | Text Alignment Preference |
|----------|--------------------------|--------------------------|
| Open-Sora | **79.4%** for ours | **74.2%** for ours |
| MVDIT | **82.7%** for ours | **77.9%** for ours |

---

## Deviation Severity Thresholds

When you submit your results, each metric will be classified as:

| Band | Threshold | Interpretation |
|------|-----------|----------------|
| **Excellent** | ≤ 2% | Within training variance — reproduction successful |
| **Good** | 2–5% | Minor deviation — likely explained by seed/hardware |
| **Moderate** | 5–15% | Implementation differences likely — check config |
| **Significant** | 15–30% | Probable implementation error or major config mismatch |
| **Critical** | > 30% | Fundamental issue — wrong architecture, data, or metric |

---

## Pre-Training Root Cause Decision Tree

For any deviation classified **Moderate or worse**, work through this checklist in order:

### 1. Training Convergence
- Did you complete all **4 epochs** as specified? (Sec 4.1)
- Check: loss curve should be monotonically decreasing by epoch 2
- Fix: run remaining epochs; use `--resume` flag

### 2. Config Mismatches (High-Risk Assumed Values)

| Parameter | Paper Value | SIR Confidence | Action if Different |
|-----------|-------------|----------------|---------------------|
| Optimizer | **ASSUMED: AdamW** | 0.65 | Try Lion or Adam without WD |
| LoRA rank (Hunyuan) | **ASSUMED: 16** | < 0.60 | Sweep {4, 8, 16, 32} |
| Physics weight | 0.1 (constant) | 0.78 | Try cosine-annealed from 0.05→0.15 |
| Temporal window T | **ASSUMED: 16** | 0.72 | Try T=12 |
| Stop-grad on weights | **ASSUMED: False** | 0.70 | Try True |
| τ_E energy gate | 0.10 | 0.90 | Verify Table 5 value |

### 3. Data Mismatch
- Are you using the **official OpenVID-1M split**? (not a random subset)
- Is preprocessing (H=256, W=256, T=16) matching the config?
- Is EvalCrafter run from its **official implementation** (not a reimplementation)?

### 4. Implementation Stubs
- **Backbone loaders are stubs** — confirm `_load_open_sora()` etc. are correctly implemented
- Verify the backbone's `forward()` returns `{"loss": ..., "x0_hat": ...}` correctly
- Check that `x0_hat` is reconstructed from the denoised output at every timestep `t`

### 5. Metric Computation
- VQA A/T, SD Score, Action Recognition, BLIP-BLEU must use **official EvalCrafter**
- Do NOT compute these metrics manually — results will not be comparable
- Warping Error requires RAFT checkpoint — verify it is the same checkpoint

### 6. Hardware/Precision
- Paper uses 4× A100 (BF16 backbone, FP32 spectral) — single-GPU runs may differ
- Verify `mixed_precision_backbone: bf16` and `mixed_precision_spectral: fp32` in config
- If running on < 80GB VRAM, batch size reduction may affect BatchNorm statistics

---

## Reproducibility Score Formula

$$
\text{score} = \max\left(0,\; \bar{s} - \underbrace{(1 - \bar{c}_{\text{SIR}}) \times 0.15}_{\text{SIR penalty}} - \underbrace{\frac{n_{\text{unmatched}}}{n_{\text{total}}} \times 0.20}_{\text{coverage penalty}}\right)
$$

where $\bar{s}$ = mean(1 − min(|%dev| / 50, 1)) over matched pairs,  
$\bar{c}_{\text{SIR}}$ = mean SIR confidence (0.89 for this paper),  
giving a baseline SIR confidence penalty of **(1 − 0.89) × 0.15 = 0.017**.

**Pre-training score: N/A** — no user results submitted.  
**Expected score range if within 5% of paper:** 0.82–0.90 (after SIR penalty).

---

*This report will be automatically populated when you submit experimental results.*  
*See `verification_log.md` for the full audit trail.*
