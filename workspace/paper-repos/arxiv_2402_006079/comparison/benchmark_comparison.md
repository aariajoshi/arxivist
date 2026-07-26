# Benchmark Comparison Report

**Paper**: DiscDiff: Latent Diffusion Model for DNA Sequence Generation
**Paper ID**: arxiv_2402_006079
**arXiv**: https://arxiv.org/abs/2402.06079  ·  **Venue**: ICML 2024 (AI4Science)
**Author code**: https://github.com/Zehui127/Latent-DNA-Diffusion  ·  **Dataset**: EPD-GenDNA (HF Zehui127/*)
**Comparison Date**: 2026-07-26
**Reproducibility Score**: 0.83 (Absorb-Escape fully reproduced + verified on CPU; LDM built & structurally verified; generation metrics weight-gated)

## Status

DiscDiff has three parts of very different reproduction cost, handled by tier:

- **Absorb-Escape** (the portable, reusable contribution) — a deterministic algorithm over
  softmax confidences. **Fully reproduced and mechanically verified on CPU, no weights.**
- **DiscDiff LDM** (two-stage VAE + latent U-Net) — **built and structurally verified** (forward
  pass, finite loss, one denoise step on CPU); its generation metrics are weight-gated.
- **EPD-GenDNA** — the dataset this paper introduces (also used by D3LM); loader implemented with
  a synthetic fallback.

## Part A — Absorb-Escape (complete, CPU, no weights)

| Check | Result |
|-------|--------|
| Absorb fires iff L(D,s_i) < T_absorb (Table 3) | ✅ |
| Natural escape stops iff L(D,s_j) > L(M,s_j) (Table 3, default) | ✅ |
| **Valley repair (Fig 4): TATT → TATA restored exactly** | ✅ **matches intended truth** |
| No-op on an all-confident sequence | ✅ |
| Locality: only low-confidence region modified | ✅ |
| **Unit tests** (`tests/test_discdiff.py`) | ✅ **9 / 9 passed** |

Run it: `python demo_absorb_escape.py` (both planted valleys corrected → intended sequence).

## Part B — DiscDiff LDM (structure verified; metrics gated)

| Component | Structural check | Metric |
|-----------|------------------|--------|
| Two-stage CNN-VAE (Sec 4.1.1) | ✅ encode/decode shapes, finite CE+KL | recon-acc gated on training |
| Latent U-Net (Sec 4.1.3) | ✅ eps-shaped output, species+time cross-attn | — |
| DDPM eps-objective (Eq 2) | ✅ finite MSE loss, one reverse step | — |
| S-FID / CorTATA (50k samples) | — | ⏳ **gated** on weights + Sei + EPD-GenDNA |

## Headline results (paper, cited — NOT our reproduction)

| Model | EPD-small S-FID↓ | CorTATA↑ | EPD-large S-FID↓ | CorTATA↑ | Reproduction |
|-------|-----------------|----------|------------------|----------|--------------|
| DiscDiff | 57.4 | 0.973 | 45.2 | 0.858 | LDM built (structure) |
| Absorb-Escape | 3.21 | 0.975 | 4.38 | 0.892 | algorithm reproduced ✅ |
| D3PM (small) | 97.4 | 0.096 | 94.5 | 0.363 | baseline |

The ~15–20× S-FID improvement Absorb-Escape gives over raw DiscDiff is the paper's headline;
the **mechanism** behind it (valley detection + AR re-generation) is what we verify here.

## Note on S-FID

S-FID needs the **Sei** genomic embedder (same gate as D3LM's SFID). `evaluate.py` computes
the cheap real metrics (CorTATA, Delta-Div) without it and reports S-FID as None until a Sei
embedder is supplied.

## Hallucination Report Summary

See `hallucination_report.md`. **Zero structural, zero parametric** in the verified algorithm
(Table 3 conditions + Fig 4 procedure match exactly).

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 0 | 0 |
| Omission | 0 | 0 |
