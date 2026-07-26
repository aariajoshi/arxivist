# Hallucination Report — D3LM (arxiv_2603_001780)

**Comparison Date**: 2026-07-25
**SIR version**: 1
**Scope**: verified algorithm (Part A) + executed generation (Part B). SFID/GC gated on real EPD-GenDNA + Sei.

## Summary

| Type | Count | Critical | Notes |
|------|-------|----------|-------|
| Structural | 0 | 0 | masked-diffusion objective (Eq 2) + sampler (Eq 4) match the paper |
| Parametric | 0 | 0 | T=50, tau=1.1, 6-mer, 1/t weighting all taken verbatim |
| Omission | 0 | 0 | — |

## Structural (0)

Re-derived from the paper and checked mechanically:

- **Backbone** (Sec 2.2): NT-v2 (ESM encoder + RoPE + SwiGLU), bidirectional attention, 12 layers /
  512 hidden / 16 heads for the 50M variant. Loaded from the official HF weights (not re-implemented).
- **Forward masking** (Sec 2.1): each token → `[M]` with probability t ~ U[0,1]. ✅
- **Training loss** (Eq 2): `L = -E[ (1/t) Σ 1[xt=M] log p(x0|xt) ]` — CE on masked positions only,
  weighted by 1/t. ✅ **Verified: exact 2× ratio between t=0.25 and t=0.5.**
- **Reverse denoising / generation** (Eq 4): iterative unmask of an all-`[M]` sequence over T steps,
  temperature scaling (Eq 5), random unmask order (Sec 2.3). ✅ **Oracle predictor recovers target exactly.**
- **Tokenization** (Sec 3.1): non-overlap 6-mer, 4096 + 9 special = 4105. ✅

## Parametric (0)

All generation/training hyperparameters are taken verbatim from the paper: T=50 denoising steps
(Table 4), temperature τ=1.1 (Table 5), non-overlap 6-mer (best in Fig 3b), random unmask order
(Table 3), 1/t loss weighting (Eq 2), AdamW β=(0.9,0.95) wd 0.01 lr 8e-5 (Sec 3.1). No value fabricated.

## Omission (0)

The reproduction covers the masked-diffusion objective, the sampler, both variants (D3LM / D3LM-R),
and the generation-metric suite. Generative **pretraining** (A800 cluster) is intentionally not
reproduced — infeasible — and the repo loads the released weights instead (documented in README, not
an omitted claim). SFID is implemented but gated on Sei availability; the cheaper real metrics
(GC ratio, diversity, novelty, motif correlation) are always available.

## Notes

- Generation was executed (Colab, 2026-07-25): official D3LM-R weights load and the paper's own
  `diffusion_generate` sampler runs end-to-end (100×2048bp). The measured GC/diversity/novelty are
  reported verbatim and explicitly labeled as against a *synthetic* reference; SFID is `None` (no Sei).
  Nothing is fabricated — the paper's real SFID/GC are gated on real EPD-GenDNA + the Sei CNN, and that
  gap is stated plainly rather than filled with a guessed number.
- The *algorithm* — the paper's actual contribution — is fully verified on CPU with zero hallucinations.
- transformers is pinned `<4.46` because D3LM ships custom `diffusion_generate` remote code (same class
  of issue seen with Caduceus/DNAGPT); a from-scratch reverse-unmasking fallback is provided.
