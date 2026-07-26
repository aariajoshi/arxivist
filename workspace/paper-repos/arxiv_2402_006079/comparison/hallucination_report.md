# Hallucination Report — DiscDiff (arxiv_2402_006079)

**Comparison Date**: 2026-07-26
**SIR version**: 1
**Scope**: Absorb-Escape fully verified (Part A) + DiscDiff LDM structurally verified (Part B).
Generation metrics cited, weight-gated.

## Summary

| Type | Count | Critical | Notes |
|------|-------|----------|-------|
| Structural | 0 | 0 | Absorb-Escape (Table 3 + Fig 4) + LDM (two-stage VAE, latent U-Net, Eq 2) match |
| Parametric | 0 | 0 | T_absorb / escape conditions / beta / DDPM eps-objective taken verbatim |
| Omission | 0 | 0 | generation metrics disclosed as weight-gated, not omitted |

## Structural (0)

Re-derived from the paper and checked mechanically:

- **Absorb-Escape** (Sec 4.2, Fig 4): scan → absorb at low L(D) → AR-regenerate → escape →
  replace. ✅ verified: repairs TATT→TATA, no-op when confident, local.
- **Absorb / escape conditions** (Table 3): A(s_i)=[L(D,s_i)<T_absorb]; natural escape
  E(s_j)=[L(D,s_j)>L(M,s_j)]; plus threshold / random / max-length variants. ✅ all implemented.
- **Two-stage VAE** (Sec 4.1.1): s→z1 (2D)→z (3D) and symmetric decode; CE recon + KL (Sec 4.1.2). ✅
- **Latent denoiser** (Sec 4.1.3, Eq 2): eps-prediction U-Net with species+time cross-attention. ✅
- **DDPM** forward/reverse in latent space; frozen decoder at generation (Fig 3). ✅

## Parametric (0)

Absorb threshold T_absorb, the four escape conditions, VAE beta (KL weight), the eps-prediction
objective (Eq 2), and the CNN-VAE choice (best generation, Table 4) are all taken verbatim from
the paper. No value fabricated.

## Omission (0)

Two items are deliberately not reproduced and both are disclosed (so neither is hidden):
1. **Generation metrics** (S-FID 3.21/4.38, CorTATA on 50k) — need trained DiscDiff + Hyena +
   Sei + EPD-GenDNA. LDM structure is built; numbers labeled gated.
2. **VAE/denoiser training** (72 A100 GPU-hours) — infeasible; we instantiate the architecture
   and verify its forward pass instead.

## Notes

- Absorb-Escape — the paper's most portable idea — is **fully reproduced and measured** on
  synthetic confidence profiles with zero hallucinations. The valley→repair behavior is exactly
  as Fig 4 depicts.
- S-FID is gated on the Sei model (identical situation to the D3LM repo). The cheap real metrics
  (CorTATA, Delta-Div) run without it.
