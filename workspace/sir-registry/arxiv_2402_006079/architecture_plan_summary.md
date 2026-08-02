# Architecture Plan Summary — DiscDiff (arxiv_2402_006079)

**Paper type:** latent diffusion **model** + **algorithm** (Absorb-Escape) + **dataset** (EPD-GenDNA).

## Reproduction strategy (QOSI) — two tiers

- **Tier 1 (fully reproduced, CPU, no weights):** Absorb-Escape (Sec 4.2, Table 3, Fig 4) — a
  deterministic refinement over softmax confidences. Verified mechanically with synthetic
  'valleys': repairs TATT->TATA exactly, no-op when confident, local.
- **Tier 2 (built + structurally verified):** the DiscDiff LDM (two-stage CNN-VAE + latent U-Net
  + DDPM eps-objective, Eq 2). Forward pass, finite CE+KL/MSE loss, one reverse step run on tiny
  CPU configs. Generation metrics (S-FID/CorTATA) are weight/Sei/data-gated (Colab).

## Verified mechanically (9/9 tests)

absorb fires iff L(D)<T_absorb · natural escape iff L(D)>L(M) · **Fig-4 valley repair** ·
no-op safety · locality · VAE round-trip shapes+loss · U-Net eps + DDPM step · CorTATA=1 /
Delta-Div=0 · S-FID gated->None.

## Gated (disclosed)

S-FID 3.21/4.38 + CorTATA on 50k samples need trained DiscDiff+Hyena weights + Sei + EPD-GenDNA.

**Score: 0.83.**
