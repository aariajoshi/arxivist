# DiscDiff — Latent Diffusion Model for DNA Sequence Generation (reproduction)

Reproduction of **Li et al., "DiscDiff: Latent Diffusion Model for DNA Sequence
Generation"** (ICML 2024 AI4Science), [arXiv:2402.06079](https://arxiv.org/abs/2402.06079).
Author code: <https://github.com/Zehui127/Latent-DNA-Diffusion> · Dataset: EPD-GenDNA
(HuggingFace `Zehui127/*`).

## Two contributions, two reproduction tiers

DiscDiff has three parts — the **DiscDiff LDM** (two-stage VAE + latent U-Net), the
**Absorb-Escape** refinement algorithm, and the **EPD-GenDNA** dataset. They differ sharply
in reproduction cost, so we treat them honestly by tier:

### Tier 1 — Absorb-Escape: fully reproduced, CPU, no weights ✅

Absorb-Escape (Sec 4.2, Table 3, Fig 4) is a **deterministic** post-processing algorithm over
per-position softmax confidences. It needs no trained model to verify — we drive it with
synthetic confidence 'valleys' and prove it repairs them exactly (Fig 4's **TATT → TATA**):

```bash
pip install -e .
pytest tests/ -q            # 9 tests: Absorb-Escape repair/no-op + LDM structure + metrics
python demo_absorb_escape.py
```

```
DiscDiff (raw)   : CGCGCATATTCGCGATCTAAA
low-conf valleys :         ^       ^
Absorb-Escape    : CGCGCATATACGCGATATAAA     <- both valleys corrected
```

### Tier 2 — DiscDiff LDM: built & structurally verified, metrics weight-gated ⏳

The two-stage **CNN-VAE** (`models/vae.py`), latent **U-Net** denoiser (`models/unet.py`),
and **DDPM** objective (`models/ddpm.py`, Eq 2) are implemented and run on CPU at tiny configs
(forward pass, finite CE+KL loss, one reverse step — all tested). Their **generation metrics**
(S-FID 3.21/4.38, CorTATA on 50k samples) need trained DiscDiff weights + fine-tuned Hyena +
the Sei model + EPD-GenDNA — a Colab/GPU run. `evaluate.py` computes CorTATA/Delta-Div with no
extra model; S-FID is gated on a Sei embedder.

## Headline numbers (paper, for reference)

| Model | EPD-small S-FID↓ | CorTATA↑ | EPD-large S-FID↓ | CorTATA↑ |
|-------|------------------|----------|------------------|----------|
| DiscDiff | 57.4 | 0.973 | 45.2 | 0.858 |
| **Absorb-Escape** | **3.21** | **0.975** | **4.38** | **0.892** |
| best diffusion baseline (D3PM-s) | 97.4 | 0.096 | 94.5 | 0.363 |

Absorb-Escape's ~15–20× S-FID drop over raw DiscDiff is the paper's key generation result.

## Layout

```
src/discdiff/
  refine/absorb_escape.py   # Absorb-Escape algorithm (Table 3 + Fig 4)  <- verified core
  refine/conditions.py      # absorb + 4 escape conditions (Table 3)
  models/vae.py             # two-stage CNN-VAE (Sec 4.1.1) + CE/KL loss
  models/unet.py            # latent U-Net denoiser, species+time cross-attn (Sec 4.1.3)
  models/ddpm.py            # DDPM schedule + eps-objective (Eq 2) + sampling
  data/epd_gendna.py        # EPD-GenDNA loader (HF) + loud synthetic fallback
  evaluation/metrics.py     # CorTATA, diversity/Delta-Div, S-FID (Sei-gated), recon acc
tests/test_discdiff.py      # 9 CPU tests
demo_absorb_escape.py       # runnable Fig-4 demo (no weights)
evaluate.py                 # CorTATA/Delta-Div (+ S-FID if Sei supplied)
```

## Honesty

Absorb-Escape correctness is **measured** on synthetic profiles, not asserted. The LDM is
built and its forward pass verified, but its S-FID/CorTATA are **not** reproduced without
weights and are labeled gated. All Table 5/6 numbers are cited as the paper's.
