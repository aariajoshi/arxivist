# Architecture Plan — D3LM (arxiv_2603_001780)

**Framework:** PyTorch ≥2.1 · HuggingFace (official weights, `trust_remote_code`) · YAML · **CPU-testable algorithm; GPU only for generation**

## Strategy
D3LM is a **masked-diffusion DNA language model** on the **NT-v2** backbone (ESM encoder + RoPE +
SwiGLU). Weights are on HF (`Hengchang-Liu/D3LM-from-nt` = D3LM, `D3LM-scratch` = D3LM-R), loaded via
`AutoModelForMaskedLM(trust_remote_code=True)` with a bundled `diffusion_generate()`. We (a)
re-implement the **masked-diffusion algorithm from scratch** — variable-ratio forward masking, the
**1/t-weighted CE** loss (Eq 2), and the **iterative reverse unmasking** sampler (Eq 4, temperature
1.1, random order) — fully CPU-unit-testable; and (b) load the **official weights** to generate and
evaluate. Generation quality uses **cheap real metrics** (GC ratio/Chargaff, Diversity, Novelty,
motif Pearson corr); **SFID** (needs the Sei genomic CNN) only if Sei loads.

## Module hierarchy
```
src/d3lm/
├── models/
│   ├── masked_diffusion.py  # forward_mask, 1/t loss (Eq 2), generate() reverse loop (Eq 4) — CPU
│   └── d3lm.py              # D3LMGenerator.from_pretrained (HF diffusion_generate) + classifier head
├── data/
│   ├── tokenizer.py        # non-overlap 6-mer (vocab ~4105)
│   └── epd_gendna.py       # EPD-GenDNA loader (DiscDiff dataset) + synthetic fallback
├── evaluation/
│   ├── metrics.py          # GC ratio, Diversity, Novelty, SFID (if Sei available)
│   └── motif.py            # TATA/Initiator/GC-box/CCAAT distributions + CorTATA
└── utils/config.py         # YAML + seed + model registry (T=50, tau=1.1)
```

## Core algorithm (Eq 2 / Eq 4)
Train: `L = -E[ (1/t) · Σ 1[xt=M] log p(x0|xt) ]`, t~U[0,1].
Generate: start all-`[M]` → for T=50 steps predict p(x0|xt), unmask a random subset with temp 1.1, repeat.

## Variants
| | D3LM (from NT-v2) | D3LM-R (scratch) |
|---|---|---|
| init | pretrained NT-v2-50m | random |
| params | 50M | 50M |
| best at | understanding (splice donor MCC 0.959) | generation (SFID 10.92) |

## Entrypoints
- `generate.py` — `--config --n --length --steps --temperature --out`
- `evaluate.py` — `--config --generated --reference`
- `data/download.py` — HF weights + EPD-GenDNA

## Reproduction targets (paper)
2048bp generation (D3LM-R): **SFID 10.92** (Truth 7.85; HyenaDNA 29.16; DiscDiff 62.74), **GC ratio 1.07** (Truth 1.06). Understanding (D3LM): splice donor MCC 0.959.

## Top risks
1. **[Med]** SFID needs the Sei CNN → attempt if loadable; GC ratio/diversity/motif are cheap real proxies (GC alone separates good vs collapsed: 1.07 vs Evo 0.86 / P2 12.7).
2. **[Med]** custom `diffusion_generate()` remote code vs new transformers → pin `<4.46`; ship from-scratch reverse loop as fallback.
3. **[Low]** EPD-GenDNA large → synthetic fallback keeps pipeline/tests running.
4. **[Low]** vocab 4105 vs 4107 → read from tokenizer at load.
