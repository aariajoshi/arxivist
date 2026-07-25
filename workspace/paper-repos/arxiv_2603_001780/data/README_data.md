# Data & Weights

## Official weights (HuggingFace, public)
D3LM ships two 50M checkpoints (both loadable via `AutoModelForMaskedLM`, `trust_remote_code=True`):

| variant | HF repo | best for | paper |
|---|---|---|---|
| **D3LM-R** | `Hengchang-Liu/D3LM-scratch` | generation | SFID **10.92** (2048bp) |
| **D3LM** | `Hengchang-Liu/D3LM-from-nt` | understanding | splice donor MCC 0.959 |

```bash
python data/download.py --weights D3LM-R
```

## EPD-GenDNA dataset (DiscDiff)
D3LM trains/evaluates on **EPD-GenDNA** (Li et al. 2024 — the DiscDiff dataset): 160k
sequences, 15 species, 2048/256 bp centered on the TSS. D3LM uses the **mammalian subset**
(~80k). Fetch via HF datasets:

```bash
python data/download.py --data epd_gendna
```

Or place a local FASTA at `data/epd_gendna/{train,val,test}.fasta` (one sequence per record).

**No data?** The loader falls back to a **clearly-labeled synthetic** promoter set (weak TATA
motif) so the pipeline and tests run — its metrics are NOT comparable to the paper.

## Metrics
- **GC ratio** (#G/#C, Chargaff parity ~1.0) — cheap, real, discriminative (paper: D3LM-R 1.07 vs Evo 0.86).
- **Diversity / Novelty** — edit-distance based, no external model.
- **Motif correlation** (CorTATA etc.) — per-position motif frequency vs natural DNA.
- **SFID** — needs the pretrained **Sei** genomic CNN (20k chromatin profiles); computed only
  if a Sei embedder is supplied to `evaluation/metrics.py:sfid`.

## Note on pretraining
Generative pretraining used an NVIDIA A800 cluster — infeasible to reproduce. This repo loads
the released weights and generates/evaluates instead.
