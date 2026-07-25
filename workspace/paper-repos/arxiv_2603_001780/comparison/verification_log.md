# Verification Log — D3LM (arxiv_2603_001780)

## Run metadata
- Algorithm verification: 2026-07-25, local CPU (torch 2.5.0+cpu)
- Generation run: 2026-07-25, Colab CPU — official weights + **paper's own `diffusion_generate` sampler**
- SFID/GC paper-comparison: gated on real EPD-GenDNA + Sei CNN (this run used synthetic reference)
- SIR version used: 1
- Manual review required: No

## Part A — algorithm verification (executed)

```
pytest tests/ -q        # 8 passed (local CPU)
```
Key checks:
- Forward masking masks ~t (t=0.3 -> ~0.30 fraction; t=0 none; t=1 all).
- Loss (Eq 2): CE on masked positions only; **1/t weighting exact** — loss(t=0.25) == 2 * loss(t=0.5).
- Generation (Eq 4): zero leftover [M]; oracle predictor -> exact target recovery.
- Tokenizer: 4096 6-mers + 9 special = 4105; N/6 tokens; round-trip.
- Metrics: GC ratio, diversity, novelty, motif correlation behave correctly.

Entrypoint sanity:
```
python generate.py --help                                  # OK
python generate.py --config configs/config_debug.yaml --dry-run   # builds pipeline
D3LMGenerator.MODELS -> {'D3LM':'Hengchang-Liu/D3LM-from-nt','D3LM-R':'Hengchang-Liu/D3LM-scratch'}
tokenizer: vocab 4105, mask_id 4, pad_id 1
```

## Part B — generation (executed)

```
python data/download.py --weights D3LM-R       # cached Hengchang-Liu/D3LM-scratch (HF)
python generate.py --config configs/config.yaml --n 100
  -> [d3lm] loaded ... vocab=4107 mask_id=2
  -> Diffusion: 100% 50/50  x100      # <-- model's OWN diffusion_generate (no fallback message)
  -> wrote 100 sequences -> results/generated.fasta
python evaluate.py --config configs/config.yaml --generated results/generated.fasta
  -> [epd] SYNTHETIC test (n=200)     # <-- reference is synthetic, NOT real EPD-GenDNA
  -> [results] gc_ratio 1.2377, diversity 67.3, novelty 1931.85,
               motif_corr ~0, sfid None (no Sei)
```

Confirmed:
- Official weights load (vocab 4107, mask_id 2) and the **paper's bundled `diffusion_generate`
  sampler runs** — the earlier `mask_token_id must be set` error is fixed (mask_token_id injected
  into generation_config); no fallback-to-from-scratch message this run.
- 100 valid 2048bp DNA sequences produced.

Gated (not defects):
- **GC 1.24 / motif ~0** are against the *synthetic* smoke-test reference (banner fired). Not
  paper-comparable — needs real EPD-GenDNA.
- **SFID None** — needs the Sei genomic CNN. Implemented, gated on availability.

To make paper-comparable: supply real EPD-GenDNA (`data/README_data.md`) + a Sei embedder, re-run
`evaluate.py`.

## Config
- Stock configs/config.yaml (D3LM-R, T=50, tau=1.1, random unmask order, 6-mer).
- No user modifications recorded.

## Integrity
- No generation metric fabricated. Reported numbers (GC 1.24, diversity 67.3, novelty 1931.85) are
  the actual `evaluate.py` output, explicitly labeled as measured against a *synthetic* reference.
  SFID is `None` (Sei not supplied), not invented. The gap to the paper's real SFID/GC is traced to
  root cause (no real EPD-GenDNA + no Sei), not papered over.
- Algorithm facts (8/8 tests, exact 1/t scaling, oracle recovery) are measured, not asserted.
