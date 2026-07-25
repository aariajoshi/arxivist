# Verification Log — DiscDiff (arxiv_2402_006079)

## Run metadata
- Algorithm + structural verification: 2026-07-26, local CPU (torch cpu, numpy)
- Generation metrics (S-FID/CorTATA on 50k): **gated** on trained weights + Sei + EPD-GenDNA
- SIR version used: 1
- Manual review required: No

## Part A — Absorb-Escape (executed, CPU, no weights) ✅

Absorb-Escape (Sec 4.2, Table 3, Fig 4) is a deterministic algorithm over per-position
softmax confidences, so it is verifiable WITHOUT any trained model.

```
pytest tests/ -q          # 9 passed (local CPU)
python demo_absorb_escape.py
```

Verified:
- **Absorb** fires iff L(D, s_i) < T_absorb (Table 3). ✅
- **Natural escape** stops iff L(D, s_j) > L(M, s_j) (diffusion re-confident). ✅
- **Valley repair (Fig 4):** a `TATT` region with a low-confidence nucleotide is handed to
  the AR model and restored to `TATA` — output matches the intended sequence EXACTLY. ✅
- **No-op safety:** an all-confident sequence is returned unchanged (nothing absorbed). ✅
- **Locality:** only the low-confidence region is modified; confident flanks preserved. ✅

Demo output (both planted valleys corrected):
```
DiscDiff (raw)   : CGCGCATATTCGCGATCTAAA
Absorb-Escape    : CGCGCATATACGCGATATAAA   (== intended truth)
[result] 2 nucleotide(s) corrected; matches intended: True
```

## Part B — DiscDiff LDM (structural verification, CPU) ✅ structure / ⏳ metrics

The two-stage VAE + latent U-Net + DDPM are implemented (Sec 4.1) and run on tiny CPU configs:
- **Two-stage CNN-VAE:** encode(s)->(mu,logvar)->z->decode->logits [B,4,L]; CE recon + KL loss
  finite; reconstruction accuracy computable. ✅ shapes + loss verified.
- **Latent U-Net:** eps_theta(z_t, t, species) returns noise-shaped output; species+time
  cross-conditioning wired. ✅
- **DDPM (Eq 2):** q_sample + MSE eps-loss finite; one reverse p_sample step runs. ✅

Generation METRICS (S-FID 3.21/4.38, CorTATA on 50k samples) are NOT produced here — they need
trained DiscDiff weights + fine-tuned Hyena + the Sei genomic model + EPD-GenDNA. Structure is
correct; numbers are weight/data-gated (offered as a Colab run, like D3LM).

## Part B2 — full pipeline executed end-to-end (untrained, CPU)

Beyond the isolated Absorb-Escape demo, the WHOLE generate->refine->evaluate path was run:

```
python generate.py --config configs/config.yaml --n 50
  -> [discdiff] !! UNTRAINED (random init) ...
  -> latent DDPM sample (1000 steps) -> VAE decode -> Absorb-Escape -> 50x256bp
  -> wrote 50 sequences -> results/generated.fasta
python evaluate.py --config configs/config.yaml --generated results/generated.fasta
  -> [epd] SYNTHETIC test (reference not real EPD-GenDNA)
  -> [results] {'CorTATA(TATA)': -0.033, 'delta_div': 0.106, 'sfid': None}
```

**Honest read.** This proves the *pipeline* runs (DDPM sampling, VAE decode, Absorb-Escape
refinement, metric evaluation all wire together and execute on CPU). The metrics are NOT
paper-comparable and are not presented as such: the VAE/U-Net are RANDOMLY INITIALIZED (no
trained weights) and the reference is the SYNTHETIC smoke-test set. A random model has no TATA
motif structure, so CorTATA ~ 0 is the expected, correct result — exactly what an untrained
model should score. To get paper numbers (S-FID 3.21, CorTATA 0.975) supply trained DiscDiff +
Hyena weights + Sei + real EPD-GenDNA (`--weights`, Sei embedder). The gap is traced to root
cause (untrained + synthetic reference), not hidden.

## Part C — metrics behavior

- **CorTATA** (motif positional-frequency Pearson corr) = 1.0 for identical distributions. ✅
- **Delta-Div** = 0 when generated matches natural diversity. ✅
- **S-FID** returns None without a Sei embedder (gated, not faked). ✅

## Integrity
- Absorb-Escape correctness is MEASURED on synthetic confidence profiles (valley repair,
  no-op, locality), not asserted.
- The LDM forward pass is verified; its S-FID/CorTATA are explicitly NOT reproduced without
  weights and labeled gated. No generation number fabricated.
- Table 5/6 values are cited as the paper's throughout.
