# Notebooks

`reproduce_arxiv_2607_20415.ipynb` — primary walkthrough notebook. Executed end-to-end
during generation (all 20 cells ran without error; outputs then cleared before committing,
per repo convention). Covers: environment check, editable install, paper overview mapped to
implementation files, component-by-component demos (operator split, Zalesak limiter,
FCDF-B, FCDF-DC, active-set solver), a mini end-to-end OU-benchmark reproduction, and a
paper-vs-reproduction results comparison table.

No exploratory notebook was produced: this paper has no learned representations, embeddings,
attention maps, or generative outputs to visualize beyond what the primary notebook already
covers (density profiles, limiter activity) — a second notebook would be redundant.

Run with: `jupyter notebook reproduce_arxiv_2607_20415.ipynb` (or `jupyter lab`).
