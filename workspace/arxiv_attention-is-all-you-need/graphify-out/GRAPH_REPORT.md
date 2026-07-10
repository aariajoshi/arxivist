# Graph Report - F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need  (2026-07-11)

## Corpus Check
- Corpus is ~4,886 words - fits in a single context window. You may not need a graph.

## Summary
- 159 nodes · 210 edges · 34 communities (20 shown, 14 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25

## God Nodes (most connected - your core abstractions)
1. `MultiHeadAttention` - 16 edges
2. `PositionwiseFeedForward` - 16 edges
3. `PositionalEncoding` - 16 edges
4. `make_model()` - 12 edges
5. `Transformer` - 11 edges
6. `LayerNorm` - 10 edges
7. `Encoder` - 10 edges
8. `Decoder` - 10 edges
9. `SublayerConnection` - 9 edges
10. `EncoderLayer` - 8 edges

## Surprising Connections (you probably didn't know these)
- `BLEU WMT 2014 English-German test` --semantically_similar_to--> `BLEU WMT 2014 English-German`  [INFERRED] [semantically similar]
  comparison/benchmark_comparison.md → README.md
- `WMT 2014 English-German dataset` --semantically_similar_to--> `BLEU WMT 2014 English-German`  [INFERRED] [semantically similar]
  data/README_data.md → README.md
- `BLEU WMT 2014 English-French test` --semantically_similar_to--> `BLEU WMT 2014 English-French`  [INFERRED] [semantically similar]
  comparison/benchmark_comparison.md → README.md
- `vocab_size 37000` --semantically_similar_to--> `Model Configuration`  [INFERRED] [semantically similar]
  comparison/hallucination_report.md → configs/config.yaml
- `main()` --calls--> `set_seed()`  [EXTRACTED]
  inference.py → src/transformer/utils/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Transformer Architecture Components** — comparison_hallucination_report_encoder_stack, comparison_hallucination_report_decoder_stack, comparison_hallucination_report_multi_head_attention, comparison_hallucination_report_position_wise_ffn, comparison_hallucination_report_positional_encoding [EXTRACTED 1.00]

## Communities (34 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (14): PositionalEncoding, PositionwiseFeedForward, Tensor, Feed-forward networks and positional encodings. Paper Sections: 3.3, 3.5, Position-wise Feed-Forward Network.     Paper Reference: Section 3.3, Equation 2, Positional Encoding using sine and cosine functions.     Paper Reference: Sectio, Decoder, Embeddings (+6 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (14): Module, clones(), DecoderLayer, Encoder, EncoderLayer, LayerNorm, Produce N identical layers., Construct a layernorm module. (+6 more)

### Community 2 - "Community 2"
Cohesion: 0.24
Nodes (11): main(), Compute BLEU and perplexity on test sets using beam search., main(), Translate single sentences from the command line., load_config(), Configuration loading and seeding utilities. Implements reproducibility requirem, Set seed for reproducibility across Python, NumPy, and PyTorch., Load YAML configuration file. (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.17
Nodes (6): Optimizer, NoamLRScheduler, Training loop and custom learning rate scheduler. Paper Section: 5.3, Custom learning rate scheduler from Section 5.3.          lrate = d_model^{-0.5}, STUB: This component implements the training loop.     SIR ambiguity: Handling v, Trainer

### Community 4 - "Community 4"
Cohesion: 0.19
Nodes (6): MultiHeadAttention, Tensor, Attention mechanisms. Paper Sections: 3.2.1, 3.2.2, Scaled Dot-Product Attention.     Paper Reference: Section 3.2.1, Equation 1, Multi-Head Attention.     Paper Reference: Section 3.2.2          MultiHead(Q, K, ScaledDotProductAttention

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (4): Tensor, A standard Encoder-Decoder architecture.     Paper Reference: Section 3.1, Take in and process masked src and target sequences., Transformer

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (3): Dataset, STUB: Translation dataset.     SIR ambiguity: Handling variable sequence lengths, TranslationDataset

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (3): LabelSmoothingLoss, Loss functions. Paper Section: 5.4, Label smoothing loss.     Paper Reference: Section 5.4

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): compute_bleu(), compute_perplexity(), Compute perplexity from cross-entropy loss., STUB: Compute corpus BLEU score.     Replace this stub before training.

### Community 9 - "Community 9"
Cohesion: 0.50
Nodes (3): Data transformations and tokenization., STUB: Tokenization transformations.     Replace this stub before training., TokenizerTransform

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (3): BLEU WMT 2014 English-German test, WMT 2014 English-German dataset, BLEU WMT 2014 English-German

## Knowledge Gaps
- **15 isolated node(s):** `Transformer`, `Attention Is All You Need`, `NoamLRScheduler`, `BLEU WMT 2014 English-French`, `BLEU WMT 2014 English-German test` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MultiHeadAttention` connect `Community 4` to `Community 0`, `Community 1`, `Community 5`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `Transformer` connect `Community 5` to `Community 0`, `Community 1`, `Community 4`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `PositionalEncoding` connect `Community 0` to `Community 1`, `Community 5`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `MultiHeadAttention` (e.g. with `Decoder` and `DecoderLayer`) actually correct?**
  _`MultiHeadAttention` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `PositionwiseFeedForward` (e.g. with `Decoder` and `DecoderLayer`) actually correct?**
  _`PositionwiseFeedForward` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `PositionalEncoding` (e.g. with `Decoder` and `DecoderLayer`) actually correct?**
  _`PositionalEncoding` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Transformer` (e.g. with `MultiHeadAttention` and `PositionalEncoding`) actually correct?**
  _`Transformer` has 4 INFERRED edges - model-reasoned connections that need verification._