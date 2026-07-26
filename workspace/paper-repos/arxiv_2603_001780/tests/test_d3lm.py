"""Unit tests for the D3LM reproduction (CPU, no weights).

Verifies the paper's masked-diffusion algorithm + tokenizer + metrics that can be
checked without the official checkpoint:

* Forward masking masks ~t fraction (Sec 2.1).
* 1/t-weighted CE loss scores only masked positions and scales with 1/t (Eq 2).
* Generation fills every [M] and recovers a target when the predictor is oracle (Eq 4).
* Non-overlap 6-mer tokenizer: 4096 k-mers, N/6 tokens, round-trip.
* Metrics: GC ratio, diversity, novelty, motif correlation behave sensibly.
"""
import torch

from src.d3lm.data.tokenizer import SixMerTokenizer, _all_6mers
from src.d3lm.evaluation.metrics import diversity, gc_ratio, novelty
from src.d3lm.evaluation.motif import motif_correlation
from src.d3lm.models.masked_diffusion import MaskedDiffusion


def test_kmer_vocab():
    """Non-overlap 6-mer over ACGT -> 4096 k-mers; +9 special = 4105 vocab."""
    assert len(_all_6mers()) == 4 ** 6 == 4096
    tok = SixMerTokenizer()
    assert tok.vocab_size == 4105


def test_non_overlap_tokenization():
    """N/6 tokens (shift == 6) + round-trip."""
    tok = SixMerTokenizer()
    seq = "ACGTAC" * 4          # 24 bp
    ids = tok.encode(seq)
    assert len(ids) == 24 // 6 == 4
    assert tok.decode(ids) == seq


def test_forward_mask_fraction():
    """Forward masking masks approximately t of the tokens."""
    torch.manual_seed(0)
    diff = MaskedDiffusion(mask_id=4, vocab_size=4105)
    x0 = torch.randint(9, 4105, (1, 2000))   # non-special tokens
    xt = diff.forward_mask(x0, t=0.3, generator=torch.Generator().manual_seed(0))
    frac = (xt == 4).float().mean().item()
    assert 0.25 < frac < 0.35, frac
    # t=0 masks nothing, t=1 masks everything
    assert (diff.forward_mask(x0, 0.0) == 4).sum() == 0
    assert (diff.forward_mask(x0, 1.0) == 4).all()


def test_loss_only_masked_and_inverse_t():
    """CE is computed on masked positions only and scales as 1/t (Eq 2)."""
    torch.manual_seed(0)
    diff = MaskedDiffusion(mask_id=4, vocab_size=10)
    x0 = torch.randint(5, 10, (2, 8))
    xt = x0.clone(); xt[:, :4] = 4               # mask first half
    logits = torch.randn(2, 8, 10)
    l_lo = diff.loss(logits, x0, xt, t=0.25)
    l_hi = diff.loss(logits, x0, xt, t=0.5)
    # same CE, but 1/t weighting -> t=0.25 gives 2x the t=0.5 loss
    assert abs(l_lo.item() - 2 * l_hi.item()) < 1e-4
    # no masked tokens -> zero loss
    assert diff.loss(logits, x0, x0.clone(), t=0.5).item() == 0.0


def test_generation_fills_all_masks():
    """generate() leaves no [M] and, with an oracle predictor, recovers the target."""
    torch.manual_seed(0)
    V, L = 12, 16
    diff = MaskedDiffusion(mask_id=0, vocab_size=V)
    target = torch.randint(1, V, (1, L))

    def oracle(input_ids):
        # predict the target token everywhere with high confidence
        logits = torch.full((*input_ids.shape, V), -10.0)
        for j in range(input_ids.size(1)):
            logits[0, j, target[0, j]] = 10.0
        return logits

    out = diff.generate(oracle, length=L, batch=1, steps=8, temperature=0.1,
                        order="random", generator=torch.Generator().manual_seed(0))
    assert (out == 0).sum() == 0                  # no leftover masks
    assert torch.equal(out, target)               # oracle recovered exactly


def test_gc_ratio():
    assert abs(gc_ratio(["GGGCCC"]) - 1.0) < 1e-9        # 3 G / 3 C
    assert abs(gc_ratio(["GGGGCC"]) - 2.0) < 1e-9        # 4 G / 2 C


def test_diversity_and_novelty():
    a = ["ACGTACGT", "ACGTACGT"]                   # identical -> diversity 0
    assert diversity(a) == 0.0
    b = ["ACGTACGT", "TTTTTTTT"]                    # differ -> > 0
    assert diversity(b) > 0
    # novelty: distance to nearest train seq
    assert novelty(["ACGTACGT"], ["ACGTACGT"]) == 0.0
    assert novelty(["TTTTTTTT"], ["ACGTACGT"]) > 0


def test_motif_correlation_self():
    """A set correlated with itself has motif correlation ~1."""
    seqs = ["AAATATAAACCC" * 3, "GGGTATAAATTT" * 3, "CCCTATAAAGGG" * 3]
    assert motif_correlation(seqs, seqs, "TATA") > 0.99
