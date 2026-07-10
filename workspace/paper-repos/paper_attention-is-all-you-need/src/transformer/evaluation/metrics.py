"""
Evaluation metrics.
"""

def compute_bleu(hypotheses, references):
    """
    STUB: Compute corpus BLEU score.
    Replace this stub before training.
    """
    raise NotImplementedError("See docstring — component requires manual implementation")

def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss.
    """
    import math
    return math.exp(loss)
