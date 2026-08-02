"""Pure-PyTorch reference for the D3LM masked-diffusion algorithm (Sec 2).

This implements the paper's core contribution — masked diffusion over discrete
DNA tokens — in plain PyTorch, so it is unit-testable on CPU without the official
weights or any custom kernels. Three pieces:

* **Forward masking** (Sec 2.1): each token becomes ``[M]`` with probability t,
  where t ~ U[0,1] is the (variable) masking ratio.
* **Training loss** (Eq 2): cross-entropy on the masked positions only, weighted
  by 1/t:
      L = -E_{t,x0,xt}[ (1/t) Σ_i 1[xt_i = [M]] log p_θ(x0_i | xt) ]
* **Reverse denoising / generation** (Eq 4): start from an all-``[M]`` sequence
  and, over T steps, predict p(x0|xt), unmask a scheduled subset (random order by
  default, Sec 2.3), and re-mask the rest — with temperature scaling (Eq 5).

The actual mask-predictor is the official NT-v2 / ESM backbone (see d3lm.py); here
any callable ``predictor(input_ids) -> logits [B,L,V]`` is accepted, so the algebra
can be verified against a trivial predictor on CPU.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


class MaskedDiffusion:
    """The D3LM masked-diffusion objective + sampler (Sec 2)."""

    def __init__(self, mask_id: int, vocab_size: int) -> None:
        self.mask_id = mask_id
        self.vocab_size = vocab_size

    # --- forward process (Sec 2.1) ----------------------------------------
    def forward_mask(self, x0: torch.Tensor, t: float,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Mask each token independently with probability ``t`` -> xt."""
        assert 0.0 <= t <= 1.0, f"masking ratio t must be in [0,1], got {t}"
        probs = torch.rand(x0.shape, generator=generator, device=x0.device)
        xt = x0.clone()
        xt[probs < t] = self.mask_id
        return xt

    # --- training loss (Eq 2) ---------------------------------------------
    def loss(self, logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor,
             t: float, eps: float = 1e-4) -> torch.Tensor:
        """1/t-weighted cross-entropy on masked positions only (Eq 2).

        Args:
            logits: [B, L, V] mask-predictor output.
            x0: [B, L] clean tokens (targets).
            xt: [B, L] masked input (to locate [M] positions).
            t: masking ratio used for this batch.
        """
        masked = (xt == self.mask_id)                      # [B, L]
        if masked.sum() == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(
            logits[masked], x0[masked], reduction="mean"
        )
        return ce / max(t, eps)                            # 1/t weighting

    # --- reverse process / generation (Eq 4, Eq 5) ------------------------
    @torch.no_grad()
    def generate(self, predictor: Callable[[torch.Tensor], torch.Tensor],
                 length: int, batch: int = 1, steps: int = 50,
                 temperature: float = 1.1, order: str = "random",
                 device: str = "cpu",
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Iteratively unmask an all-[M] sequence into DNA tokens (Sec 2.3).

        Args:
            predictor: callable input_ids[B,L] -> logits[B,L,V].
            order: 'random' (paper default, best for DNA) | 'confidence' (MaskGit).
        """
        x = torch.full((batch, length), self.mask_id, dtype=torch.long, device=device)
        # linear unmasking schedule: how many tokens are still masked after step s
        for s in range(steps):
            masked = (x == self.mask_id)
            n_masked = int(masked[0].sum())
            if n_masked == 0:
                break
            logits = predictor(x)                          # [B, L, V]
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), 1, generator=generator
            ).reshape(batch, length)
            # how many to unmask this step (reveal an even share over remaining steps)
            remaining_steps = steps - s
            n_unmask = max(1, n_masked // remaining_steps)
            for b in range(batch):
                masked_idx = torch.nonzero(x[b] == self.mask_id, as_tuple=False).squeeze(-1)
                if masked_idx.numel() == 0:
                    continue
                if order == "confidence":
                    conf = probs[b, masked_idx].max(dim=-1).values
                    pick = masked_idx[torch.topk(conf, min(n_unmask, masked_idx.numel())).indices]
                else:  # random (paper default)
                    perm = torch.randperm(masked_idx.numel(), generator=generator, device=device)
                    pick = masked_idx[perm[:n_unmask]]
                x[b, pick] = sampled[b, pick]
        # any leftover masks: fill greedily
        leftover = (x == self.mask_id)
        if leftover.any():
            logits = predictor(x)
            x[leftover] = logits.argmax(-1)[leftover]
        return x
