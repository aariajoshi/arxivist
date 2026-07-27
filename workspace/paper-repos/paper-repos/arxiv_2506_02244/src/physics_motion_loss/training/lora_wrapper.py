"""
training/lora_wrapper.py — LoRA/PEFT integration for Hunyuan backbone.

Implements the LoRA fine-tuning strategy from Section 4.2 of arXiv 2506.02244v2:
"For the Hunyuan model, we adopt LoRA (PEFT) due to its size, inserting low-rank
adapters into attention and time-related linear layers while freezing all other weights."

WARNING — HIGH CONFIDENCE RISK (R1):
  The paper does NOT specify LoRA rank, alpha, or exact target modules.
  All values below are ASSUMED based on common practice (conf < 0.60).
  Run an ablation sweep over rank ∈ {4, 8, 16, 32} before comparing to paper results.

Paper reference: Section 4.2.
"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn


class LoRAWrapper:
    """Applies PEFT LoRA adapters to a video diffusion backbone.

    Paper reference: Section 4.2 — LoRA on attention and time-related linear layers.

    Args:
        rank:           LoRA rank r. ASSUMED: 16 (not stated in paper, conf < 0.60).
        alpha:          LoRA alpha scaling. ASSUMED: 32 (2×rank, conf < 0.60).
        target_modules: List of module name patterns to adapt.
                        ASSUMED: ['attn', 'time_embed'] (conf < 0.60).
        dropout:        LoRA dropout. ASSUMED: 0.1 (conf < 0.60).
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.1,
    ) -> None:
        # ASSUMED: all LoRA hyperparams — not specified in paper (conf < 0.60)
        # TODO: ablation sweep over rank ∈ {4, 8, 16, 32}
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["attn", "time_embed"]
        self.dropout = dropout

    def wrap(self, model: nn.Module) -> nn.Module:
        """Apply LoRA adapters and freeze all non-LoRA weights.

        Args:
            model: Pretrained backbone (Hunyuan) as nn.Module.

        Returns:
            PEFT-wrapped model with LoRA adapters inserted.

        Raises:
            ImportError: If the `peft` package is not installed.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "The `peft` package is required for LoRA fine-tuning. "
                "Install with: pip install peft>=0.9.0"
            )

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
        )

        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    @staticmethod
    def freeze_non_lora(model: nn.Module) -> None:
        """Freeze all parameters that are not LoRA adapter weights.

        Paper reference: Section 4.2 — "freezing all other weights."
        This is typically handled automatically by get_peft_model, but can
        be called explicitly for safety.

        Args:
            model: A PEFT-wrapped model.
        """
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    def __repr__(self) -> str:
        return (
            f"LoRAWrapper(rank={self.rank}, alpha={self.alpha}, "
            f"target_modules={self.target_modules}, dropout={self.dropout})"
        )
