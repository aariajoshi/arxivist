"""D3LM model wrapper: load official weights + generate (Sec 2.3, 3.2).

Loads the released D3LM / D3LM-R from the HuggingFace Hub via
``AutoModelForMaskedLM(trust_remote_code=True)`` and generates DNA either through
the model's bundled ``diffusion_generate()`` (preferred, exact paper sampler) or,
as a fallback for transformers-version issues, through the from-scratch reverse
unmasking loop in ``masked_diffusion.py`` driven by the model's plain masked-LM
forward pass.
"""
from __future__ import annotations

from typing import List

import torch

from ..data.tokenizer import SixMerTokenizer
from .masked_diffusion import MaskedDiffusion

# HF repo ids for the released weights.
MODELS = {
    "D3LM": "Hengchang-Liu/D3LM-from-nt",     # initialized from NT-v2 (best for understanding)
    "D3LM-R": "Hengchang-Liu/D3LM-scratch",   # random init (best for generation, SFID 10.92)
}


class D3LMGenerator:
    """Official-weights D3LM with unconditional DNA generation."""

    def __init__(self, model, tokenizer: SixMerTokenizer, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, variant: str = "D3LM-R", device: str = "cpu") -> "D3LMGenerator":
        from transformers import AutoModelForMaskedLM

        model_name = MODELS.get(variant, variant)
        tok = SixMerTokenizer(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device).eval()
        # The bundled diffusion_generate reads mask_token_id from generation_config.
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.mask_token_id = tok.mask_id
        print(f"[d3lm] loaded {model_name} | vocab={tok.vocab_size} | mask_id={tok.mask_id}")
        return cls(model, tok, device)

    # --- generation --------------------------------------------------------
    def generate(self, n: int = 4, length: int = 2048, steps: int = 50,
                 temperature: float = 1.1, top_p: float = 1.0) -> List[str]:
        """Generate ``n`` DNA sequences of ``length`` bp via masked diffusion."""
        # length in bp -> tokens (non-overlap 6-mer)
        n_tokens = length // 6
        seqs: List[str] = []

        # Preferred: the model's own diffusion_generate (exact paper sampler).
        if hasattr(self.model, "diffusion_generate"):
            try:
                from transformers import GenerationConfig  # noqa: F401
                for _ in range(n):
                    out = self._hf_diffusion_generate(n_tokens, steps, temperature, top_p)
                    seqs.append(out)
                return seqs
            except Exception as exc:  # noqa: BLE001
                print(f"[d3lm] diffusion_generate failed ({str(exc)[:120]}); "
                      f"falling back to the from-scratch reverse loop.")

        # Fallback: from-scratch reverse unmasking driving the plain masked-LM forward.
        diff = MaskedDiffusion(mask_id=self.tokenizer.mask_id, vocab_size=self.tokenizer.vocab_size)

        def predictor(input_ids: torch.Tensor) -> torch.Tensor:
            out = self.model(input_ids=input_ids)
            return out.logits

        ids = diff.generate(predictor, length=n_tokens, batch=n, steps=steps,
                            temperature=temperature, order="random", device=self.device)
        for b in range(n):
            seqs.append(self.tokenizer.decode(ids[b].tolist()))
        return seqs

    def _hf_diffusion_generate(self, n_tokens: int, steps: int,
                               temperature: float, top_p: float) -> str:
        """Call the model's bundled diffusion_generate with an all-mask prompt.

        The bundled sampler needs ``mask_token_id`` set on the model's
        generation_config (else it raises "mask_token_id must be set"). We inject
        the tokenizer's mask id so the official sampler runs.
        """
        mask_id = self.tokenizer.mask_id
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.mask_token_id = mask_id
        prompt = torch.full((1, n_tokens), mask_id, dtype=torch.long, device=self.device)
        gen = self.model.diffusion_generate(
            prompt, steps=steps, temperature=temperature, top_p=top_p,
            mask_token_id=mask_id,
        )
        ids = gen[0].tolist() if hasattr(gen, "__getitem__") else gen.tolist()
        return self.tokenizer.decode(ids)
