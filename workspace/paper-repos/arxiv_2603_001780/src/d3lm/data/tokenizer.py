"""Non-overlapping 6-mer DNA tokenizer (Sec 2.2, 3.1).

D3LM follows NT-v2: non-overlapping 6-mers over {A,C,G,T} give 4^6 = 4096 k-mer
tokens plus 9 special tokens (incl. the mask token [M] and [PAD]) for a vocab of
4105. When the official weights are present we defer to their bundled tokenizer;
otherwise this faithful built-in reproduces the same scheme so the pipeline and
unit tests run without downloads.
"""
from __future__ import annotations

import itertools
from typing import List, Optional

_BASES = "ACGT"
_SPECIAL = ["[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]", "[BOS]", "[EOS]", "[R1]", "[R2]"]


def _all_6mers() -> List[str]:
    return ["".join(p) for p in itertools.product(_BASES, repeat=6)]  # 4096


class SixMerTokenizer:
    """Non-overlapping 6-mer tokenizer (vocab 4105)."""

    def __init__(self, hf_name: Optional[str] = None) -> None:
        self.hf = None
        if hf_name is not None:
            try:
                from transformers import AutoTokenizer

                self.hf = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            except Exception:
                self.hf = None

        vocab = _SPECIAL + _all_6mers()          # specials first, then 4096 6-mers
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.mask_token = "[MASK]"
        self.pad_token = "[PAD]"

    @property
    def vocab_size(self) -> int:
        return int(self.hf.vocab_size) if self.hf is not None else len(self.stoi)

    @property
    def mask_id(self) -> int:
        if self.hf is not None:
            mid = self.hf.mask_token_id
            return int(mid) if mid is not None else 0
        return self.stoi[self.mask_token]

    @property
    def pad_id(self) -> int:
        if self.hf is not None:
            pid = self.hf.pad_token_id
            return int(pid) if pid is not None else 0
        return self.stoi[self.pad_token]

    def encode(self, seq: str) -> List[int]:
        """Non-overlapping 6-mer ids (shift == 6)."""
        if self.hf is not None:
            return self.hf(seq, add_special_tokens=False)["input_ids"]
        seq = seq.upper()
        ids = []
        for i in range(0, len(seq) - 5, 6):
            ids.append(self.stoi.get(seq[i:i + 6], self.stoi["[UNK]"]))
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.hf is not None:
            return self.hf.decode(ids, skip_special_tokens=True).replace(" ", "")
        return "".join(self.itos.get(int(i), "") for i in ids
                       if self.itos.get(int(i), "").isalpha())
