"""
utils/precision.py — Mixed-precision context manager.

The paper (Sec 4.1) specifies that spectral processing and WLS solver blocks
must run in FP32 regardless of backbone precision (BF16). This module provides
a context manager that disables torch.autocast for the physics loss forward pass.
"""

from __future__ import annotations

import torch


class FP32Context:
    """Context manager that forces FP32 computation by disabling autocast.

    Paper reference: Section 4.1 — "runs in FP32 with autocast off"

    Usage::

        with FP32Context():
            loss = physics_loss(x0_hat)  # Always runs in FP32
    """

    def __init__(self) -> None:
        self._prev_enabled: bool = False
        self._autocast_ctx = None

    def __enter__(self) -> "FP32Context":
        # Disable autocast for the duration of this block
        if torch.is_autocast_enabled():
            self._prev_enabled = True
            self._autocast_ctx = torch.amp.autocast("cuda", enabled=False)
            self._autocast_ctx.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._prev_enabled and self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(*args)

    def __repr__(self) -> str:
        return "FP32Context()"
