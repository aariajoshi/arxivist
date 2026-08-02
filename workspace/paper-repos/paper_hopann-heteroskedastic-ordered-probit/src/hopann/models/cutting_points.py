"""
hopann.models.cutting_points — Monotone threshold (cutting point) parameter layer.

Implements the ordered probit cutting points (c_1, c_2, ..., c_{J-1}) with a
softplus + cumulative-sum parameterisation that guarantees strict monotonicity:

    c_1       is unconstrained                                       [Eq. CP-1]
    c_k = c_1 + cumsum(softplus(delta_k)), k = 2 .. J-1             [Eq. CP-2]

This ensures c_1 < c_2 < ... < c_{J-1} at all times, satisfying the ordered
probit model assumption.

Reference: Standard ordered probit cutting-point parameterisation; see also
Greene & Hensher (2010) "Modeling Ordered Choices".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuttingPoints(nn.Module):
    """
    Learnable monotone cutting points for ordered probit models.

    Parameterises J-1 thresholds as:
        c_1 : unconstrained scalar (raw learnable parameter)
        c_k : c_1 + cumsum(softplus(delta_2), ..., softplus(delta_k))  for k > 1

    The softplus transformation ensures each increment is strictly positive, so
    the sequence is strictly increasing.

    Args:
        num_classes (int): J — total number of ordinal classes.
                           The layer learns J-1 thresholds.

    Raises:
        ValueError: If num_classes < 2.

    Shape:
        Output of forward(): (J-1,) — sorted cutting point values
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2 for an ordered model, got {num_classes}"
            )
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # J - 1

        # c_1: unconstrained first threshold
        # delta_2, ..., delta_{J-1}: raw parameters for increments (J-2 values)
        # We store all J-1 raw params as a single vector; the first is c_1,
        # the rest are delta values transformed via softplus.
        self.raw_params = nn.Parameter(torch.zeros(self.num_thresholds))
        self._init_params()

    def _init_params(self) -> None:
        """
        Initialise cutting points to be approximately equally spaced in [-2, 2].

        This gives sensible starting probabilities before training.
        """
        with torch.no_grad():
            # Set c_1 so that thresholds start around -1.0
            self.raw_params[0] = -1.0
            if self.num_thresholds > 1:
                # delta_k such that softplus(delta_k) ≈ 2/(J-2) for equal spacing
                target_increment = 2.0 / max(self.num_thresholds - 1, 1)
                # softplus^{-1}(y) = log(exp(y) - 1) for y > 0
                raw_delta = torch.log(
                    torch.exp(torch.tensor(target_increment)) - 1.0
                )
                self.raw_params[1:] = raw_delta

    def forward(self) -> torch.Tensor:
        """
        Compute and return the ordered cutting points c_1 < c_2 < ... < c_{J-1}.

        Returns:
            Tensor of shape (J-1,) with strictly increasing values.
        """
        # Eq. CP-1: c_1 is unconstrained
        c1 = self.raw_params[0:1]  # shape (1,)

        if self.num_thresholds == 1:
            return c1

        # Eq. CP-2: increments delta_k passed through softplus to ensure positivity
        deltas = self.raw_params[1:]               # shape (J-2,)
        increments = F.softplus(deltas)            # strictly positive increments
        cumulative = torch.cumsum(increments, dim=0)  # cumulative sums

        # c_k = c_1 + cumsum(softplus(delta_2), ..., softplus(delta_k))
        cutting_points = torch.cat([c1, c1 + cumulative], dim=0)  # (J-1,)

        assert cutting_points.shape == (self.num_thresholds,), (
            f"CuttingPoints shape mismatch: expected ({self.num_thresholds},), "
            f"got {cutting_points.shape}"
        )
        # Verify monotonicity (should always hold by construction)
        assert (cutting_points[1:] > cutting_points[:-1]).all(), (
            "CuttingPoints are not strictly increasing — check softplus increments"
        )
        return cutting_points

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, num_thresholds={self.num_thresholds}"

    def __repr__(self) -> str:
        return (
            f"CuttingPoints(num_classes={self.num_classes}, "
            f"num_thresholds={self.num_thresholds})"
        )
