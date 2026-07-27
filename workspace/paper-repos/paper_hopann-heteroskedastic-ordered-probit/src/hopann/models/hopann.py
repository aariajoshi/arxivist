"""
hopann.models.hopann — Heteroskedastic Ordered Probit ANN (HOPANN).

Extends OPANN (Section 3.1) with a per-sample variance sub-network to model
input-dependent dispersion (Section 3.2).

Model equations (Section 3.2):
    sigma_i = exp(variance_ANN(z_i))                                 [Eq. HOPANN-1]

    P(y=j | x, z) = Phi((c_j - f(x)) / sigma_i)
                  - Phi((c_{j-1} - f(x)) / sigma_i)                 [Eq. HOPANN-2]

    where:
        f(x, theta) : latent index from MeanNetwork (Eq. OPANN-2)
        sigma_i     : per-sample std from VarianceNetwork
        c_j         : cutting points (monotone, via CuttingPoints)
        Phi         : standard normal CDF

    # WARNING: low-confidence implementation (conf 0.55)
    # TODO: z_i is ASSUMED to equal x_i (same feature vector as mean network).
    #       The paper does not explicitly specify the conditioning variable.
    #       If z_i differs from x_i (e.g. a subset of features), update
    #       VarianceNetworkANN.input_dim and the forward() call accordingly.

HOPANN inherits its cutting-point layer and _compute_probs from OPANN.
"""

from __future__ import annotations

import torch

from hopann.models.opann import OPANN
from hopann.models.variance_network import VarianceNetworkBase, build_variance_network


class HOPANN(OPANN):
    """
    Heteroskedastic Ordered Probit Artificial Neural Network.

    Extends OPANN by adding a per-sample variance sub-network that produces
    sigma_i > 0 for each sample, making the model heteroskedastic.

    Paper reference: Section 3.2.

    Args:
        input_dim (int):       K — number of input features (inferred from data).
        num_classes (int):     J — number of ordinal classes.
        hidden_dim (int):      Q — hidden nodes in MeanNetwork.
                               ASSUMED: Q=16 (conf 0.52).
        variance_type (str):   Variance network variant: 'ann' or 'linear'.
                               ASSUMED: 'ann' (conf 0.62); swappable via config.
        variance_hidden_dim (int): Hidden units in variance ANN.
                               ASSUMED: same as hidden_dim (conf 0.52).
        dropout_rate (float):  Dropout on mean network hidden layer.
                               ASSUMED: 0.0 (not mentioned in paper).

    Shape:
        Input:  x of shape (B, K)
        Output: probs of shape (B, J) — class probability distribution
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 16,              # ASSUMED: Q=16 (conf 0.52)
        variance_type: str = "ann",        # ASSUMED: ANN variance (conf 0.62)
        variance_hidden_dim: int | None = None,  # defaults to hidden_dim
        dropout_rate: float = 0.0,         # ASSUMED: no dropout
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.variance_type = variance_type
        variance_hidden_dim = variance_hidden_dim or hidden_dim

        # Variance sub-network
        # WARNING: low-confidence implementation (conf 0.55)
        # TODO: z_i = x_i is ASSUMED; variance_network input_dim = K
        self.variance_network: VarianceNetworkBase = build_variance_network(
            variance_type=variance_type,
            input_dim=input_dim,    # ASSUMED: z_i = x_i → same dim as mean net
            hidden_dim=variance_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute P(y=j | x, z) for all classes j=1..J (heteroskedastic).

        Args:
            x: Feature tensor of shape (B, K).

        Returns:
            Class probability tensor of shape (B, J). Each row sums to 1.

        Note:
            # WARNING: low-confidence implementation
            # TODO: z_i is ASSUMED to equal x_i (conf 0.55). If the paper
            # specifies a different conditioning variable, pass it as a
            # separate argument and update this call.
        """
        assert x.dim() == 2, f"Expected x of shape [B, K], got {x.shape}"
        B = x.size(0)

        # Eq. OPANN-2: latent index f(x, theta), shape (B, 1)
        f = self.mean_network(x)   # (B, 1)

        # Eq. HOPANN-1: sigma_i = exp(variance_ANN(z_i))
        # ASSUMED: z_i = x_i (conf 0.55)
        sigma = self.variance_network(x)  # (B, 1); z_i = x_i (ASSUMED)

        # Cutting points c_1 < ... < c_{J-1}, shape (J-1,)
        c = self.cutting_points()  # (J-1,)

        # Eq. HOPANN-2: P(y=j|x,z) using heteroskedastic CDF differences
        probs = self._compute_probs(f, c, sigma=sigma)  # (B, J)

        assert probs.shape == (B, self.num_classes), (
            f"HOPANN probs shape mismatch: expected ({B}, {self.num_classes}), "
            f"got {probs.shape}"
        )
        return probs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-sample standard deviation sigma_i for a batch.

        Useful for inspecting the learned heteroskedasticity pattern.

        Args:
            x: Feature tensor of shape (B, K).

        Returns:
            sigma_i of shape (B, 1).
        """
        with torch.no_grad():
            # ASSUMED: z_i = x_i (conf 0.55)
            return self.variance_network(x)

    def __repr__(self) -> str:
        return (
            f"HOPANN(input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"hidden_dim={self.hidden_dim}, "
            f"variance_type='{self.variance_type}', "
            f"params={self.param_count()})"
        )
