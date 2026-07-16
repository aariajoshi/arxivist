"""
hopann.models — Model definitions for OPANN, HOPANN, and baselines.

Implements:
- MeanNetwork (Section 3.1): latent index f(x, theta)
- VarianceNetwork variants (Section 3.2): per-sample dispersion sigma_i
- CuttingPoints (Section 2.3): monotone threshold parameter layer
- OPANN (Section 3.1): ordered probit with ANN index
- HOPANN (Section 3.2): heteroskedastic extension of OPANN
- BaselineModel ABC + 5 concrete baseline classifiers
"""

from hopann.models.mean_network import MeanNetwork
from hopann.models.variance_network import (
    VarianceNetworkBase,
    VarianceNetworkANN,
    VarianceNetworkLinear,
)
from hopann.models.cutting_points import CuttingPoints
from hopann.models.opann import OPANN
from hopann.models.hopann import HOPANN
from hopann.models.baselines import (
    BaselineModel,
    OrderedProbitBaseline,
    ANNBaseline,
    SVMBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)

__all__ = [
    "MeanNetwork",
    "VarianceNetworkBase",
    "VarianceNetworkANN",
    "VarianceNetworkLinear",
    "CuttingPoints",
    "OPANN",
    "HOPANN",
    "BaselineModel",
    "OrderedProbitBaseline",
    "ANNBaseline",
    "SVMBaseline",
    "RandomForestBaseline",
    "XGBoostBaseline",
]
