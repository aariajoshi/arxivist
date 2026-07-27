"""
hopann — Heteroskedastic Ordered Probit Artificial Neural Network.

This package provides a PyTorch implementation of the OPANN and HOPANN models
described in the paper "Heteroskedastic Ordered Probit Artificial Neural Networks"
(Ni et al. 2019 dataset context).

Modules
-------
models      : OPANN, HOPANN, and baseline model definitions
data        : Dataset loading, splitting, and feature transforms
training    : Loss functions, trainer, hyperparameter search, early stopping
evaluation  : Ordinal classification metrics
utils       : Configuration loading, reproducibility helpers
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hopann")
except PackageNotFoundError:  # running from source
    __version__ = "0.1.0.dev"

__all__ = ["__version__"]
