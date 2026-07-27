"""
hopann.data.transforms — Feature preprocessing transforms.

Implements SelectiveStandardScaler: a StandardScaler that standardises only
a subset of columns (those without a skip mask), leaving others unchanged.

This is required because the paper specifies that continuous features are
standardised EXCEPT:
    - Asin_verified_share  (a share/proportion in [0,1])
    - Day dummy columns    (binary 0/1 indicators)

The scaler is fitted on the training set and applied consistently to
validation and test sets (no data leakage).
"""

from __future__ import annotations

import numpy as np


class SelectiveStandardScaler:
    """
    StandardScaler that standardises only a configurable subset of columns.

    Columns in `skip_mask` (True entries) are passed through unchanged.
    Columns not in `skip_mask` are standardised: (x - mean) / std.

    Fitted on training data only (fit or fit_transform), then applied to
    val/test data via transform.

    Args:
        skip_mask (list[bool] | None): Boolean mask of length K where True
            means the corresponding column should NOT be standardised.
            If None, all columns are standardised (standard behaviour).
        eps (float): Small constant added to std to avoid division by zero.

    Raises:
        ValueError: If skip_mask length does not match input column count.
        RuntimeError: If transform is called before fit.

    Example:
        >>> scaler = SelectiveStandardScaler(skip_mask=[False, True, False])
        >>> X_train_scaled = scaler.fit_transform(X_train)
        >>> X_test_scaled  = scaler.transform(X_test)
    """

    def __init__(
        self,
        skip_mask: list[bool] | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.skip_mask = skip_mask
        self.eps = eps
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SelectiveStandardScaler":
        """
        Compute mean and std on the training set for non-skipped columns.

        Args:
            X: Feature matrix of shape (N, K).

        Returns:
            self (fitted)
        """
        assert X.ndim == 2, f"Expected X of shape (N, K), got {X.shape}"
        K = X.shape[1]
        self._validate_mask(K)

        self._mean = np.mean(X, axis=0)   # (K,)
        self._std  = np.std(X, axis=0)    # (K,)

        # Zero out mean/std for skipped columns so transform leaves them unchanged
        if self.skip_mask is not None:
            skip = np.array(self.skip_mask, dtype=bool)
            self._mean[skip] = 0.0
            self._std[skip]  = 1.0  # no-op: (x - 0) / 1 = x

        # Guard against zero variance columns
        zero_std = self._std < self.eps
        if zero_std.any():
            import logging
            logging.getLogger(__name__).warning(
                "SelectiveStandardScaler: %d column(s) have near-zero std "
                "(< %g); setting std=1.0 for those columns.",
                int(zero_std.sum()), self.eps
            )
            self._std[zero_std] = 1.0

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardisation using fitted mean and std.

        Args:
            X: Feature matrix of shape (N, K).

        Returns:
            Standardised feature matrix of shape (N, K), same dtype as X.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "SelectiveStandardScaler.fit() must be called before transform()."
            )
        assert X.ndim == 2, f"Expected X of shape (N, K), got {X.shape}"
        self._validate_mask(X.shape[1])
        return ((X - self._mean) / self._std).astype(X.dtype)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on X and return the transformed X in one call.

        Args:
            X: Feature matrix of shape (N, K).

        Returns:
            Standardised feature matrix of shape (N, K).
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Inverse
    # ------------------------------------------------------------------

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse the standardisation.

        Args:
            X_scaled: Standardised feature matrix of shape (N, K).

        Returns:
            Original-scale feature matrix of shape (N, K).
        """
        if not self._fitted:
            raise RuntimeError(
                "SelectiveStandardScaler.fit() must be called before inverse_transform()."
            )
        return (X_scaled * self._std + self._mean).astype(X_scaled.dtype)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _validate_mask(self, K: int) -> None:
        if self.skip_mask is not None and len(self.skip_mask) != K:
            raise ValueError(
                f"skip_mask length ({len(self.skip_mask)}) does not match "
                f"number of columns ({K})."
            )

    @property
    def mean_(self) -> np.ndarray | None:
        """Fitted column means (None until fit is called)."""
        return self._mean

    @property
    def std_(self) -> np.ndarray | None:
        """Fitted column standard deviations (None until fit is called)."""
        return self._std

    def __repr__(self) -> str:
        n_skipped = sum(self.skip_mask) if self.skip_mask else 0
        return (
            f"SelectiveStandardScaler("
            f"n_skipped={n_skipped}, "
            f"fitted={self._fitted})"
        )
