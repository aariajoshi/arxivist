"""
hopann.data.dataset — Amazon Software Product Review dataset and experiment splitter.

Dataset: Amazon Software Product Reviews (Ni et al. 2019)
    URL: https://nijianmo.github.io/amazon/index.html  (Software category)
    N = 6,173 reviews (post filtering)
    Target: Star rating (1-5 stars)
    Features (K ≈ 12-19; exact count inferred dynamically from CSV columns):
        - Price
        - Asin_n_rev        (number of reviews for the product)
        - Asin_n_reviewers  (number of unique reviewers for the product)
        - Asin_verified_share  (share of verified purchases — NOT standardised)
        - User_n_rev        (number of reviews by the user)
        - User_reviewed_asin (number of ASINs reviewed by user)
        - Day dummies       (binary indicators — NOT standardised)

Four experiments (Table 1 in paper):
    Exp 1: 5-class imbalanced    (all 5 star ratings)
    Exp 2: 3-class imbalanced    (1-2 → 0, 3 → 1, 4-5 → 2)
    Exp 3: modified-3-class imbalanced (collapsed differently)
    Exp 4: modified-3-class balanced   (undersampled to balance classes)

Data splits: Time-ordered (no date overlap between train/val/test).
    ASSUMED: 70%/15%/15% split by temporal order (conf 0.52; paper says
    "time-ordered" but does not give exact proportions).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hopann.data.transforms import SelectiveStandardScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class ExperimentData:
    """Holds feature/label arrays for a single experiment split."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    num_classes: int
    experiment_id: int
    scaler: SelectiveStandardScaler | None = None

    def class_counts(self) -> dict[str, np.ndarray]:
        """Return class distribution for each split."""
        return {
            "train": np.bincount(self.y_train, minlength=self.num_classes),
            "val":   np.bincount(self.y_val,   minlength=self.num_classes),
            "test":  np.bincount(self.y_test,  minlength=self.num_classes),
        }


# =============================================================================
# PyTorch Dataset
# =============================================================================

class AmazonReviewDataset(Dataset):
    """
    PyTorch Dataset wrapping the Amazon Software Product Review feature matrix.

    Features are already preprocessed (standardised) by ExperimentSplitter;
    this class simply wraps numpy arrays as torch tensors.

    Args:
        X (np.ndarray): Feature matrix of shape (N, K).
        y (np.ndarray): Integer class labels of shape (N,), 0-indexed.

    Shape:
        __getitem__: returns (x, y) where x is FloatTensor (K,) and y is LongTensor.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.ndim == 2, f"Expected X of shape (N, K), got {X.shape}"
        assert y.ndim == 1, f"Expected y of shape (N,), got {y.shape}"
        assert len(X) == len(y), (
            f"X and y must have the same length: {len(X)} != {len(y)}"
        )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_features = X.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return (
            f"AmazonReviewDataset(n_samples={len(self)}, "
            f"n_features={self.num_features})"
        )


# =============================================================================
# Experiment Splitter
# =============================================================================

class ExperimentSplitter:
    """
    Loads, preprocesses, and splits the Amazon Software review dataset for all
    four experimental conditions described in the paper.

    The four experiments differ in the ordinal target encoding:
        Exp 1: 5-class, original star ratings (1-5) → classes 0-4
        Exp 2: 3-class, {1,2}→0, {3}→1, {4,5}→2  (imbalanced)
        Exp 3: modified 3-class, {1,2,3}→0, {4}→1, {5}→2  (ASSUMED mapping,
               paper description is ambiguous — conf 0.52)
        Exp 4: Same as Exp 3 but with class balancing via undersampling

    Data split: Time-ordered 70%/15%/15% (ASSUMED proportions, conf 0.52).
    The dataset must be sorted by review date before splitting.

    Standardisation: All continuous features standardised EXCEPT:
        - Asin_verified_share
        - Day dummy columns (binary 0/1)

    Args:
        data_path (str):      Path to the preprocessed CSV file.
        train_frac (float):   Fraction for training set. ASSUMED: 0.70.
        val_frac (float):     Fraction for validation set. ASSUMED: 0.15.
        random_state (int):   Seed for undersampling (Exp 4 only).
        date_col (str):       Name of the date/timestamp column for ordering.
                              ASSUMED: 'reviewTime' (common in Amazon datasets).
        rating_col (str):     Name of the star-rating column.
                              ASSUMED: 'overall' (Ni et al. 2019 field name).

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError:        If required columns are missing.
    """

    # Columns that must NOT be standardised (binary / share in [0,1])
    NO_STANDARDISE_PATTERNS: list[str] = ["verified_share", "Day", "_day"]

    def __init__(
        self,
        data_path: str,
        train_frac: float = 0.70,    # ASSUMED: conf 0.52; paper says time-ordered
        val_frac: float = 0.15,      # ASSUMED: conf 0.52
        random_state: int = 42,
        date_col: str = "reviewTime",  # ASSUMED: Ni et al. 2019 field
        rating_col: str = "overall",   # ASSUMED: Ni et al. 2019 field
    ) -> None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"See data/README_data.md for instructions on obtaining the dataset."
            )
        self.data_path = data_path
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac
        self.random_state = random_state
        self.date_col = date_col
        self.rating_col = rating_col

        if not (0 < self.test_frac < 1):
            raise ValueError(
                f"train_frac + val_frac must be < 1, got sum = "
                f"{train_frac + val_frac:.3f}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_experiment(self, experiment_id: int) -> ExperimentData:
        """
        Load and prepare data for one of the four experimental conditions.

        Args:
            experiment_id: Integer in {1, 2, 3, 4}.

        Returns:
            ExperimentData with preprocessed train/val/test splits.
        """
        if experiment_id not in {1, 2, 3, 4}:
            raise ValueError(
                f"experiment_id must be in {{1, 2, 3, 4}}, got {experiment_id}"
            )

        logger.info("Loading experiment %d from %s", experiment_id, self.data_path)
        df = self._load_and_sort()
        df = self._encode_target(df, experiment_id)

        feature_cols = self._get_feature_columns(df)
        logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.int64)

        # Time-ordered split (no shuffling)
        n = len(X)
        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test,  y_test  = X[n_train + n_val:], y[n_train + n_val:]

        # Balance classes via undersampling for Exp 4
        if experiment_id == 4:
            X_train, y_train = self._undersample(X_train, y_train)

        # Fit scaler on train, apply to train/val/test
        no_scale_cols = self._no_standardise_mask(feature_cols)
        scaler = SelectiveStandardScaler(skip_mask=no_scale_cols)
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        num_classes = int(y.max()) + 1
        logger.info(
            "Exp %d: train=%d val=%d test=%d classes=%d",
            experiment_id, len(X_train), len(X_val), len(X_test), num_classes,
        )

        return ExperimentData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            num_classes=num_classes,
            experiment_id=experiment_id,
            scaler=scaler,
        )

    def load_all_experiments(self) -> dict[int, ExperimentData]:
        """
        Load all four experimental conditions and return as a dict keyed by ID.
        """
        return {i: self.load_experiment(i) for i in range(1, 5)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_sort(self) -> pd.DataFrame:
        """Load CSV and sort by date column for time-ordered splitting."""
        df = pd.read_csv(self.data_path)
        required_cols = {self.rating_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns missing from CSV: {missing}. "
                f"Available: {list(df.columns)}"
            )

        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
            df = df.sort_values(self.date_col).reset_index(drop=True)
            logger.info("Data sorted by '%s' for time-ordered split.", self.date_col)
        else:
            logger.warning(
                "Date column '%s' not found in CSV. "
                "Assuming data is already in temporal order.",
                self.date_col,
            )
        return df

    def _encode_target(self, df: pd.DataFrame, experiment_id: int) -> pd.DataFrame:
        """
        Encode the star rating into ordinal target classes for each experiment.

        Experiment encodings:
            Exp 1: star 1-5 → class 0-4 (5-class)
            Exp 2: {1,2}→0, {3}→1, {4,5}→2  (3-class imbalanced)
            Exp 3: {1,2,3}→0, {4}→1, {5}→2  (3-class modified imbalanced)
                   WARNING: Exp 3 mapping is ASSUMED (conf 0.52); paper description
                   is ambiguous about the exact boundaries.
            Exp 4: Same mapping as Exp 3, then balanced via undersampling
        """
        df = df.copy()
        stars = df[self.rating_col].astype(int)

        if experiment_id == 1:
            # 5-class: 1→0, 2→1, 3→2, 4→3, 5→4
            df["target"] = stars - 1

        elif experiment_id == 2:
            # 3-class imbalanced: {1,2}→0, {3}→1, {4,5}→2
            mapping = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
            df["target"] = stars.map(mapping)

        elif experiment_id in (3, 4):
            # WARNING: ASSUMED mapping (conf 0.52) — paper says "modified-3-class"
            # but does not give explicit boundaries. Using {1,2,3}→0, {4}→1, {5}→2.
            # TODO: Verify this mapping against the paper's Table 1.
            mapping = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}
            df["target"] = stars.map(mapping)

        # Drop rows with unmapped targets
        before = len(df)
        df = df.dropna(subset=["target"]).copy()
        df["target"] = df["target"].astype(int)
        after = len(df)
        if after < before:
            logger.warning("Dropped %d rows with unmapped star ratings.", before - after)

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Infer feature columns dynamically from the DataFrame.

        Excludes: target column, date column, rating column, and other
        non-feature columns (e.g. review text, reviewer ID).

        Feature detection heuristic: include columns that are numeric and
        match known feature name patterns from the paper, or all numeric
        columns if no pattern match.

        Never hardcodes K — feature count is always inferred from data.
        """
        known_patterns = [
            "Price", "price",
            "Asin_n_rev", "asin_n_rev",
            "Asin_n_reviewers", "asin_n_reviewers",
            "Asin_verified_share", "asin_verified_share", "verified_share",
            "User_n_rev", "user_n_rev",
            "User_reviewed_asin", "user_reviewed_asin",
            "Day", "day", "_day",
        ]
        exclude_cols = {
            "target", self.date_col, self.rating_col,
            "reviewerID", "asin", "reviewText", "summary",
            "vote", "image", "style",
        }

        # Try to find columns matching known patterns
        candidate_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not candidate_cols:
            raise ValueError(
                "No numeric feature columns found in the dataset. "
                "Check that the CSV has been preprocessed correctly."
            )

        return candidate_cols

    def _no_standardise_mask(self, feature_cols: list[str]) -> list[bool]:
        """
        Return a boolean mask indicating which columns should NOT be standardised.

        Columns skipped: those matching NO_STANDARDISE_PATTERNS.
            - Asin_verified_share: share in [0,1], no need to standardise
            - Day dummies:         binary 0/1
        """
        mask = []
        for col in feature_cols:
            skip = any(
                pattern.lower() in col.lower()
                for pattern in self.NO_STANDARDISE_PATTERNS
            )
            mask.append(skip)
        return mask

    def _undersample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Undersample majority classes to match the size of the minority class.

        Used only for Experiment 4 (balanced 3-class). Preserves temporal order
        within each class stratum.

        Args:
            X: Feature matrix (N, K).
            y: Labels (N,).

        Returns:
            (X_balanced, y_balanced) with equal class counts.
        """
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        logger.info(
            "Undersampling to %d samples per class. Original counts: %s",
            min_count, dict(zip(classes, counts))
        )

        indices_per_class = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            selected = rng.choice(cls_idx, size=min_count, replace=False)
            selected.sort()  # preserve temporal order within class
            indices_per_class.append(selected)

        all_indices = np.concatenate(indices_per_class)
        all_indices.sort()  # restore global temporal order

        return X[all_indices], y[all_indices]

    def __repr__(self) -> str:
        return (
            f"ExperimentSplitter(data_path='{self.data_path}', "
            f"train_frac={self.train_frac}, val_frac={self.val_frac})"
        )
