"""
hopann.models.baselines — Baseline classifiers for comparison with OPANN/HOPANN.

Implements five baseline models as described in the paper's experimental section:
    1. OrderedProbitBaseline  — statsmodels OrderedModel (distr='probit')
    2. ANNBaseline            — standard PyTorch feedforward with cross-entropy
    3. SVMBaseline            — sklearn SVC with probability=True
    4. RandomForestBaseline   — sklearn RandomForestClassifier
    5. XGBoostBaseline        — xgboost XGBClassifier

All baselines share a common interface via BaselineModel (ABC) with fit(),
predict(), predict_proba(), and get_params() methods.

These models are scikit-learn/statsmodels wrappers designed to be interchangeable
with the PyTorch OPANN/HOPANN pipeline for fair comparison on the same splits.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaselineModel(abc.ABC):
    """
    Abstract base class for all baseline classifiers.

    Defines the common interface used in run_baselines.py for fair comparison
    with OPANN and HOPANN on the Amazon review ordinal classification task.

    All concrete subclasses must implement:
        fit(X_train, y_train)     → self
        predict(X)                → np.ndarray of class indices (0-indexed)
        predict_proba(X)          → np.ndarray of shape (N, J) probabilities
        get_params()              → dict of hyperparameters for logging

    Args:
        num_classes (int): J — number of ordinal classes (inferred from data).
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, num_classes: int, random_state: int = 42) -> None:
        self.num_classes = num_classes
        self.random_state = random_state
        self._fitted = False

    @abc.abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineModel":
        """Fit the model on training data."""
        ...

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices (0-indexed) for each sample in X."""
        ...

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates of shape (N, J) for each class."""
        ...

    @abc.abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return a dict of model hyperparameters for logging/reproducibility."""
        ...

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}.fit() must be called before predict()."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes})"


# =============================================================================
# 1. Ordered Probit Baseline (statsmodels)
# =============================================================================

class OrderedProbitBaseline(BaselineModel):
    """
    Classical ordered probit model via statsmodels OrderedModel.

    Uses statsmodels' maximum-likelihood estimation with a probit link function.
    This is the traditional econometric baseline that OPANN and HOPANN improve upon.

    Reference: statsmodels.miscmodels.ordinal_model.OrderedModel(distr='probit')
    Paper reference: Section 2 (traditional ordered probit model).

    Args:
        num_classes (int): J — number of ordinal classes.
        random_state (int): Seed (passed to optimiser for reproducibility where possible).
        method (str): Optimisation method for statsmodels fit. Default 'bfgs'.
        disp (bool): Whether to display optimisation output.
    """

    def __init__(
        self,
        num_classes: int,
        random_state: int = 42,
        method: str = "bfgs",
        disp: bool = False,
    ) -> None:
        super().__init__(num_classes=num_classes, random_state=random_state)
        self.method = method
        self.disp = disp
        self._result = None  # fitted OrderedResults object

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "OrderedProbitBaseline":
        """
        Fit the ordered probit model via MLE.

        Args:
            X_train: Feature matrix of shape (N, K).
            y_train: Integer class labels of shape (N,) — 0-indexed.

        Returns:
            self (fitted)
        """
        try:
            import pandas as pd
            from statsmodels.miscmodels.ordinal_model import OrderedModel
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for OrderedProbitBaseline. "
                "Install with: pip install statsmodels"
            ) from exc

        logger.info("Fitting OrderedProbitBaseline via MLE (%s)...", self.method)
        # statsmodels expects ordered categories; convert to pandas Categorical
        y_cat = pd.Categorical(y_train, categories=sorted(np.unique(y_train)), ordered=True)
        model = OrderedModel(y_cat, X_train, distr="probit")
        self._result = model.fit(method=self.method, disp=self.disp)
        self._classes = sorted(np.unique(y_train))
        self._fitted = True
        logger.info("OrderedProbitBaseline fitted. AIC=%.4f", self._result.aic)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices (0-indexed) as argmax of proba."""
        self._check_fitted()
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates of shape (N, J).

        Uses statsmodels' predict() which returns the J conditional probabilities
        for each observation.
        """
        self._check_fitted()
        proba = self._result.predict(X)  # returns (N, J) as DataFrame
        if hasattr(proba, "values"):
            proba = proba.values
        return proba.astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": "OrderedProbit",
            "method": self.method,
            "distr": "probit",
            "aic": float(self._result.aic) if self._result is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"OrderedProbitBaseline(num_classes={self.num_classes}, "
            f"method='{self.method}')"
        )


# =============================================================================
# 2. ANN Baseline (PyTorch cross-entropy)
# =============================================================================

class ANNBaseline(BaselineModel):
    """
    Standard feedforward ANN baseline with softmax output and cross-entropy loss.

    This serves as a direct comparison to OPANN/HOPANN to quantify the benefit
    of the ordered probit loss over a standard classification loss.

    Architecture:
        Input → Linear(K, Q) → Sigmoid → Linear(Q, J) → Softmax
        ASSUMED: same Q=16 hidden units as OPANN (conf 0.52)
        ASSUMED: lr=1e-3, batch_size=64, epochs=200 (conf 0.52)

    Args:
        num_classes (int):   J — number of ordinal classes.
        hidden_dim (int):    Hidden layer size. ASSUMED: 16 (conf 0.52).
        lr (float):          Learning rate. ASSUMED: 1e-3 (conf 0.52).
        batch_size (int):    Training batch size. ASSUMED: 64 (conf 0.52).
        max_epochs (int):    Maximum training epochs. ASSUMED: 200.
        patience (int):      Early stopping patience. ASSUMED: 15 (conf 0.52).
        random_state (int):  Random seed.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 16,       # ASSUMED: Q=16 (conf 0.52)
        lr: float = 1e-3,           # ASSUMED: conf 0.52
        batch_size: int = 64,       # ASSUMED: conf 0.52
        max_epochs: int = 200,      # ASSUMED: not specified in paper
        patience: int = 15,         # ASSUMED: conf 0.52
        random_state: int = 42,
    ) -> None:
        super().__init__(num_classes=num_classes, random_state=random_state)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self._model = None
        self._input_dim: int | None = None

    def _build_model(self, input_dim: int) -> "torch.nn.Module":  # type: ignore[name-defined]
        """Construct the ANN architecture."""
        import torch.nn as nn
        self._input_dim = input_dim
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ANNBaseline":
        """
        Train the ANN with cross-entropy loss and early stopping on validation loss.

        Uses a fixed 90/10 train/val split within the provided training set for
        early stopping. No stratification — time-order is preserved.

        Args:
            X_train: Feature matrix (N, K).
            y_train: Integer class labels (N,), 0-indexed.

        Returns:
            self (fitted)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        logger.info("Fitting ANNBaseline (hidden_dim=%d, lr=%g)...", self.hidden_dim, self.lr)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build model
        input_dim = X_train.shape[1]
        self._model = self._build_model(input_dim)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Validation split (last 10% of training set — time-ordered)
        n_val = max(1, int(0.1 * len(X_train)))
        X_t = torch.tensor(X_train[:-n_val], dtype=torch.float32)
        y_t = torch.tensor(y_train[:-n_val], dtype=torch.long)
        X_v = torch.tensor(X_train[-n_val:], dtype=torch.float32)
        y_v = torch.tensor(y_train[-n_val:], dtype=torch.long)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._model.train()
        for epoch in range(self.max_epochs):
            # Mini-batch SGD
            permutation = torch.randperm(X_t.size(0))
            for start in range(0, X_t.size(0), self.batch_size):
                idx = permutation[start: start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_v)
                val_loss = criterion(val_logits, y_v).item()
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info("ANNBaseline early stopping at epoch %d.", epoch + 1)
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        self._fitted = True
        logger.info("ANNBaseline fitted. Best val_loss=%.4f", best_val_loss)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices (argmax of softmax logits)."""
        self._check_fitted()
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probability estimates of shape (N, J)."""
        import torch
        import torch.nn.functional as F

        self._check_fitted()
        self._model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self._model(x_t)
            proba = F.softmax(logits, dim=-1).numpy()
        return proba.astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": "ANNBaseline",
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }

    def __repr__(self) -> str:
        return (
            f"ANNBaseline(num_classes={self.num_classes}, "
            f"hidden_dim={self.hidden_dim}, lr={self.lr})"
        )


# =============================================================================
# 3. SVM Baseline
# =============================================================================

class SVMBaseline(BaselineModel):
    """
    Support Vector Machine baseline using sklearn SVC with probability estimates.

    Uses Platt scaling (probability=True) to produce class probabilities for
    ROC AUC and PR AUC computation.

    Args:
        num_classes (int): J — number of ordinal classes.
        C (float):         Regularisation parameter. ASSUMED: 1.0 (sklearn default).
        kernel (str):      Kernel type. ASSUMED: 'rbf' (sklearn default).
        random_state (int): Seed.
    """

    def __init__(
        self,
        num_classes: int,
        C: float = 1.0,       # ASSUMED: sklearn default
        kernel: str = "rbf",  # ASSUMED: sklearn default
        random_state: int = 42,
    ) -> None:
        super().__init__(num_classes=num_classes, random_state=random_state)
        self.C = C
        self.kernel = kernel
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMBaseline":
        """Fit SVC with Platt scaling."""
        from sklearn.svm import SVC

        logger.info("Fitting SVMBaseline (C=%g, kernel=%s)...", self.C, self.kernel)
        self._model = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            random_state=self.random_state,
        )
        self._model.fit(X_train, y_train)
        self._fitted = True
        logger.info("SVMBaseline fitted.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": "SVM",
            "C": self.C,
            "kernel": self.kernel,
        }

    def __repr__(self) -> str:
        return (
            f"SVMBaseline(num_classes={self.num_classes}, "
            f"C={self.C}, kernel='{self.kernel}')"
        )


# =============================================================================
# 4. Random Forest Baseline
# =============================================================================

class RandomForestBaseline(BaselineModel):
    """
    Random Forest baseline using sklearn RandomForestClassifier.

    Args:
        num_classes (int):  J — number of ordinal classes.
        n_estimators (int): Number of trees. ASSUMED: 100 (sklearn default).
        max_depth (int|None): Max tree depth. ASSUMED: None (unlimited).
        random_state (int): Seed.
    """

    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 100,      # ASSUMED: sklearn default
        max_depth: int | None = None, # ASSUMED: unlimited
        random_state: int = 42,
    ) -> None:
        super().__init__(num_classes=num_classes, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseline":
        """Fit RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier

        logger.info(
            "Fitting RandomForestBaseline (n_estimators=%d)...", self.n_estimators
        )
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        self._fitted = True
        logger.info("RandomForestBaseline fitted.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": "RandomForest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }

    def __repr__(self) -> str:
        return (
            f"RandomForestBaseline(num_classes={self.num_classes}, "
            f"n_estimators={self.n_estimators})"
        )


# =============================================================================
# 5. XGBoost Baseline
# =============================================================================

class XGBoostBaseline(BaselineModel):
    """
    XGBoost gradient boosting baseline.

    Args:
        num_classes (int):   J — number of ordinal classes.
        n_estimators (int):  Number of boosting rounds. ASSUMED: 100.
        max_depth (int):     Max tree depth. ASSUMED: 6 (XGBoost default).
        learning_rate (float): Boosting learning rate. ASSUMED: 0.3 (XGBoost default).
        random_state (int):  Seed.
    """

    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 100,     # ASSUMED: reasonable default
        max_depth: int = 6,          # ASSUMED: XGBoost default
        learning_rate: float = 0.3,  # ASSUMED: XGBoost default (eta)
        random_state: int = 42,
    ) -> None:
        super().__init__(num_classes=num_classes, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostBaseline":
        """Fit XGBClassifier with multiclass softmax."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostBaseline. "
                "Install with: pip install xgboost"
            ) from exc

        objective = "multi:softprob" if self.num_classes > 2 else "binary:logistic"
        logger.info(
            "Fitting XGBoostBaseline (n_estimators=%d, objective=%s)...",
            self.n_estimators, objective
        )
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=objective,
            num_class=self.num_classes if self.num_classes > 2 else None,
            random_state=self.random_state,
            eval_metric="mlogloss",
            verbosity=0,
            use_label_encoder=False,
        )
        self._model.fit(X_train, y_train)
        self._fitted = True
        logger.info("XGBoostBaseline fitted.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": "XGBoost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }

    def __repr__(self) -> str:
        return (
            f"XGBoostBaseline(num_classes={self.num_classes}, "
            f"n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate})"
        )
