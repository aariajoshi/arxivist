"""
hopann.evaluation.metrics — Ordinal classification evaluation metrics.

Implements the 7 metrics reported in the paper:
    1. F1-macro         (macro-averaged F1 score across all classes)
    2. Accuracy         (overall classification accuracy)
    3. PR AUC           (Precision-Recall AUC, one-vs-rest macro average)
    4. MSE              (mean squared error of predicted class vs true class)
    5. MAE              (mean absolute error of predicted class vs true class)
    6. Cohen's Kappa    (agreement beyond chance, linear weights for ordinal)
    7. ROC AUC          (one-vs-rest macro average)

All metrics take (y_true, probs) as input and return a dict[str, float].

OvR = one-vs-rest. For ordinal tasks with J classes, each class j is treated as
a binary problem (j vs. all others) and scores are macro-averaged.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class OrdinalMetrics:
    """
    Computes the 7 evaluation metrics reported in the HOPANN paper.

    All metrics are class methods so they can be called without instantiation.

    Paper reference: Experimental evaluation section.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray,
        probs: np.ndarray,
        average: str = "macro",
    ) -> dict[str, float]:
        """
        Compute all 7 ordinal metrics from true labels and predicted probabilities.

        Args:
            y_true:  Integer class labels of shape (N,), 0-indexed.
            probs:   Probability estimates of shape (N, J). Each row should sum to 1.
            average: Averaging strategy for multi-class metrics. Default: 'macro'.

        Returns:
            Dict with keys: 'f1_macro', 'accuracy', 'pr_auc', 'mse', 'mae',
                            'cohen_kappa', 'roc_auc'.
        """
        assert y_true.ndim == 1, f"y_true must be 1D, got shape {y_true.shape}"
        assert probs.ndim == 2, f"probs must be 2D, got shape {probs.shape}"
        assert len(y_true) == len(probs), (
            f"y_true ({len(y_true)}) and probs ({len(probs)}) must have same length"
        )

        y_pred = probs.argmax(axis=1)

        metrics: dict[str, float] = {}

        # 1. F1-macro
        metrics["f1_macro"] = OrdinalMetrics.f1_macro(y_true, y_pred)

        # 2. Accuracy
        metrics["accuracy"] = OrdinalMetrics.accuracy(y_true, y_pred)

        # 3. PR AUC (OvR macro)
        metrics["pr_auc"] = OrdinalMetrics.pr_auc(y_true, probs)

        # 4. MSE
        metrics["mse"] = OrdinalMetrics.mse(y_true, y_pred)

        # 5. MAE
        metrics["mae"] = OrdinalMetrics.mae(y_true, y_pred)

        # 6. Cohen's Kappa (linear weights — appropriate for ordinal scale)
        metrics["cohen_kappa"] = OrdinalMetrics.cohen_kappa(y_true, y_pred)

        # 7. ROC AUC (OvR macro)
        metrics["roc_auc"] = OrdinalMetrics.roc_auc(y_true, probs)

        return metrics

    # ------------------------------------------------------------------
    # Individual metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Macro-averaged F1 score.

        F1_macro = mean over classes of F1_j
        F1_j = 2 * precision_j * recall_j / (precision_j + recall_j)
        """
        from sklearn.metrics import f1_score
        try:
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except Exception as exc:
            logger.warning("f1_macro computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Overall classification accuracy."""
        from sklearn.metrics import accuracy_score
        try:
            return float(accuracy_score(y_true, y_pred))
        except Exception as exc:
            logger.warning("accuracy computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def pr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
        """
        Macro-averaged Precision-Recall AUC (one-vs-rest).

        For multi-class: average the AUC of the PR curve for each class
        treated as a binary problem.
        """
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize

        classes = np.unique(y_true)
        J = probs.shape[1]

        if J == 2:
            # Binary case: use positive class probabilities
            try:
                return float(average_precision_score(y_true, probs[:, 1]))
            except Exception as exc:
                logger.warning("pr_auc (binary) computation failed: %s", exc)
                return float("nan")

        # Multi-class OvR
        try:
            y_bin = label_binarize(y_true, classes=list(range(J)))
            # Ensure probs has J columns even if some classes absent in y_true
            return float(
                average_precision_score(
                    y_bin, probs, average="macro"
                )
            )
        except Exception as exc:
            logger.warning("pr_auc (macro OvR) computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error of predicted class index vs true class index.

        Treats ordinal class labels as integer scores (0, 1, ..., J-1).
        """
        try:
            return float(np.mean((y_true - y_pred) ** 2))
        except Exception as exc:
            logger.warning("mse computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error of predicted class index vs true class index.

        Treats ordinal class labels as integer scores (0, 1, ..., J-1).
        """
        try:
            return float(np.mean(np.abs(y_true - y_pred)))
        except Exception as exc:
            logger.warning("mae computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Cohen's Kappa with linear weights (appropriate for ordinal scales).

        Linear-weighted kappa penalises disagreements proportionally to their
        ordinal distance, which is more informative than unweighted kappa for
        ordered categories.
        """
        from sklearn.metrics import cohen_kappa_score
        try:
            return float(cohen_kappa_score(y_true, y_pred, weights="linear"))
        except Exception as exc:
            logger.warning("cohen_kappa computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
        """
        Macro-averaged ROC AUC (one-vs-rest).

        For multi-class: average the AUC of the ROC curve for each class
        treated as a binary problem.
        """
        from sklearn.metrics import roc_auc_score

        J = probs.shape[1]
        classes = np.unique(y_true)

        if len(classes) < 2:
            logger.warning("roc_auc requires at least 2 distinct classes.")
            return float("nan")

        if J == 2:
            try:
                return float(roc_auc_score(y_true, probs[:, 1]))
            except Exception as exc:
                logger.warning("roc_auc (binary) computation failed: %s", exc)
                return float("nan")

        try:
            return float(
                roc_auc_score(
                    y_true, probs, multi_class="ovr", average="macro"
                )
            )
        except Exception as exc:
            logger.warning("roc_auc (macro OvR) computation failed: %s", exc)
            return float("nan")

    @staticmethod
    def format_table(metrics: dict[str, float]) -> str:
        """
        Format a metrics dict as a readable table string.

        Args:
            metrics: Output from OrdinalMetrics.compute().

        Returns:
            Formatted string with one metric per line.
        """
        lines = ["Metric Results:", "-" * 30]
        for k, v in metrics.items():
            lines.append(f"  {k:<20} {v:.4f}")
        lines.append("-" * 30)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "OrdinalMetrics(f1_macro, accuracy, pr_auc, mse, mae, cohen_kappa, roc_auc)"
