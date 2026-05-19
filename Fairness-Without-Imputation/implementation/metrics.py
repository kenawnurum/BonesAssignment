from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score


def _validate_binary_sensitive(sensitive: np.ndarray) -> tuple[int, int]:
    groups = np.unique(sensitive)
    if len(groups) != 2:
        raise ValueError("This implementation expects a binary sensitive attribute.")
    return int(groups[0]), int(groups[1])


def demographic_parity_difference(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """Absolute difference in positive prediction rates between the two groups."""
    g0, g1 = _validate_binary_sensitive(sensitive)
    rate_0 = float(y_pred[sensitive == g0].mean()) if np.any(sensitive == g0) else 0.0
    rate_1 = float(y_pred[sensitive == g1].mean()) if np.any(sensitive == g1) else 0.0
    return abs(rate_0 - rate_1)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Maximum absolute difference in TPR and FPR across the two groups."""
    g0, g1 = _validate_binary_sensitive(sensitive)

    def _rate(mask: np.ndarray) -> float:
        return float(y_pred[mask].mean()) if np.any(mask) else 0.0

    tpr_0 = _rate((sensitive == g0) & (y_true == 1))
    tpr_1 = _rate((sensitive == g1) & (y_true == 1))
    fpr_0 = _rate((sensitive == g0) & (y_true == 0))
    fpr_1 = _rate((sensitive == g1) & (y_true == 0))

    return max(abs(tpr_0 - tpr_1), abs(fpr_0 - fpr_1))


@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    demographic_parity_difference: float
    equalized_odds_difference: float


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> MetricBundle:
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        demographic_parity_difference=demographic_parity_difference(y_pred, sensitive),
        equalized_odds_difference=equalized_odds_difference(y_true, y_pred, sensitive),
    )