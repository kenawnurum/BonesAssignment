from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

try:
    from .metrics import evaluate_predictions
except ImportError:
    from metrics import evaluate_predictions


@dataclass
class FairMissingValueClassifier:
    """Paper-inspired classifier for fairness with missing values.

    The model uses a tree ensemble that handles NaN values natively, then learns
    group-specific decision thresholds on a validation split to reduce fairness gaps.
    """

    test_size: float = 0.2
    random_state: int = 42
    fairness_weight: float = 1.0
    threshold_grid: Iterable[float] = field(
        default_factory=lambda: np.round(np.linspace(0.05, 0.95, 37), 3)
    )
    model_params: Dict | None = None

    base_model: HistGradientBoostingClassifier = field(init=False)
    group_thresholds: Dict[int, float] = field(init=False, default_factory=dict)
    groups_: np.ndarray = field(init=False, default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        params = {
            "learning_rate": 0.05,
            "max_depth": 4,
            "max_iter": 250,
            "random_state": self.random_state,
        }
        if self.model_params:
            params.update(self.model_params)
        self.base_model = HistGradientBoostingClassifier(**params)

    @staticmethod
    def _validate_sensitive(sensitive: np.ndarray) -> np.ndarray:
        groups = np.unique(sensitive)
        if len(groups) != 2:
            raise ValueError("This implementation currently supports binary sensitive attributes.")
        return groups

    def _search_group_thresholds(
        self,
        validation_proba: np.ndarray,
        y_val: np.ndarray,
        sensitive_val: np.ndarray,
    ) -> Dict[int, float]:
        groups = self._validate_sensitive(sensitive_val)
        best_score = float("inf")
        best_thresholds = {int(groups[0]): 0.5, int(groups[1]): 0.5}

        for threshold_0 in self.threshold_grid:
            for threshold_1 in self.threshold_grid:
                candidate = np.zeros_like(y_val)
                candidate[sensitive_val == groups[0]] = (
                    validation_proba[sensitive_val == groups[0]] >= threshold_0
                ).astype(int)
                candidate[sensitive_val == groups[1]] = (
                    validation_proba[sensitive_val == groups[1]] >= threshold_1
                ).astype(int)

                metrics = evaluate_predictions(y_val, candidate, sensitive_val)
                fairness_gap = (
                    metrics.demographic_parity_difference + metrics.equalized_odds_difference
                )
                objective = (1.0 - metrics.accuracy) + self.fairness_weight * fairness_gap

                if objective < best_score:
                    best_score = objective
                    best_thresholds = {
                        int(groups[0]): float(threshold_0),
                        int(groups[1]): float(threshold_1),
                    }

        return best_thresholds

    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> "FairMissingValueClassifier":
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X,
            y,
            sensitive,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.base_model.fit(X_train, y_train)
        validation_proba = self.base_model.predict_proba(X_val)[:, 1]
        self.groups_ = self._validate_sensitive(s_train)
        self.group_thresholds = self._search_group_thresholds(validation_proba, y_val, s_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.base_model, "classes_"):
            raise RuntimeError("The model must be fitted before calling predict_proba().")
        return self.base_model.predict_proba(X)

    def predict(self, X: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        if not self.group_thresholds:
            raise RuntimeError("The model must be fitted before calling predict().")

        proba = self.predict_proba(X)[:, 1]
        groups = self._validate_sensitive(sensitive)
        y_pred = np.zeros(len(proba), dtype=int)

        for group in groups:
            threshold = self.group_thresholds[int(group)]
            mask = sensitive == group
            y_pred[mask] = (proba[mask] >= threshold).astype(int)

        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> dict:
        y_pred = self.predict(X, sensitive)
        metrics = evaluate_predictions(y, y_pred, sensitive)
        return {
            "accuracy": metrics.accuracy,
            "demographic_parity_difference": metrics.demographic_parity_difference,
            "equalized_odds_difference": metrics.equalized_odds_difference,
        }