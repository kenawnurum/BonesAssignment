from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

try:
    from .fair_missing_classifier import FairMissingValueClassifier
except ImportError:
    from fair_missing_classifier import FairMissingValueClassifier


def make_missing_fairness_data(n_samples: int = 3000, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=random_state,
    )

    sensitive = rng.integers(0, 2, size=n_samples)

    # Inject group-dependent missingness to mimic the paper's setting.
    missing_prob = 0.08 + 0.18 * sensitive[:, None]
    missing_mask = rng.random(size=X.shape) < missing_prob
    X = X.astype(float)
    X[missing_mask] = np.nan

    return X, y.astype(int), sensitive.astype(int)


def main() -> None:
    X, y, sensitive = make_missing_fairness_data()
    model = FairMissingValueClassifier(fairness_weight=1.5)
    model.fit(X, y, sensitive)
    metrics = model.evaluate(X, y, sensitive)

    print("Group thresholds:", model.group_thresholds)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()