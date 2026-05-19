"""CLI runner for the Fairness-without-Imputation implementation.

This is a compact entrypoint inspired by typical `main.py` scripts: it
supports dataset selection, training, evaluation, and model saving.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np

try:
    from implementation.run_demo import make_missing_fairness_data
    from implementation.fair_missing_classifier import FairMissingValueClassifier
except Exception:  # pragma: no cover - resilient import for different execution contexts
    from run_demo import make_missing_fairness_data
    from fair_missing_classifier import FairMissingValueClassifier


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fairness without Imputation - runner")
    p.add_argument("--dataset", choices=["synthetic", "adult", "compas"], default="synthetic")
    p.add_argument("--out", type=str, default="results/model.joblib")
    p.add_argument("--lambda-fair", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def ensure_outdir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.info("Starting run with args: %s", args)

    if args.dataset == "synthetic":
        X, y, sensitive = make_missing_fairness_data(random_state=args.random_state)
    elif args.dataset == "adult":
        try:
            from implementation.data_loaders import load_adult
        except Exception:
            from data_loaders import load_adult

        X, y, sensitive = load_adult()
    elif args.dataset == "compas":
        try:
            from implementation.data_loaders import load_compas
        except Exception:
            from data_loaders import load_compas

        X, y, sensitive = load_compas()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    clf = FairMissingValueClassifier(
        fairness_weight=args.lambda_fair, test_size=args.test_size, random_state=args.random_state
    )
    logging.info("Fitting model...")
    clf.fit(X, y, sensitive)

    logging.info("Evaluating on full data...")
    metrics = clf.evaluate(X, y, sensitive)

    ensure_outdir(args.out)
    joblib.dump({"model": clf, "metrics": metrics}, args.out)

    print(json.dumps({"metrics": metrics, "group_thresholds": clf.group_thresholds}, indent=2))
    logging.info("Saved model to %s", args.out)


if __name__ == "__main__":
    main()
