"""Dataset loaders for Adult and COMPAS used by the demo.

Loaders try remote fetch when possible, otherwise fall back to local CSVs in
the `datasets/` folder. Each loader returns `(X, y, sensitive)` where sensitive
is a binary array.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.datasets import fetch_openml
except Exception:
    fetch_openml = None

try:
    import requests
except Exception:
    requests = None


def _to_numpy_features(df: pd.DataFrame) -> np.ndarray:
    # Convert object/categorical columns to one-hot, keep numeric as-is.
    obj = df.select_dtypes(include=[object, "category"]) 
    num = df.select_dtypes(include=["number"]) 
    if not obj.empty:
        obj_encoded = pd.get_dummies(obj, dummy_na=True, drop_first=False)
        out = pd.concat([num, obj_encoded], axis=1)
    else:
        out = num
    return out.values.astype(float)


def load_adult(local_path: str | None = None, sensitive: str = "sex") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the UCI Adult dataset.

    Returns (X, y, sensitive) where `y` is 0/1 for <=50K / >50K and `sensitive`
    is binary based on the `sensitive` column (default `sex` with Male=1).
    """
    df = None
    if fetch_openml is not None:
        try:
            data = fetch_openml("adult", version=2, as_frame=True)
            df = data.frame.copy()
        except Exception:
            df = None

    if df is None:
        path = Path(local_path) if local_path else Path("datasets/adult.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"Adult dataset not found locally at {path}. Try `fetch_openml('adult')` or place the CSV in datasets/"
            )
        df = pd.read_csv(path)

    # Normalize common target column names
    if "class" in df.columns:
        target_col = "class"
    elif "income" in df.columns:
        target_col = "income"
    else:
        # try to find a column with '>' in values
        for c in df.columns:
            if df[c].astype(str).str.contains(">").any():
                target_col = c
                break
        else:
            raise RuntimeError("Could not find target column in Adult dataframe.")

    df.replace("?", np.nan, inplace=True)
    y = (df[target_col].astype(str).str.contains(">50K")).astype(int).values

    if sensitive not in df.columns:
        raise KeyError(f"Sensitive column {sensitive} not found in Adult dataset.")
    s_raw = df[sensitive].astype(str)
    s = (s_raw == "Male").astype(int).values

    # drop target and sensitive from features
    Xdf = df.drop(columns=[target_col, sensitive], errors="ignore")
    X = _to_numpy_features(Xdf)
    return X, y.astype(int), s.astype(int)


def load_compas(local_path: str | None = None, sensitive: str = "race") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the COMPAS dataset from ProPublica or local CSV.

    Attempts to download the canonical COMPAS CSV from ProPublica if `requests`
    is available; otherwise reads `datasets/compas.csv`.
    """
    path = Path(local_path) if local_path else Path("datasets/compas-scores-two-years.csv")
    df = None
    if path.exists():
        df = pd.read_csv(path)
    else:
        # try to fetch from ProPublica repo raw file
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        if requests is not None:
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                from io import StringIO

                df = pd.read_csv(StringIO(r.text))
            except Exception:
                df = None

    if df is None:
        raise FileNotFoundError(
            f"COMPAS dataset not found locally at {path} and could not download. Please place 'compas-scores-two-years.csv' in datasets/."
        )

    # Typical preprocessing used in COMPAS analyses.
    # Use 'two_year_recid' as target when present.
    if "two_year_recid" in df.columns:
        y = df["two_year_recid"].astype(int).values
    elif "two_year" in df.columns:
        y = df["two_year"].astype(int).values
    else:
        raise RuntimeError("Could not find two-year recidivism column in COMPAS CSV.")

    if sensitive not in df.columns:
        raise KeyError(f"Sensitive column {sensitive} not found in COMPAS dataset.")
    s_raw = df[sensitive].astype(str)
    unique = s_raw.unique()
    # Map the two most common groups to 0/1
    if len(unique) >= 2:
        grp0, grp1 = unique[:2]
        s = (s_raw == grp1).astype(int).values
    else:
        s = np.zeros(len(s_raw), dtype=int)

    # Select a small set of features (age, priors_count, c_charge_degree, sex)
    feats = []
    for c in ("age", "priors_count", "c_charge_degree", "sex"):
        if c in df.columns:
            feats.append(c)
    if not feats:
        # fallback: use numeric columns except the target
        num = df.select_dtypes(include=["number"]).columns.tolist()
        feats = [c for c in num if c != "two_year_recid"]

    Xdf = df[feats].copy()
    Xdf.replace("?", np.nan, inplace=True)
    X = _to_numpy_features(Xdf)

    return X, y.astype(int), s.astype(int)
