"""Client-side sufficient statistics for federated logistic regression (no row upload)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sample_local_batch(
    X: pd.DataFrame,
    y: pd.Series,
    batch_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample from local data (client-side only)."""
    n = len(X)
    if n <= batch_size:
        return X, y
    from sklearn.utils import resample

    strat = y if len(np.unique(y)) > 1 else None
    X_b, y_b = resample(
        X,
        y,
        n_samples=batch_size,
        random_state=random_state,
        stratify=strat,
    )
    return X_b, y_b


def compute_logistic_gradient_stat(
    X: pd.DataFrame,
    y: pd.Series,
    w_global: np.ndarray,
    feature_columns: list,
) -> np.ndarray:
    """
    Mean logistic gradient at w_global on local batch.

    Returns g in R^{d+1} (bias last).
    """
    Xa = X.reindex(columns=feature_columns, fill_value=0.0)
    Xn = Xa.to_numpy(dtype=float, copy=False)
    yn = y.to_numpy(dtype=int, copy=False)
    if Xn.shape[0] == 0:
        return np.zeros(w_global.shape[0], dtype=float)

    Xb = np.hstack([Xn, np.ones((Xn.shape[0], 1), dtype=float)])
    z = Xb @ w_global
    p = _sigmoid(z)
    err = p - yn
    grad = (Xb * err[:, None]).mean(axis=0)
    return grad.astype(float)


def compute_logistic_irls_stats(
    X: pd.DataFrame,
    y: pd.Series,
    w_global: np.ndarray,
    feature_columns: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Newton gradient and Hessian at w_global on local batch.

    Returns (g, H) each in R^{d+1} (bias last).
    """
    Xa = X.reindex(columns=feature_columns, fill_value=0.0)
    Xn = Xa.to_numpy(dtype=float, copy=False)
    yn = y.to_numpy(dtype=int, copy=False)
    d1 = w_global.shape[0]
    if Xn.shape[0] == 0:
        return np.zeros(d1, dtype=float), np.zeros((d1, d1), dtype=float)

    Xb = np.hstack([Xn, np.ones((Xn.shape[0], 1), dtype=float)])
    z = Xb @ w_global
    p = _sigmoid(z)
    err = p - yn
    w_diag = p * (1.0 - p)
    n = Xb.shape[0]
    g = (Xb * err[:, None]).mean(axis=0)
    H = (Xb * w_diag[:, None]).T @ Xb / n
    return g.astype(float), H.astype(float)
