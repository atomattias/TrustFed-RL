"""Fixed-scale integer quantization for LR parameter vectors (not adaptive)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

DEFAULT_SCALE = 1_000_000


def pack_lr_parameters(params: Dict[str, Any]) -> np.ndarray:
    """Flatten logistic-regression ``coef`` and ``intercept`` to one float vector."""
    coef = np.asarray(params["coef"], dtype=np.float64).ravel()
    intercept = np.asarray(params["intercept"], dtype=np.float64).ravel()
    return np.concatenate([coef, intercept])


def unpack_lr_parameters(
    vec: np.ndarray, n_features: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Restore ``coef`` shape ``(1, d)`` and ``intercept`` from a packed vector."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size != n_features + 1:
        raise ValueError(f"expected dim {n_features + 1}, got {v.size}")
    coef = v[:n_features].reshape(1, n_features)
    intercept = v[n_features:]
    return coef, intercept


def quantize(vec: np.ndarray, scale: int = DEFAULT_SCALE) -> np.ndarray:
    scaled = np.rint(np.asarray(vec, dtype=np.float64) * float(scale))
    return np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(
        np.int64
    )


def dequantize(ivec: np.ndarray, scale: int = DEFAULT_SCALE) -> np.ndarray:
    return np.asarray(ivec, dtype=np.float64) / float(scale)
