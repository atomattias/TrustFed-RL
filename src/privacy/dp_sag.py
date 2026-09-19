"""Simulated Secure Aggregation + Gaussian DP on gradient statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DPSAGConfig:
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5
    learning_rate: float = 0.05
    secure_aggregation: bool = True
    trust_beta: float = 0.8
    hessian_reg: float = 0.01
    inner_steps: int = 1


def clip_vector(v: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= clip_norm or norm == 0.0:
        return v.astype(float, copy=False)
    return (v * (clip_norm / norm)).astype(float)


def _trust_weights(trust_scores: Sequence[float], beta: float) -> np.ndarray:
    t = np.maximum(np.asarray(trust_scores, dtype=float), 0.0)
    powered = t ** beta
    s = powered.sum()
    if s <= 0:
        return np.ones(len(t), dtype=float) / max(len(t), 1)
    return powered / s


def aggregate_clipped_gradients(
    client_gradients: List[np.ndarray],
    trust_scores: Optional[Sequence[float]],
    config: DPSAGConfig,
    rng: np.random.Generator,
    equal_weight: bool = False,
) -> Dict[str, np.ndarray | float]:
    """
    SA (simulated sum) + optional DP noise on weighted mean gradient.

    Returns dict with aggregated_grad, noisy_grad, per-client norms.
    """
    if not client_gradients:
        raise ValueError("No client gradients provided")

    clipped = [clip_vector(g, config.clip_norm) for g in client_gradients]

    if equal_weight or trust_scores is None:
        weights = np.ones(len(clipped), dtype=float) / len(clipped)
    else:
        weights = _trust_weights(trust_scores, config.trust_beta)

    # Weighted mean (simulated SA reveals only this aggregate)
    stacked = np.stack(clipped, axis=0)
    mean_grad = (weights[:, None] * stacked).sum(axis=0)

    noisy_grad = mean_grad.copy()
    if config.noise_multiplier > 0:
        sigma = config.noise_multiplier * config.clip_norm
        noisy_grad = mean_grad + rng.normal(0.0, sigma, size=mean_grad.shape)

    return {
        "mean_grad": mean_grad,
        "noisy_grad": noisy_grad,
        "weights": weights,
        "client_norms": [float(np.linalg.norm(g)) for g in client_gradients],
    }


def aggregate_trust_weighted_hessian_grad(
    client_grads: List[np.ndarray],
    client_hessians: List[np.ndarray],
    trust_scores: Optional[Sequence[float]],
    config: DPSAGConfig,
    rng: np.random.Generator,
    equal_weight: bool = False,
) -> Dict[str, np.ndarray]:
    """Trust-weighted mean of clipped (g, H) with optional DP noise on g only (v1)."""
    if not client_grads or not client_hessians:
        raise ValueError("Missing client gradients or Hessians")

    clipped_g = [clip_vector(g, config.clip_norm) for g in client_grads]
    # Frobenius clip on Hessian per client
    clipped_h = []
    for H in client_hessians:
        norm = float(np.linalg.norm(H, ord="fro"))
        if norm <= config.clip_norm or norm == 0.0:
            clipped_h.append(H.astype(float, copy=False))
        else:
            clipped_h.append((H * (config.clip_norm / norm)).astype(float))

    if equal_weight or trust_scores is None:
        weights = np.ones(len(clipped_g), dtype=float) / len(clipped_g)
    else:
        weights = _trust_weights(trust_scores, config.trust_beta)

    mean_g = sum(w * g for w, g in zip(weights, clipped_g))
    mean_h = sum(w * H for w, H in zip(weights, clipped_h))

    noisy_g = mean_g.copy()
    if config.noise_multiplier > 0:
        sigma = config.noise_multiplier * config.clip_norm
        noisy_g = mean_g + rng.normal(0.0, sigma, size=mean_g.shape)

    return {"mean_grad": mean_g, "noisy_grad": noisy_g, "mean_hessian": mean_h, "weights": weights}


def irls_newton_step(
    grad: np.ndarray,
    hessian: np.ndarray,
    hessian_reg: float,
    learning_rate: float,
) -> np.ndarray:
    """Compute damped Newton step delta = lr * (H + reg I)^{-1} g."""
    d = grad.shape[0]
    A = hessian + hessian_reg * np.eye(d, dtype=float)
    try:
        delta = np.linalg.solve(A, grad)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(A, grad, rcond=None)[0]
    return learning_rate * delta


def estimate_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    num_clients: int,
    delta: float,
    clip_norm: float = 1.0,
) -> Optional[float]:
    """
    Rough Gaussian-mechanism order-of-magnitude ε (not a full accountant).

    Returns None if noise_multiplier <= 0.
    """
    if noise_multiplier <= 0:
        return None
    # q=1 full participation per round
    q = 1.0
    # Simplified: sigma effective scales with noise_multiplier * C
    sigma = noise_multiplier * clip_norm
    if sigma <= 0:
        return None
    # Order-of-magnitude bound (McMahan-style heuristic for reporting only)
    steps = num_rounds * q
    eps = math.sqrt(2 * steps * math.log(1.25 / delta)) / sigma
    return float(eps)
