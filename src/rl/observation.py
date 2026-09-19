"""Shared state encoding for PPO training and inference.

Master MDP (§III-F):
  S_t = {T_i, A_i, C_i, R_i}
  A   = {Monitor, Alert, Throttle, Isolate, Block, Escalate}

Δτ trust calibration is applied outside the action space (trust-evolution channel).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from agents.autonomous_agent import ACTION_SEVERITY, DetectionResult, normalize_action_type

ACTION_TYPES = list(ACTION_SEVERITY.keys())
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
OBS_DIM = 4  # Ti, Ai, Ci, Ri


def build_observation(
    det: DetectionResult,
    trust: float,
    max_rounds: int = 10,
    device_criticality: Optional[float] = None,
    contextual_risk: Optional[float] = None,
) -> np.ndarray:
    """Master 4-dim state vector S_t = {T_i, A_i, C_i, R_i}."""
    del max_rounds  # retained for call-site compatibility
    Ti = float(max(0.0, min(1.0, trust)))
    Ai = float(
        getattr(det, "attack_severity", None)
        if getattr(det, "attack_severity", None) is not None
        else (det.confidence if det.predicted_label == 1 else 0.0)
    )
    Ai = max(0.0, min(1.0, Ai))
    Ci = float(
        device_criticality
        if device_criticality is not None
        else getattr(det, "device_criticality", 0.5)
    )
    Ci = max(0.0, min(1.0, Ci))
    Ri = float(
        contextual_risk
        if contextual_risk is not None
        else getattr(det, "contextual_risk", 0.0)
    )
    Ri = max(0.0, min(1.0, Ri))
    return np.array([Ti, Ai, Ci, Ri], dtype=np.float32)


def build_action_mask(allowed_actions: Optional[List[str]] = None) -> np.ndarray:
    """Boolean mask over ACTION_TYPES; all True when governance does not restrict."""
    mask = np.ones(len(ACTION_TYPES), dtype=bool)
    if allowed_actions is not None:
        allowed = {normalize_action_type(a) for a in allowed_actions}
        for i, action in enumerate(ACTION_TYPES):
            mask[i] = action in allowed
    if not mask.any():
        monitor_idx = ACTION_TO_IDX.get("monitor", 0)
        mask[monitor_idx] = True
    return mask
