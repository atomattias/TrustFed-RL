"""
AutonomousAgent wraps FederatedClient with detection and response interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from federated_client import FederatedClient
from local_training import evaluate_model


@dataclass
class DetectionResult:
    predicted_label: int
    confidence: float
    attack_class: Optional[str]
    agent_id: str
    round_num: int
    ground_truth: Optional[int] = None
    device_type: str = "gateway"
    # Master MDP extras (§III-F): Ai / Ci / Ri components
    attack_severity: float = 0.0
    device_criticality: float = 0.5
    contextual_risk: float = 0.0


@dataclass
class ResponseAction:
    action_type: str
    severity: float
    confidence: float
    agent_id: str
    round_num: int
    target: str = ""


# Master action space A = {Monitor, Alert, Throttle, Isolate, Block, Escalate}
ACTION_SEVERITY = {
    "monitor": 0.1,
    "alert": 0.2,
    "throttle": 0.4,
    "isolate": 0.9,
    "block": 0.7,
    "escalate": 0.3,
}

# Map older honeypot/handoff action names → master actions
LEGACY_ACTION_MAP = {
    "soft_block": "block",
    "hard_block": "block",
    "deception_escalate": "escalate",
    "no_action": "monitor",  # DB NO_ACTION → effector no-op (severity set separately)
}


def normalize_action_type(action_type: str) -> str:
    a = str(action_type).strip().lower()
    return LEGACY_ACTION_MAP.get(a, a)


# Device criticality C_i ∈ [0, 1] for MDP state
_DEVICE_CRITICALITY = {
    "infusion_pump": 1.0,
    "vitals_monitor": 1.0,
    "ecg": 1.0,
    "patient_monitor": 1.0,
    "imaging": 0.55,
    "gateway": 0.55,
    "mqtt_broker": 0.55,
    "wearable": 0.25,
    "bluetooth_sensor": 0.25,
    "printer": 0.2,
}


def device_criticality_score(device_type: str) -> float:
    return float(_DEVICE_CRITICALITY.get(str(device_type).lower(), 0.5))


# Default IoMT device mix when telemetry has no device column
_DEFAULT_DEVICES = [
    "infusion_pump", "vitals_monitor", "ecg", "patient_monitor",
    "imaging", "gateway", "mqtt_broker",
    "wearable", "bluetooth_sensor", "printer",
]


class AutonomousAgent:
    """Distributed hospital agent for detection, FL participation, and response."""

    def __init__(
        self,
        client: FederatedClient,
        global_model: Any = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.agent_id = client.client_id
        self.global_model = global_model
        self.config = agent_config or {}
        self.response_history: List[Dict[str, Any]] = []
        self.device_types: List[str] = list(
            self.config.get("device_types", _DEFAULT_DEVICES)
        )

    def observe(self) -> Tuple[Any, np.ndarray]:
        return self.client.X_val, self.client.y_val

    def _align_features(self, X: Any) -> Any:
        model = self.global_model
        if model is None or not hasattr(model, "feature_names_in_"):
            return X
        cols = list(model.feature_names_in_)
        if isinstance(X, pd.DataFrame):
            return X.reindex(columns=cols, fill_value=0.0)
        return X

    def set_global_model(self, model: Any) -> None:
        """Store the aggregated model for detection only.

        Do not overwrite ``client.model`` — that is the last *local* train and
        must keep matching ``val_metrics`` (V). Detection uses ``global_model``.
        """
        self.global_model = model

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        round_num: int = 0,
    ) -> List[DetectionResult]:
        if self.global_model is None:
            raise ValueError(f"Agent {self.agent_id}: global model not set")

        X = self._align_features(X)
        model = self.global_model
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            Xn = np.asarray(X, dtype=float)
            w = np.asarray(model.coef_, dtype=float).reshape(-1)
            b = float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0])
            z = (Xn * w).sum(axis=1) + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            preds = (p >= 0.5).astype(int)
            confidences = np.maximum(p, 1.0 - p)
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            preds = model.predict(X)
            confidences = proba.max(axis=1)
        else:
            preds = model.predict(X)
            confidences = np.ones(len(preds))

        y_true = np.asarray(y).reshape(-1) if y is not None else None
        results: List[DetectionResult] = []
        devices = self.device_types or _DEFAULT_DEVICES
        for i in range(len(preds)):
            gt = int(y_true[i]) if y_true is not None else None
            device = devices[i % len(devices)]
            conf = float(confidences[i])
            pred = int(preds[i])
            results.append(
                DetectionResult(
                    predicted_label=pred,
                    confidence=conf,
                    attack_class="attack" if pred == 1 else "benign",
                    agent_id=self.agent_id,
                    round_num=round_num,
                    ground_truth=gt,
                    device_type=device,
                    attack_severity=conf if pred == 1 else 0.0,
                    device_criticality=device_criticality_score(device),
                )
            )
        return results

    def train_local(self) -> Dict[str, Any]:
        return self.client.get_model_update()

    def report_behavior(self) -> Dict[str, Any]:
        return {
            "client_id": self.agent_id,
            "trust_score": getattr(self.client, "trust_score", 0.5),
            "val_metrics": getattr(self.client, "val_metrics", {}),
            "performance_history": getattr(self.client, "performance_history", []),
        }

    def propose_response(
        self,
        detection: DetectionResult,
        policy: str = "static",
    ) -> ResponseAction:
        if policy == "static":
            if detection.confidence > 0.9 and detection.predicted_label == 1:
                action_type = "block"
            elif detection.confidence > 0.75 and detection.predicted_label == 1:
                action_type = "alert"
            else:
                action_type = "monitor"
        else:
            action_type = "monitor"

        return ResponseAction(
            action_type=action_type,
            severity=ACTION_SEVERITY.get(action_type, 0.1),
            confidence=detection.confidence,
            agent_id=self.agent_id,
            round_num=detection.round_num,
        )

    def record_response_outcome(self, event: Dict[str, Any]) -> None:
        self.response_history.append(event)
        window = int(self.config.get("response_fidelity_window", 10))
        if len(self.response_history) > window:
            self.response_history = self.response_history[-window:]

    def get_response_fidelity_penalty(self) -> float:
        if not self.response_history:
            return 0.0
        n = len(self.response_history)
        fp_rate = sum(1 for e in self.response_history if e.get("fp_response")) / n
        viol_rate = sum(1 for e in self.response_history if e.get("policy_violation")) / n
        severity_mis = sum(
            e.get("severity", 0.0) for e in self.response_history if e.get("inappropriate_severity")
        ) / n
        return float(0.4 * fp_rate + 0.4 * viol_rate + 0.2 * severity_mis)
