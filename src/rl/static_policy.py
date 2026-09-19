"""Static rule-based response baseline (B2)."""

from __future__ import annotations

from typing import Any, Dict

from agents.autonomous_agent import DetectionResult, ResponseAction, ACTION_SEVERITY


class StaticResponsePolicy:
    def __init__(self, config: Dict[str, Any]):
        baseline = config.get("static_policy_baseline", {})
        self.block_confidence = float(baseline.get("block_confidence_threshold", 0.9))
        self.block_trust = float(baseline.get("block_trust_threshold", 0.6))
        self.alert_confidence = float(baseline.get("alert_confidence_threshold", 0.75))

    def select_action(self, detection: DetectionResult, trust_score: float) -> ResponseAction:
        if detection.predicted_label == 1 and detection.confidence >= self.block_confidence and trust_score >= self.block_trust:
            action_type = "block"
        elif detection.predicted_label == 1 and detection.confidence >= self.alert_confidence:
            action_type = "alert"
        else:
            action_type = "monitor"

        return ResponseAction(
            action_type=action_type,
            severity=ACTION_SEVERITY.get(action_type, 0.1),
            confidence=detection.confidence,
            agent_id=detection.agent_id,
            round_num=detection.round_num,
            target=getattr(detection, "device_type", "") or "",
        )
