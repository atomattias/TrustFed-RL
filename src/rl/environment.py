"""Simulated response outcome and reward computation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from agents.autonomous_agent import ACTION_SEVERITY, DetectionResult, ResponseAction


BLOCKING_ACTIONS = {"throttle", "isolate", "block"}


def simulate_action_outcome(
    detection: DetectionResult,
    action: ResponseAction,
    validation_status: str,
) -> Dict[str, Any]:
    gt = detection.ground_truth
    pred = detection.predicted_label
    # Authorized outcomes execute the (possibly substituted) final action.
    # "rejected" = no executable action (should be rare after Eq.2 substitutes).
    executed = validation_status in ("approved", "modified", "escalated", "postponed")

    tp_contain = int(gt == 1 and pred == 1 and executed and action.action_type in BLOCKING_ACTIONS)
    fp_response = int(gt == 0 and pred == 1 and executed and action.action_type in BLOCKING_ACTIONS)
    fn_miss = int(gt == 1 and (pred == 0 or (executed and action.action_type == "monitor")))
    # Violation = unauthorized / blocked with no substitute execution — not a
    # successful governance intervention (REJECT→ALERT counts as compliance).
    policy_violation = int(validation_status == "rejected")
    human_review = int(
        validation_status in ("escalated", "postponed")
        or action.action_type == "escalate"
    )
    inappropriate_severity = int(
        executed and action.severity > 0.6 and detection.confidence < 0.75
    )

    return {
        "tp_contain": tp_contain,
        "fp_response": fp_response,
        "fn_miss": fn_miss,
        "policy_violation": policy_violation,
        "human_review": human_review,
        "fp_response_flag": bool(fp_response),
        "inappropriate_severity": bool(inappropriate_severity),
        "executed": executed,
    }


def compute_reward(
    outcome: Dict[str, Any],
    action: ResponseAction,
    trust_score: float,
    reward_weights: Dict[str, float],
    action_costs: Dict[str, float],
    device_type: str = "",
) -> float:
    w = reward_weights
    cost = float(action_costs.get(action.action_type, 0))
    critical = device_type.lower() in {
        "infusion_pump", "vitals_monitor", "ecg", "patient_monitor",
    }
    patient_risk = int(
        critical and action.action_type in BLOCKING_ACTIONS and outcome.get("executed", False)
    )
    return (
        w.get("w_tp", 10.0) * outcome.get("tp_contain", 0)
        + w.get("w_fp", -15.0) * outcome.get("fp_response", 0)
        + w.get("w_fn", -20.0) * outcome.get("fn_miss", 0)
        - abs(w.get("w_cost", -1.0)) * cost
        + w.get("w_viol", -50.0) * outcome.get("policy_violation", 0)
        + w.get("w_trust", 5.0) * trust_score * outcome.get("tp_contain", 0)
        + w.get("w_patient_risk", 0.0) * patient_risk
    )


class ResponseSimulator:
    """Evaluates proposed actions against labeled detections."""

    def __init__(self, reward_weights: Dict[str, float], action_costs: Dict[str, float]):
        self.reward_weights = reward_weights
        self.action_costs = action_costs
        self.history: list = []

    def step(
        self,
        detection: DetectionResult,
        action: ResponseAction,
        validation_status: str,
        trust_score: float,
    ) -> Tuple[Dict[str, Any], float]:
        outcome = simulate_action_outcome(detection, action, validation_status)
        reward = compute_reward(
            outcome,
            action,
            trust_score,
            self.reward_weights,
            self.action_costs,
            device_type=getattr(detection, "device_type", "") or "",
        )
        record = {
            "agent_id": detection.agent_id,
            "round": detection.round_num,
            "action_type": action.action_type,
            "validation_status": validation_status,
            "device_type": getattr(detection, "device_type", ""),
            **outcome,
            "reward": reward,
        }
        self.history.append(record)
        return outcome, reward

    def summarize(self) -> Dict[str, float]:
        if not self.history:
            return {
                "response_precision": 0.0,
                "fp_response_rate": 0.0,
                "cumulative_defense_utility": 0.0,
                "violation_rate": 0.0,
            }
        executed = [h for h in self.history if h.get("executed")]
        correct = sum(1 for h in executed if h.get("tp_contain") or (h.get("fn_miss") == 0 and not h.get("fp_response")))
        precision = correct / len(executed) if executed else 0.0
        fp_rate = sum(1 for h in self.history if h.get("fp_response")) / len(self.history)
        utility = sum(h.get("reward", 0.0) for h in self.history)
        viol = sum(1 for h in self.history if h.get("policy_violation")) / len(self.history)
        return {
            "response_precision": float(precision),
            "fp_response_rate": float(fp_rate),
            "cumulative_defense_utility": float(utility),
            "violation_rate": float(viol),
        }
