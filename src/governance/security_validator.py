"""Pre-execution validation of proposed response actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.autonomous_agent import ResponseAction
    from governance.policy_repository import PolicyRepository

from agents.autonomous_agent import ACTION_SEVERITY, normalize_action_type
from governance.context_builder import normalize_db_action
from governance.policy_engine import PolicyEngine

# Aggressive / clinically impactful actions subject to per-agent rate limits
AGGRESSIVE_ACTIONS = {"throttle", "block", "isolate"}

_OUTCOME_TO_STATUS = {
    "APPROVE": "approved",
    "MODIFY": "modified",
    "ESCALATE": "escalated",
    "POSTPONE": "postponed",
    "REJECT": "rejected",
}


@dataclass
class ValidationResult:
    status: str
    action: Any
    reason: str
    human_review: bool = False
    decision_id: Optional[int] = None
    matched_rule_ids: Optional[list] = None


def db_action_to_effector(final_action: str) -> Tuple[str, float]:
    """Map DB final_action → (rl action_type, severity). NO_ACTION → monitor @ 0."""
    up = normalize_db_action(final_action)
    if up == "NO_ACTION":
        return "monitor", 0.0
    low = up.lower()
    return low, float(ACTION_SEVERITY.get(low, 0.1))


class SecurityValidator:
    def __init__(
        self,
        config: Dict[str, Any],
        policy_engine: PolicyEngine,
        policy_repository: Optional["PolicyRepository"] = None,
    ):
        self.config = config
        self.policy_engine = policy_engine
        self.policy_repository = policy_repository
        resp = config.get("response", {})
        self.allowed_actions = {normalize_action_type(a) for a in resp.get("allowed_actions", [])}
        self.min_confidence_for_block = float(resp.get("min_confidence_for_block", 0.75))
        self.min_confidence_for_isolate = float(resp.get("min_confidence_for_isolate", 0.85))
        self.rate_limit_per_round = int(resp.get("rate_limit_per_round", 10))
        self.device_criticality = config.get("device_criticality", {})
        self._device_lookup = self._build_device_lookup()
        self._round_action_counts: Dict[Tuple[int, str], int] = {}

    def _build_device_lookup(self) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for tier_name, tier_cfg in self.device_criticality.items():
            for device in tier_cfg.get("devices", []):
                lookup[str(device).lower()] = {
                    "tier": tier_name,
                    "blocked_actions": {
                        normalize_action_type(a) for a in tier_cfg.get("blocked_actions", [])
                    },
                    "max_auto_severity": float(tier_cfg.get("max_auto_severity", 1.0)),
                    "require_escalation": bool(tier_cfg.get("require_escalation", False)),
                }
        return lookup

    def _device_policy(self, device_type: Optional[str]) -> Optional[Dict[str, Any]]:
        if not device_type:
            return None
        return self._device_lookup.get(str(device_type).lower())

    def validate(
        self,
        action: "ResponseAction",
        trust_score: float,
        round_num: int = 0,
        agent_id: Optional[str] = None,
        device_type: Optional[str] = None,
        attack_severity: Optional[float] = None,
        contextual_risk: Optional[float] = None,
        device_criticality: Optional[float] = None,
        attack_class: Optional[str] = None,
        experiment_run_id: Optional[str] = None,
    ) -> ValidationResult:
        action.action_type = normalize_action_type(action.action_type)
        agent_key = agent_id or getattr(action, "agent_id", "unknown")
        count_key = (round_num, agent_key)
        count = self._round_action_counts.get(count_key, 0)

        if self.allowed_actions and action.action_type not in self.allowed_actions:
            return ValidationResult("rejected", action, "action_not_allowed")

        repo = self.policy_repository
        if repo is not None and getattr(repo, "runtime_api_ready", False):
            ctx = {
                "rl_action": action.action_type,
                "trust": float(trust_score),
                "trust_score": float(trust_score),
                "attack_severity": float(
                    attack_severity if attack_severity is not None else action.confidence
                ),
                "attack_confidence": float(action.confidence),
                "contextual_risk": float(contextual_risk or 0.0),
                "risk_score": float(contextual_risk or 0.0),
                "device_criticality": float(device_criticality or 0.5),
                "device_type": device_type or getattr(action, "target", None),
                "participant_id": agent_key,
                "agent_id": agent_key,
                "attack_class": attack_class,
            }
            decision = repo.evaluate(ctx)

            # Soft rate-limit on aggressive auto-executions (never write escalate into final_action)
            if (
                decision.outcome == "APPROVE"
                and action.action_type in AGGRESSIVE_ACTIONS
                and count >= self.rate_limit_per_round
            ):
                decision.outcome = "ESCALATE"
                decision.final_action = "MONITOR"
                decision.human_review = True
                decision.review_status = "PENDING"
                decision.explanation = "rate_limit_downgrade"

            effector, sev = db_action_to_effector(decision.final_action)
            action.action_type = effector
            action.severity = sev

            repo.record_decision(
                decision,
                ctx,
                round_num=round_num,
                agent_id=agent_key,
                experiment_run_id=experiment_run_id,
            )

            if decision.outcome == "APPROVE" and action.action_type in AGGRESSIVE_ACTIONS:
                self._round_action_counts[count_key] = count + 1

            outcome = decision.outcome
            if outcome == "REJECT":
                if normalize_db_action(decision.final_action) == "NO_ACTION":
                    status = "rejected"
                else:
                    status = "modified"
                reason = f"governance_REJECT→{action.action_type}: {decision.explanation}"
            else:
                status = _OUTCOME_TO_STATUS.get(outcome, "modified")
                reason = decision.explanation

            return ValidationResult(
                status=status,
                action=action,
                reason=reason,
                human_review=decision.human_review,
                decision_id=decision.decision_id,
                matched_rule_ids=list(decision.matched_rule_ids),
            )

        # Legacy procedural path (no repository / WP1-only)
        device_pol = self._device_policy(device_type or getattr(action, "target", None))
        if device_pol:
            if action.action_type in device_pol["blocked_actions"]:
                action.action_type = "escalate"
                action.severity = min(action.severity, 0.3)
                return ValidationResult(
                    "escalated", action, f"device_criticality_{device_pol['tier']}", human_review=True
                )
            if device_pol["require_escalation"] and action.action_type in AGGRESSIVE_ACTIONS:
                if action.severity > device_pol["max_auto_severity"]:
                    action.action_type = "escalate"
                    action.severity = 0.3
                    return ValidationResult(
                        "escalated", action, "device_requires_escalation", human_review=True
                    )
            if action.severity > device_pol["max_auto_severity"]:
                action.severity = device_pol["max_auto_severity"]
                if action.action_type in ("block", "isolate"):
                    action.action_type = "throttle"
                return ValidationResult("modified", action, "device_severity_capped")

        tier_allowed = [
            normalize_action_type(a) for a in self.policy_engine.get_allowed_actions(trust_score)
        ]
        if action.action_type not in tier_allowed:
            downgraded = "alert" if "alert" in tier_allowed else "monitor"
            action.action_type = downgraded
            action.severity = min(action.severity, self.policy_engine.get_max_severity(trust_score))
            return ValidationResult("modified", action, "trust_tier_downgrade")

        if action.action_type in AGGRESSIVE_ACTIONS and action.confidence < self.min_confidence_for_block:
            action.action_type = "alert"
            action.severity = 0.2
            return ValidationResult("modified", action, "insufficient_confidence_for_block")

        if action.action_type == "isolate" and action.confidence < self.min_confidence_for_isolate:
            action.action_type = "escalate"
            action.severity = 0.3
            return ValidationResult("escalated", action, "isolate_requires_escalation", human_review=True)

        max_sev = self.policy_engine.get_max_severity(trust_score)
        if action.severity > max_sev:
            action.severity = max_sev
            return ValidationResult("modified", action, "severity_capped")

        if action.action_type in AGGRESSIVE_ACTIONS and count >= self.rate_limit_per_round:
            action.action_type = "escalate" if "escalate" in tier_allowed else "alert"
            action.severity = min(action.severity, 0.3)
            return ValidationResult("escalated", action, "rate_limit_downgrade", human_review=True)

        if action.action_type in AGGRESSIVE_ACTIONS:
            self._round_action_counts[count_key] = count + 1
        return ValidationResult("approved", action, "ok")


def default_policy_db_path(root: Path, dataset: str = "iomt") -> Path:
    return root / "results" / "trustfed_agent" / "governance" / f"policy_repo_{dataset}.sqlite"
