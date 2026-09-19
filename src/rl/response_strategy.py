"""
Response strategy: PPO policy when trained, static rules for cold-start (B2 / round 1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.autonomous_agent import ACTION_SEVERITY, DetectionResult, ResponseAction
from rl.policy_optimizer import PolicyOptimizer
from rl.static_policy import StaticResponsePolicy


class ResponseStrategy:
    def __init__(
        self,
        rl_config: Dict[str, Any],
        use_rl: bool = False,
        checkpoint_dir: Optional[str] = None,
        max_rounds: int = 10,
    ):
        self.use_rl = use_rl
        self.rl_config = rl_config
        self.max_rounds = max_rounds
        self.static = StaticResponsePolicy(rl_config)
        self._optimizer = (
            PolicyOptimizer(rl_config, checkpoint_dir=checkpoint_dir) if use_rl else None
        )
        self._last_train_status: Dict[str, Any] = {"status": "untrained"}
        self._proposal_total = 0
        self._invalid_proposals = 0

    @property
    def ppo_active(self) -> bool:
        return self.use_rl and self._optimizer is not None and self._optimizer.is_trained

    @property
    def train_status(self) -> Dict[str, Any]:
        return dict(self._last_train_status)

    @property
    def use_action_masking(self) -> bool:
        if self._optimizer is None:
            return bool(self.rl_config.get("use_action_masking", True))
        return bool(self._optimizer.use_action_masking)

    def reset_proposal_stats(self) -> None:
        self._proposal_total = 0
        self._invalid_proposals = 0

    def proposal_stats(self) -> Dict[str, Any]:
        total = int(self._proposal_total)
        invalid = int(self._invalid_proposals)
        return {
            "proposal_total": total,
            "invalid_proposals": invalid,
            "invalid_proposal_rate": (float(invalid) / total) if total else 0.0,
            "use_action_masking": self.use_action_masking,
        }

    def train_on_incidents(
        self,
        incidents: List[Tuple[DetectionResult, float]],
        allowed_actions_fn=None,
    ) -> Dict[str, Any]:
        if not self.use_rl or self._optimizer is None:
            self._last_train_status = {"status": "skipped", "reason": "rl disabled"}
            return self._last_train_status
        self._last_train_status = self._optimizer.train(incidents, allowed_actions_fn)
        return self._last_train_status

    def select_action(
        self,
        detection: DetectionResult,
        trust_score: float,
        allowed_actions: Optional[List[str]] = None,
    ) -> ResponseAction:
        if self.ppo_active:
            action = self._optimizer.to_response_action(
                detection,
                trust_score,
                allowed_actions,
                max_rounds=self.max_rounds,
            )
        else:
            action = self.static.select_action(detection, trust_score)

        self._proposal_total += 1
        if allowed_actions and action.action_type not in allowed_actions:
            self._invalid_proposals += 1
            fallback = "alert" if "alert" in allowed_actions else "monitor"
            action.action_type = fallback
            # Keep severity consistent with the clamped action (else isolate severity leaks into reward).
            action.severity = float(ACTION_SEVERITY.get(fallback, action.severity))
        return action
