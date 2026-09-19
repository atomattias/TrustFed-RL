"""
RL-adaptive trust calibration (Δτ) under healthcare governance constraints.

Master framing (§III-F): Δτ is a *trust-evolution channel*, not an RL action.
The PPO action space remains {Monitor, Alert, Throttle, Isolate, Block, Escalate}.
Bounded offsets are applied to multi-signal trust fusion outputs.

Participation symmetry (Option 3 / comm_skip):
  Skippers must not evade Δτ. ``apply_to_signals(..., participated=False)`` uses the
  same channel as participants and, when peer offsets exist, cannot sit above the
  peer-mean offset (prevents the ranking inversion where honest clients absorb
  −max_delta while skippers keep Δτ=0).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return float(lo if x < lo else hi if x > hi else x)


class TrustCalibrator:
    """Learns per-client trust offsets bounded by governance policy."""

    def __init__(self, rl_config: Dict[str, Any], governance_config: Dict[str, Any]):
        tc = rl_config.get("trust_calibration", {})
        gov_tc = governance_config.get("trust_calibration", {})
        self.enabled = bool(tc.get("enabled", False))
        self.max_delta = float(min(tc.get("max_delta", 0.1), gov_tc.get("max_adjustment", 0.1)))
        self.learning_rate = float(tc.get("learning_rate", 0.05))
        self.baseline_ema = float(tc.get("baseline_ema", 0.9))
        self.r_fp_weight = float(tc.get("response_fidelity_weight", 0.6))
        self.r_adv_weight = float(tc.get("reward_advantage_weight", 0.4))
        self.trust_floor = float(gov_tc.get("min_trust_floor", 0.0))
        self.trust_ceiling = float(gov_tc.get("max_trust_ceiling", 1.0))
        self._offsets: Dict[str, float] = {}
        self._baseline_reward: float = 0.0
        self._history: List[Dict[str, float]] = []

    def peer_mean_offset(self) -> Optional[float]:
        if not self._offsets:
            return None
        vals = list(self._offsets.values())
        return float(sum(vals) / len(vals))

    def effective_offset(self, client_id: str, *, participated: bool = True) -> float:
        """Offset used for Δτ; skippers cannot evade a learned peer penalty.

        Does not insert a missing skipper as 0.0 into ``_offsets`` (that would
        dilute ``peer_mean_offset`` toward zero across multiple skippers).
        """
        own = float(self._offsets.get(client_id, 0.0))
        if participated:
            return own
        peer = self.peer_mean_offset()
        if peer is None:
            return own
        return float(min(own, peer))

    def compute_delta(
        self,
        client_id: str,
        signals: Dict[str, float],
        *,
        participated: bool = True,
    ) -> float:
        if not self.enabled:
            return 0.0
        # Offset-only Δτ. Do NOT fold response_fidelity_penalty into the delta:
        # that term is available only on the participate path and can re-invert
        # attacker ranking under comm_skip before offsets hit ±max_delta.
        # Response outcomes still shape offsets via update_from_round(rewards).
        _ = signals  # kept for API compatibility / future features
        offset = self.effective_offset(client_id, participated=participated)
        return _clip(offset, -self.max_delta, self.max_delta)

    def apply_to_signals(
        self,
        client_id: str,
        signals: Dict[str, float],
        *,
        participated: bool = True,
    ) -> Dict[str, float]:
        out = dict(signals)
        out["rl_trust_calibration_delta"] = self.compute_delta(
            client_id, signals, participated=participated
        )
        out["rl_trust_calibration_participated"] = bool(participated)
        return out

    def update_from_round(
        self,
        client_rewards: Dict[str, List[float]],
        enable_rl: bool,
    ) -> Dict[str, float]:
        if not self.enabled or not enable_rl:
            return {}
        updates: Dict[str, float] = {}
        all_rewards = [r for rs in client_rewards.values() for r in rs]
        if all_rewards:
            round_mean = float(sum(all_rewards) / len(all_rewards))
            self._baseline_reward = (
                self.baseline_ema * self._baseline_reward
                + (1.0 - self.baseline_ema) * round_mean
            )
        for client_id, rewards in client_rewards.items():
            if not rewards:
                continue
            mean_r = float(sum(rewards) / len(rewards))
            advantage = mean_r - self._baseline_reward
            old = self._offsets.get(client_id, 0.0)
            step = self.learning_rate * self.r_adv_weight * math.tanh(advantage * 0.1)
            new = _clip(old + step, -self.max_delta, self.max_delta)
            self._offsets[client_id] = new
            updates[client_id] = new
        self._history.append(updates)
        return updates

    def summarize(self) -> Dict[str, Any]:
        if not self._offsets:
            return {"enabled": self.enabled, "mean_offset": 0.0, "n_clients": 0}
        vals = list(self._offsets.values())
        mean = float(sum(vals) / len(vals))
        var = float(sum((v - mean) ** 2 for v in vals) / len(vals))
        return {
            "enabled": self.enabled,
            "mean_offset": mean,
            "std_offset": float(math.sqrt(var)),
            "n_clients": len(vals),
        }
