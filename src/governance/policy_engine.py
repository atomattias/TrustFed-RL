"""Policy constraints on aggregation and response actions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agents.autonomous_agent import normalize_action_type


class PolicyEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        agg = config.get("aggregation", {})
        self.min_trust_threshold = float(agg.get("min_trust_threshold", 0.1))
        self.max_client_weight = float(agg.get("max_client_weight", 0.4))
        self.min_participating_clients = int(agg.get("min_participating_clients", 3))
        # Default β=1 → α_i ∝ T_i (paper Alg. 1). Sub-linear β<1 compresses gaps.
        self.beta = float(agg.get("trust_exponent_beta", 1.0))
        self.share_floor = float(agg.get("share_floor", 0.0))
        self.tier_policies = config.get("trust_tier_policies", {})

    def get_trust_tier(self, trust_score: float) -> str:
        if trust_score >= self.tier_policies.get("high", {}).get("min_trust", 0.60):
            return "high"
        if trust_score >= self.tier_policies.get("medium", {}).get("min_trust", 0.30):
            return "medium"
        return "low"

    def get_allowed_actions(self, trust_score: float) -> List[str]:
        tier = self.get_trust_tier(trust_score)
        tier_cfg = self.tier_policies.get(tier, {})
        actions = tier_cfg.get("allowed_actions", ["monitor", "alert"])
        # Deduplicate while preserving order after legacy→master mapping
        seen = set()
        out: List[str] = []
        for a in actions:
            na = normalize_action_type(a)
            if na not in seen:
                seen.add(na)
                out.append(na)
        return out

    def get_max_severity(self, trust_score: float) -> float:
        tier = self.get_trust_tier(trust_score)
        return float(self.tier_policies.get(tier, {}).get("max_severity", 0.2))

    def constrain_aggregation_weights(
        self,
        trust_scores: np.ndarray,
        client_ids: Optional[List[str]] = None,
    ) -> np.ndarray:
        trust = np.asarray(trust_scores, dtype=float)
        n = len(trust)
        if n == 0:
            return trust

        eligible = trust >= self.min_trust_threshold
        if self.share_floor > 0:
            eligible = eligible & (trust >= self.share_floor)
        if eligible.sum() < self.min_participating_clients:
            eligible = np.ones(n, dtype=bool)

        raw = np.zeros(n)
        eligible_trust = np.maximum(trust[eligible], 0.0) ** self.beta
        denom = eligible_trust.sum()
        if denom <= 0:
            raw[eligible] = 1.0 / eligible.sum()
        else:
            raw[eligible] = eligible_trust / denom

        for _ in range(n):
            over = raw > self.max_client_weight
            if not over.any():
                break
            excess = (raw[over] - self.max_client_weight).sum()
            raw[over] = self.max_client_weight
            under = ~over & eligible
            if under.sum() == 0:
                break
            raw[under] += excess / under.sum()

        total = raw.sum()
        if total > 0:
            raw /= total
        return raw
