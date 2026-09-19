"""Gymnasium environment for IoMT response policy training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.autonomous_agent import ACTION_SEVERITY, DetectionResult, ResponseAction, normalize_action_type
from rl.environment import compute_reward, simulate_action_outcome
from rl.observation import ACTION_TYPES, OBS_DIM, build_action_mask, build_observation


class HoneypotResponseEnv(gym.Env):
    """One step = one detection incident from a preloaded batch."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        incidents: List[Tuple[DetectionResult, float]],
        reward_weights: Dict[str, float],
        action_costs: Dict[str, float],
        allowed_actions_fn=None,
        max_rounds: int = 10,
        validate_fn=None,
    ):
        super().__init__()
        self.incidents = incidents
        self.reward_weights = reward_weights
        self.action_costs = action_costs
        self.allowed_actions_fn = allowed_actions_fn
        self.validate_fn = validate_fn
        self.max_rounds = max_rounds
        self._idx = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTION_TYPES))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        if not self.incidents:
            return np.zeros(OBS_DIM, dtype=np.float32), {}
        det, trust = self.incidents[0]
        return build_observation(det, trust, max_rounds=self.max_rounds), {}

    def step(self, action_idx: int):
        if self._idx >= len(self.incidents):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}

        det, trust = self.incidents[self._idx]
        action_type = ACTION_TYPES[int(action_idx)]
        action = ResponseAction(
            action_type=action_type,
            severity=ACTION_SEVERITY[action_type],
            confidence=det.confidence,
            agent_id=det.agent_id,
            round_num=det.round_num,
            target=getattr(det, "device_type", "") or "",
        )
        status = "approved"
        if self.validate_fn is not None:
            result = self.validate_fn(action, trust, det)
            status = getattr(result, "status", "approved")
            action = getattr(result, "action", action)
        outcome = simulate_action_outcome(det, action, status)
        reward = compute_reward(
            outcome,
            action,
            trust,
            self.reward_weights,
            self.action_costs,
            device_type=getattr(det, "device_type", "") or "",
        )

        self._idx += 1
        terminated = self._idx >= len(self.incidents)
        if terminated:
            next_obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            nd, nt = self.incidents[self._idx]
            next_obs = build_observation(nd, nt, max_rounds=self.max_rounds)
        return next_obs, float(reward), terminated, False, outcome

    def action_masks(self) -> np.ndarray:
        if self._idx < len(self.incidents) and self.allowed_actions_fn:
            _, trust = self.incidents[self._idx]
            return build_action_mask(self.allowed_actions_fn(trust))
        return build_action_mask(None)
