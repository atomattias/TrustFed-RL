"""RL policy training for TrustFed-Agent response selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.autonomous_agent import ACTION_SEVERITY, DetectionResult, ResponseAction
from rl.observation import ACTION_TYPES, build_action_mask, build_observation


class PolicyOptimizer:
    """Train/evaluate response policies (MaskablePPO or unmasked PPO).

    Controlled by ``rl_config["use_action_masking"]`` (default True).
    """

    def __init__(self, rl_config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        self.rl_config = rl_config
        training = rl_config.get("training", {})
        self.use_action_masking = bool(rl_config.get("use_action_masking", True))
        self.checkpoint_dir = Path(
            checkpoint_dir or training.get("checkpoint_dir", "results/trustfed_agent/rl_checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._train_status: Dict[str, Any] = {
            "status": "untrained",
            "use_action_masking": self.use_action_masking,
        }

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def train_status(self) -> Dict[str, Any]:
        return dict(self._train_status)

    def _checkpoint_path(self) -> Path:
        name = (
            "ppo_response_maskable.zip"
            if self.use_action_masking
            else "ppo_response_unmasked.zip"
        )
        return self.checkpoint_dir / name

    def _import_maskable(self):
        try:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
            return MaskablePPO, ActionMasker
        except ImportError as exc:
            raise ImportError(
                "Maskable PPO requires sb3-contrib and stable-baselines3. "
                "Install: pip install stable-baselines3 sb3-contrib gymnasium torch"
            ) from exc

    def _import_ppo(self):
        try:
            from stable_baselines3 import PPO
            return PPO
        except ImportError as exc:
            raise ImportError(
                "PPO requires stable-baselines3. "
                "Install: pip install stable-baselines3 gymnasium torch"
            ) from exc

    def train(
        self,
        incidents: List[Tuple[DetectionResult, float]],
        allowed_actions_fn=None,
        total_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not incidents:
            return {"status": "skipped", "reason": "no incidents"}

        from rl.gym_env import HoneypotResponseEnv

        # Unmasked bake-off: train on the full discrete action space so the
        # policy can propose governance-illegal actions (measured at inference).
        env_allowed_fn = allowed_actions_fn if self.use_action_masking else None
        env = HoneypotResponseEnv(
            incidents=incidents,
            reward_weights=self.rl_config.get("reward_weights", {}),
            action_costs=self.rl_config.get("action_costs", {}),
            allowed_actions_fn=env_allowed_fn,
        )

        hp = self.rl_config.get("hyperparameters", {})
        training = self.rl_config.get("training", {})
        if total_timesteps is None:
            per_round = int(training.get("total_timesteps_per_round", 2048))
            total_timesteps = max(per_round, len(incidents) * 32)

        common_kwargs = dict(
            learning_rate=hp.get("learning_rate", 3e-4),
            gamma=hp.get("gamma", 0.99),
            verbose=0,
        )

        if self.use_action_masking:
            MaskablePPO, ActionMasker = self._import_maskable()

            def mask_fn(e):
                return e.unwrapped.action_masks()

            train_env = ActionMasker(env, mask_fn)
            if self._model is None:
                self._model = MaskablePPO("MlpPolicy", train_env, **common_kwargs)
            else:
                self._model.set_env(train_env)
        else:
            PPO = self._import_ppo()
            if self._model is None:
                self._model = PPO("MlpPolicy", env, **common_kwargs)
            else:
                self._model.set_env(env)

        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        path = self._checkpoint_path()
        self._model.save(str(path))
        self._train_status = {
            "status": "trained",
            "checkpoint": str(path),
            "timesteps": total_timesteps,
            "incidents": len(incidents),
            "use_action_masking": self.use_action_masking,
        }
        return self._train_status

    def load(self, path: Optional[str] = None) -> bool:
        checkpoint = Path(path) if path else self._checkpoint_path()
        # Backward compatible with older single-name checkpoints
        if not checkpoint.exists() and path is None:
            legacy = self.checkpoint_dir / "ppo_response.zip"
            if legacy.exists() and self.use_action_masking:
                checkpoint = legacy
        if not checkpoint.exists():
            return False
        if self.use_action_masking:
            MaskablePPO, _ = self._import_maskable()
            self._model = MaskablePPO.load(str(checkpoint))
        else:
            PPO = self._import_ppo()
            self._model = PPO.load(str(checkpoint))
        self._train_status = {
            "status": "loaded",
            "checkpoint": str(checkpoint),
            "use_action_masking": self.use_action_masking,
        }
        return True

    def predict_action(
        self,
        detection: DetectionResult,
        trust_score: float,
        allowed_actions: Optional[List[str]] = None,
        max_rounds: int = 10,
    ) -> str:
        if self._model is None:
            raise RuntimeError("PPO model not trained")

        obs = build_observation(detection, trust_score, max_rounds=max_rounds)
        if self.use_action_masking:
            mask = build_action_mask(allowed_actions)
            action_idx, _ = self._model.predict(
                obs,
                action_masks=mask,
                deterministic=True,
            )
        else:
            # Full action space — illegal proposals are counted by ResponseStrategy.
            action_idx, _ = self._model.predict(obs, deterministic=True)
        return ACTION_TYPES[int(action_idx)]

    def to_response_action(
        self,
        detection: DetectionResult,
        trust_score: float,
        allowed_actions: Optional[List[str]] = None,
        max_rounds: int = 10,
    ) -> ResponseAction:
        action_type = self.predict_action(
            detection, trust_score, allowed_actions, max_rounds=max_rounds
        )
        return ResponseAction(
            action_type=action_type,
            severity=ACTION_SEVERITY[action_type],
            confidence=detection.confidence,
            agent_id=detection.agent_id,
            round_num=detection.round_num,
            target=getattr(detection, "device_type", "") or "",
        )
