"""Reinforcement learning response policies for TrustFed-Agent."""

from .environment import ResponseSimulator, simulate_action_outcome
from .response_strategy import ResponseStrategy
from .observation import ACTION_TYPES, build_action_mask, build_observation

try:
    from .policy_optimizer import PolicyOptimizer
    from .gym_env import HoneypotResponseEnv
except ImportError:
    PolicyOptimizer = None  # type: ignore
    HoneypotResponseEnv = None  # type: ignore

__all__ = [
    "ACTION_TYPES",
    "build_action_mask",
    "build_observation",
    "StaticResponsePolicy",
    "ResponseSimulator",
    "simulate_action_outcome",
    "ResponseStrategy",
    "PolicyOptimizer",
    "HoneypotResponseEnv",
]
