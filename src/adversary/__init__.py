"""Adversary models for TrustFed-RL experiments (master threat model)."""

from .co_adaptive import ATTACK_MODES, CoAdaptiveAdversary
from .late_compromise import (
    DEFAULT_ATTACKER_IDS,
    POISON_MODES,
    LateCompromiseConfig,
    LateCompromiseController,
    fraction_tag,
)

__all__ = [
    "CoAdaptiveAdversary",
    "ATTACK_MODES",
    "LateCompromiseConfig",
    "LateCompromiseController",
    "DEFAULT_ATTACKER_IDS",
    "POISON_MODES",
    "fraction_tag",
]
