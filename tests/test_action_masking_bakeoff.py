"""Unit tests for use_action_masking wiring (avoids heavy package imports / macOS numpy check)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path, inject: dict | None = None):
    if inject:
        for k, v in inject.items():
            sys.modules[k] = v
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_agents():
    agents = types.ModuleType("agents")
    auto = types.ModuleType("agents.autonomous_agent")

    class DetectionResult:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class ResponseAction:
        def __init__(self, action_type, severity, confidence, agent_id, round_num, target=""):
            self.action_type = action_type
            self.severity = severity
            self.confidence = confidence
            self.agent_id = agent_id
            self.round_num = round_num
            self.target = target

    ACTION_SEVERITY = {
        "monitor": 0.1,
        "alert": 0.2,
        "throttle": 0.4,
        "isolate": 0.9,
        "block": 0.7,
        "escalate": 0.3,
    }
    auto.DetectionResult = DetectionResult
    auto.ResponseAction = ResponseAction
    auto.ACTION_SEVERITY = ACTION_SEVERITY
    auto.normalize_action_type = lambda x: x
    agents.autonomous_agent = auto
    sys.modules["agents"] = agents
    sys.modules["agents.autonomous_agent"] = auto
    return auto


def test_policy_optimizer_reads_masking_flag():
    auto = _stub_agents()
    # Minimal rl.observation stub used by PolicyOptimizer import side
    obs = types.ModuleType("rl.observation")
    obs.ACTION_TYPES = ["monitor", "alert", "throttle", "isolate", "block", "escalate"]
    obs.ACTION_TO_IDX = {a: i for i, a in enumerate(obs.ACTION_TYPES)}
    obs.build_action_mask = lambda allowed=None: None
    obs.build_observation = lambda *a, **k: None
    rl_pkg = types.ModuleType("rl")
    sys.modules["rl"] = rl_pkg
    sys.modules["rl.observation"] = obs

    po = _load(
        "policy_optimizer_standalone",
        ROOT / "src" / "rl" / "policy_optimizer.py",
    )
    on = po.PolicyOptimizer({"use_action_masking": True})
    off = po.PolicyOptimizer({"use_action_masking": False})
    assert on.use_action_masking is True
    assert off.use_action_masking is False
    assert "maskable" in on._checkpoint_path().name
    assert "unmasked" in off._checkpoint_path().name
    assert auto is not None


def test_response_strategy_counts_invalid_before_clamp():
    auto = _stub_agents()
    obs = types.ModuleType("rl.observation")
    obs.ACTION_TYPES = list(auto.ACTION_SEVERITY.keys())
    obs.ACTION_TO_IDX = {a: i for i, a in enumerate(obs.ACTION_TYPES)}
    obs.build_action_mask = lambda allowed=None: None
    obs.build_observation = lambda *a, **k: None
    sys.modules["rl"] = types.ModuleType("rl")
    sys.modules["rl.observation"] = obs
    po = _load("policy_optimizer_standalone2", ROOT / "src" / "rl" / "policy_optimizer.py")
    sys.modules["rl.policy_optimizer"] = po

    static = types.ModuleType("rl.static_policy")

    class StaticResponsePolicy:
        def __init__(self, *_a, **_k):
            pass

        def select_action(self, detection, trust_score):
            return auto.ResponseAction("monitor", 0.1, 0.5, "c1", 1)

    static.StaticResponsePolicy = StaticResponsePolicy
    sys.modules["rl.static_policy"] = static

    rs = _load("response_strategy_standalone", ROOT / "src" / "rl" / "response_strategy.py")
    strat = rs.ResponseStrategy({"use_action_masking": False}, use_rl=True)
    fake = MagicMock()
    fake.is_trained = True
    fake.use_action_masking = False
    fake.to_response_action.return_value = auto.ResponseAction(
        "isolate", auto.ACTION_SEVERITY["isolate"], 0.9, "c1", 1, "sensor"
    )
    strat._optimizer = fake
    det = auto.DetectionResult(
        predicted_label=1, confidence=0.95, attack_class="X", agent_id="c1", round_num=1
    )
    out = strat.select_action(det, 0.2, allowed_actions=["monitor", "alert", "escalate"])
    assert out.action_type in ("monitor", "alert", "escalate")
    assert out.action_type != "isolate"
    stats = strat.proposal_stats()
    assert stats["invalid_proposals"] == 1
    assert stats["invalid_proposal_rate"] == 1.0
