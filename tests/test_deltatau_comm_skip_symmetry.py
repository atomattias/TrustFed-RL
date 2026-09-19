"""Regression tests: Δτ must not invert attacker ranking under comm_skip (Option 3).

Loads ``trust_calibrator.py`` by path to avoid ``rl/__init__.py`` (heavy deps / macOS numpy check).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_calibrator_cls():
    path = ROOT / "src" / "rl" / "trust_calibrator.py"
    spec = importlib.util.spec_from_file_location("trust_calibrator_standalone", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrustCalibrator


TrustCalibrator = _load_calibrator_cls()


def _calibrator(enabled: bool = True) -> TrustCalibrator:
    return TrustCalibrator(
        {"trust_calibration": {"enabled": enabled, "max_delta": 0.1}},
        {"trust_calibration": {"max_adjustment": 0.1}},
    )


def _fuse(weights, signals, delta: float) -> float:
    """Mirror TrustManager.compute_multi_signal_trust (weighted sum + Δτ)."""
    keys = ("V", "S", "D", "U", "C", "R")
    clamped = {k: max(0.0, min(1.0, float(signals.get(k, 0.0)))) for k in keys}
    ti = sum(float(weights[k]) * clamped[k] for k in keys)
    ti = max(0.0, min(1.0, ti))
    if delta != 0.0:
        ti = max(0.0, min(1.0, ti + float(delta)))
    return ti


# Natural-partition equal weights (R=0); C carries skip evidence.
_WEIGHTS = {"V": 0.2, "S": 0.2, "D": 0.2, "U": 0.2, "C": 0.2, "R": 0.0}

_HONEST = {"V": 0.80, "S": 0.80, "D": 0.80, "U": 0.80, "C": 0.95, "R": 1.0}
# Early post-t_a skip: C has only decayed modestly (EMA), while Δτ already sat at −0.1.
# That is the regime where omitting skip-path Δτ inverts ranking.
_SKIPPER_EARLY = {"V": 0.80, "S": 0.80, "D": 0.80, "U": 0.80, "C": 0.75, "R": 1.0}
# Later skip: C collapsed; with symmetric Δτ, ranking is restored by C.
_SKIPPER = {"V": 0.80, "S": 0.80, "D": 0.80, "U": 0.80, "C": 0.20, "R": 1.0}


def test_bug_without_skip_delta_inverts_ranking():
    """Historical bug: Δτ≈−0.1 on honest only → skipper looks more trusted early post-t_a."""
    honest_t = _fuse(_WEIGHTS, _HONEST, delta=-0.1)
    skipper_t = _fuse(_WEIGHTS, _SKIPPER_EARLY, delta=0.0)  # skip path omitted Δτ
    assert skipper_t > honest_t, (
        f"expected inversion without skip Δτ, got skipper={skipper_t:.4f} "
        f"honest={honest_t:.4f}"
    )


def test_option3_skip_delta_restores_ranking():
    """Same Δτ on both paths: C evidence ranks skipper below honest."""
    cal = _calibrator(True)
    # Pre-t_a saturation observed in real runs: all clients at −max_delta.
    for cid in ("honest", "skipper"):
        cal._offsets[cid] = -0.1

    d_h = cal.compute_delta("honest", _HONEST, participated=True)
    d_s = cal.compute_delta("skipper", _SKIPPER, participated=False)
    assert d_h == -0.1 and d_s == -0.1

    honest_t = _fuse(_WEIGHTS, _HONEST, delta=d_h)
    skipper_t = _fuse(_WEIGHTS, _SKIPPER, delta=d_s)
    assert skipper_t < honest_t, (
        f"Option 3 failed: skipper={skipper_t:.4f} honest={honest_t:.4f}"
    )


def test_skipper_cannot_evade_with_zero_own_offset():
    """If a skipper's stored offset is still 0 while peers sat at −0.1, still apply peer mean."""
    cal = _calibrator(True)
    for i in range(9):
        cal._offsets[f"honest_{i}"] = -0.1
    # Skipper missing / zero offset (evasion case).
    assert "skipper" not in cal._offsets or cal._offsets.get("skipper", 0.0) == 0.0

    d_s = cal.compute_delta("skipper", _SKIPPER, participated=False)
    d_h = cal.compute_delta("honest_0", _HONEST, participated=True)
    assert d_s == -0.1
    assert d_h == -0.1
    # Must not pollute peer mean by inserting skipper=0 into _offsets.
    assert "skipper" not in cal._offsets

    honest_t = _fuse(_WEIGHTS, _HONEST, delta=d_h)
    skipper_t = _fuse(_WEIGHTS, _SKIPPER, delta=d_s)
    assert skipper_t < honest_t


def test_multiple_skippers_do_not_dilute_peer_mean():
    """Three missing-offset skippers must each still get peer mean −0.1."""
    cal = _calibrator(True)
    for i in range(9):
        cal._offsets[f"honest_{i}"] = -0.1
    deltas = [
        cal.compute_delta(f"skip_{j}", _SKIPPER_EARLY, participated=False)
        for j in range(3)
    ]
    assert deltas == [-0.1, -0.1, -0.1]
    assert cal.peer_mean_offset() == -0.1
    assert all(f"skip_{j}" not in cal._offsets for j in range(3))


def test_response_fidelity_penalty_cannot_break_skip_symmetry():
    """Participate-only R-penalty must not make honest Δτ stricter than skippers."""
    cal = _calibrator(True)
    for cid in ("honest", "skipper"):
        cal._offsets[cid] = -0.05
    honest_signals = dict(_HONEST)
    honest_signals["response_fidelity_penalty"] = 1.0
    d_h = cal.compute_delta("honest", honest_signals, participated=True)
    d_s = cal.compute_delta("skipper", _SKIPPER_EARLY, participated=False)
    assert d_h == d_s == -0.05
    assert _fuse(_WEIGHTS, _SKIPPER_EARLY, d_s) < _fuse(_WEIGHTS, _HONEST, d_h)


def test_option3_also_fixes_early_skip_regime():
    """Symmetric Δτ keeps early-skip ranking correct even before C fully collapses."""
    cal = _calibrator(True)
    for cid in ("honest", "skipper"):
        cal._offsets[cid] = -0.1
    d_h = cal.compute_delta("honest", _HONEST, participated=True)
    d_s = cal.compute_delta("skipper", _SKIPPER_EARLY, participated=False)
    honest_t = _fuse(_WEIGHTS, _HONEST, delta=d_h)
    skipper_t = _fuse(_WEIGHTS, _SKIPPER_EARLY, delta=d_s)
    assert skipper_t < honest_t


def test_disabled_calibrator_is_noop():
    cal = _calibrator(False)
    cal._offsets["c1"] = -0.1
    out = cal.apply_to_signals("c1", _HONEST, participated=True)
    assert out["rl_trust_calibration_delta"] == 0.0


def test_apply_flags_participation():
    cal = _calibrator(True)
    cal._offsets["c1"] = -0.1
    skip = cal.apply_to_signals("c1", _SKIPPER, participated=False)
    part = cal.apply_to_signals("c1", _HONEST, participated=True)
    assert skip["rl_trust_calibration_participated"] is False
    assert part["rl_trust_calibration_participated"] is True
    assert skip["rl_trust_calibration_delta"] == part["rl_trust_calibration_delta"] == -0.1


def test_runner_skip_block_calls_apply_with_participated_false():
    """Static guard: agent_experiment_runner skip path must pass participated=False."""
    text = (ROOT / "src" / "agent_experiment_runner.py").read_text()
    start = text.find("if skip:")
    assert start >= 0
    block = text[start : text.find("continue", start) + len("continue")]
    assert "apply_to_signals" in block
    assert "participated=False" in block
