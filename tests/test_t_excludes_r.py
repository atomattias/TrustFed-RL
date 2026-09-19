"""WP3: behavioural trust T excludes contextual R (Trusted-plan §5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config_loader import (
    load_trust_config,
    resolve_signal_weights,
    resolve_trust_config_path,
)
from federated_server import TrustManager


def test_natural_trust_config_excludes_R():
    path = resolve_trust_config_path("iomt_natural", None, ROOT)
    assert path.endswith("trust_config_iomt_natural.json")
    cfg = load_trust_config(path)
    w = resolve_signal_weights(cfg, include_R_in_T=None)
    assert abs(w["R"]) < 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for k in ("V", "S", "D", "U", "C"):
        assert abs(w[k] - 0.2) < 1e-9


def test_natural_b1_config_is_v_only():
    path = resolve_trust_config_path("iomt_natural", trust_profile="b1", root=ROOT)
    assert path.endswith("trust_config_iomt_natural_b1.json")
    cfg = load_trust_config(path)
    w = resolve_signal_weights(cfg, include_R_in_T=False)
    assert abs(w["V"] - 1.0) < 1e-9
    for k in ("S", "D", "U", "C", "R"):
        assert abs(w[k]) < 1e-12


def test_natural_b2_profile_is_equal_behavioural():
    path = resolve_trust_config_path("iomt_natural", trust_profile="b2", root=ROOT)
    assert path.endswith("trust_config_iomt_natural.json")
    cfg = load_trust_config(path)
    w = resolve_signal_weights(cfg, include_R_in_T=False)
    for k in ("V", "S", "D", "U", "C"):
        assert abs(w[k] - 0.2) < 1e-9
    assert abs(w["R"]) < 1e-12


def test_resolve_metrics_run_id_b1v_and_b2():
    from evaluation import resolve_metrics_run_id

    assert (
        resolve_metrics_run_id(
            "trustfed_b1v",
            dataset="iomt_natural",
            uniform_contextual_prior=True,
            late_compromise_tag="poison25",
        )
        == "trustfed_b1v_uniform_iomt_natural_poison25"
    )
    assert (
        resolve_metrics_run_id(
            "trustfed_b2",
            dataset="iomt_natural",
            uniform_contextual_prior=True,
            late_compromise_tag="benign",
        )
        == "trustfed_b2_uniform_iomt_natural_benign"
    )
    assert (
        resolve_metrics_run_id(
            "trustfed_agent",
            dataset="iomt_natural",
            uniform_contextual_prior=True,
            include_R_in_T=True,
            late_compromise_tag="poison25",
        )
        == "trustfed_agent_b5r_uniform_iomt_natural_poison25"
    )


def test_include_R_in_T_ablation():
    cfg = load_trust_config(str(ROOT / "config" / "trust_config_iomt_natural.json"))
    w = resolve_signal_weights(cfg, include_R_in_T=True)
    assert w["R"] > 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_fusion_ignores_R_when_weight_zero():
    tm = TrustManager(
        use_multi_signal=True,
        signal_weights={"V": 0.2, "S": 0.2, "D": 0.2, "U": 0.2, "C": 0.2, "R": 0.0},
    )
    base = {"V": 0.8, "S": 0.8, "D": 0.8, "U": 0.8, "C": 0.8, "R": 0.0}
    t_low_r = tm.compute_multi_signal_trust({**base, "R": 0.0})
    t_high_r = tm.compute_multi_signal_trust({**base, "R": 1.0})
    assert abs(t_low_r - t_high_r) < 1e-12


def test_legacy_iomt_still_includes_R_by_default():
    cfg = load_trust_config(str(ROOT / "config" / "trust_config.json"))
    w = resolve_signal_weights(cfg)
    assert w["R"] > 0.0
