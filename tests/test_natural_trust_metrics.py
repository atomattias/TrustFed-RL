"""Unit tests for natural trust discrimination metrics (WP5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import compute_natural_trust_discrimination, delta_f1_robust


def _logs():
    # Warm-up: attackers look trusted; post t_a=3 attackers drop
    def snap(round_num, trusts, alphas=None):
        attackers = {"client_03", "client_07"}
        ct = {}
        for cid, t in trusts.items():
            ct[cid] = {
                "T": t,
                "V": t,
                "alpha": (alphas or {}).get(cid, t / sum(trusts.values())),
                "is_attacker": cid in attackers,
            }
        return {"round": round_num, "client_trust": ct, "compromise_active": round_num >= 3}

    return [
        snap(1, {"client_01": 0.8, "client_02": 0.8, "client_03": 0.8, "client_07": 0.8}),
        snap(2, {"client_01": 0.8, "client_02": 0.8, "client_03": 0.75, "client_07": 0.75}),
        snap(3, {"client_01": 0.85, "client_02": 0.82, "client_03": 0.2, "client_07": 0.25}),
        snap(4, {"client_01": 0.9, "client_02": 0.88, "client_03": 0.15, "client_07": 0.18}),
    ]


def test_auroc_and_delay():
    out = compute_natural_trust_discrimination(
        _logs(),
        compromise_round=3,
        attacker_ids=["client_03", "client_07"],
        top_k=2,
    )
    assert out["available"]
    assert out["auroc"] == 1.0
    assert out["auprc"] == 1.0
    assert out["detection_delay"] == 0  # detected at t_a
    assert out["false_distrust_count"] == 0
    assert out["mean_alpha_malicious"] is not None
    assert out["mean_alpha_malicious"] < out["mean_alpha_benign"]


def test_delta_f1():
    assert abs(delta_f1_robust(0.9, 0.8) - 0.1) < 1e-9
    assert delta_f1_robust(None, 0.8) is None
