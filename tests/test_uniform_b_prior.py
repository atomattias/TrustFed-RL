"""Unit tests for uniform contextual-risk prior (B1-U / P1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from federated_client import FederatedClient
from evaluation import resolve_metrics_run_id


def _client(client_id: str, *, uniform: bool) -> FederatedClient:
    return FederatedClient(
        client_id=client_id,
        data_path="/tmp/unused.csv",
        model_type="logistic_regression",
        uniform_contextual_prior=uniform,
    )


def test_planted_prior_by_client_id():
    assert _client("hospital_09_compromised_iomt", uniform=False).get_contextual_risk(0.0) == 0.35
    assert _client("hospital_05_low_quality_iomt", uniform=False).get_contextual_risk(0.0) == 0.2
    assert _client("hospital_01_high_quality_iomt", uniform=False).get_contextual_risk(0.0) == 0.1


def test_uniform_prior_overrides_tier_lookup():
    for cid in (
        "hospital_09_compromised_iomt",
        "hospital_05_low_quality_iomt",
        "hospital_01_high_quality_iomt",
    ):
        assert _client(cid, uniform=True).get_contextual_risk(0.0) == 0.1


def test_uniform_prior_still_adds_operational_penalty():
    c = _client("hospital_09_compromised_iomt", uniform=True)
    assert abs(c.get_contextual_risk(1.0) - 0.75) < 1e-9  # 0.1 + 0.65


def test_resolve_metrics_run_id_b1_uniform():
    assert resolve_metrics_run_id("trustfed_b1", dataset="iomt") == "trustfed_b1_iomt"
    assert (
        resolve_metrics_run_id("trustfed_b1", dataset="iomt", uniform_contextual_prior=True)
        == "trustfed_b1_uniform_iomt"
    )
