"""Unit tests for weighted pairwise masking (no IoMT data required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from privacy.exceptions import PlaintextLeakError, ShareError
from privacy.quantize import dequantize, pack_lr_parameters, quantize, unpack_lr_parameters
from privacy.secret_share import reconstruct_secret, split_secret
from privacy.weighted_secagg import (
    WeightedSecAggRound,
    assert_no_plaintext,
    install_logistic_from_avg,
    privacy_update_from_masked,
    weighted_payload,
)


def test_quantize_roundtrip():
    v = np.array([0.1, -0.25, 1.5])
    q = quantize(v)
    rec = dequantize(q)
    assert np.allclose(v, rec, atol=1e-6)


def test_pack_unpack_lr():
    params = {"coef": np.array([[0.2, -0.1, 0.05]]), "intercept": np.array([0.3])}
    packed = pack_lr_parameters(params)
    coef, intercept = unpack_lr_parameters(packed, n_features=3)
    assert np.allclose(coef, params["coef"])
    assert np.allclose(intercept, params["intercept"])


def test_shamir_roundtrip():
    secret = 123456789
    shares = split_secret(secret, n=5, threshold=3)
    rec = reconstruct_secret(shares[:3], threshold=3)
    assert rec == secret % ((1 << 127) - 1)


def test_sum_masks_equals_sum_v():
    ids = ["h1", "h2", "h3"]
    dim = 8
    rng = np.random.default_rng(0)
    thetas = [rng.normal(size=dim) for _ in ids]
    alphas = np.array([0.5, 0.3, 0.2])
    vs = [a * th for a, th in zip(alphas, thetas)]
    rnd = WeightedSecAggRound(ids, dim, threshold=2)
    rnd.setup_keys()
    masked = {cid: rnd.mask_vector(cid, v) for cid, v in zip(ids, vs)}
    qsum = rnd.unmask_sum(masked)
    rec = rnd.dequantize_sum(qsum)
    expected = sum(vs)
    assert np.allclose(rec, expected, atol=1e-5)


def test_dropout_matches_online_sum():
    ids = ["h1", "h2", "h3"]
    dim = 6
    rng = np.random.default_rng(1)
    vs = [rng.normal(size=dim) for _ in ids]
    rnd = WeightedSecAggRound(ids, dim, threshold=2)
    rnd.setup_keys()
    masked = {cid: rnd.mask_vector(cid, v) for cid, v in zip(ids, vs)}
    dropped = "h2"
    submitted = {k: v for k, v in masked.items() if k != dropped}
    qsum = rnd.unmask_sum(submitted, expected_online=ids)
    rec = rnd.dequantize_sum(qsum)
    expected = vs[0] + vs[2]
    assert np.allclose(rec, expected, atol=1e-5)


def test_alpha_applied_before_mask():
    ids = ["a", "b"]
    dim = 4
    theta = [np.ones(dim), np.ones(dim) * 2.0]
    alpha = np.array([0.25, 0.75])
    vs = [alpha[0] * theta[0], alpha[1] * theta[1]]
    rnd = WeightedSecAggRound(ids, dim, threshold=2)
    rnd.setup_keys()
    masked = {cid: rnd.mask_vector(cid, v) for cid, v in zip(ids, vs)}
    rec = rnd.dequantize_sum(rnd.unmask_sum(masked))
    assert np.allclose(rec, sum(vs), atol=1e-5)
    equal_vs = [1.0 * theta[0], 1.0 * theta[1]]
    assert not np.allclose(rec, sum(equal_vs), atol=1e-3)


def test_too_few_shares_raises():
    ids = ["h1", "h2", "h3", "h4"]
    rnd = WeightedSecAggRound(ids, 3, threshold=3)
    rnd.setup_keys()
    vs = {c: np.zeros(3) for c in ids}
    masked = {c: rnd.mask_vector(c, vs[c]) for c in ["h1"]}
    with pytest.raises(ShareError):
        rnd.unmask_sum(masked, expected_online=ids)


def test_privacy_payload_rejects_plaintext():
    bad = {"client_id": "h1", "parameters": {"coef": np.zeros((1, 2))}}
    with pytest.raises(PlaintextLeakError):
        assert_no_plaintext(bad)
    bad_x = {"client_id": "h1", "X_train": np.zeros((2, 2))}
    with pytest.raises(PlaintextLeakError):
        assert_no_plaintext(bad_x)
    ok = privacy_update_from_masked(
        client_id="h1",
        trust=0.5,
        aggregation_weight=0.5,
        masked=np.zeros(3, dtype=np.int64),
        share_blob_bytes=32,
        n_features=2,
        classes=np.array([0, 1]),
        collaboration_share=True,
        round_num=1,
    )
    assert_no_plaintext(ok)
    assert "parameters" not in ok
    assert ok["model"] is None
    assert ok["masked"] is True


def test_install_logistic_predicts():
    coef = np.array([[0.1, -0.2, 0.0]])
    intercept = np.array([0.0])
    model = install_logistic_from_avg(coef, intercept, np.array([0, 1]))
    X = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    pred = model.predict(X)
    assert pred.shape == (2,)
    proba = model.predict_proba(X)
    assert proba.shape == (2, 2)
