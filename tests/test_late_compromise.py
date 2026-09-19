"""WP4/WP7: warm-up + late compromise (label_flip, sign_flip, on_off)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adversary.late_compromise import (
    LateCompromiseConfig,
    LateCompromiseController,
    fraction_tag,
    resolve_attacker_ids,
)
from evaluation import compute_trust_recovery
from federated_client import FederatedClient


class _ToyClient:
    def __init__(self, client_id: str, y: np.ndarray):
        self.client_id = client_id
        self.y_train_original = pd.Series(y.copy())
        self.X_train_original = pd.DataFrame({"f": np.arange(len(y), dtype=float)})
        self.y_train = self.y_train_original.copy()
        self.X_train = self.X_train_original.copy()
        self.y_val = pd.Series(y.copy())

    def restore_training_data_from_original(self) -> bool:
        self.X_train = self.X_train_original.copy()
        self.y_train = self.y_train_original.copy()
        return True


def test_warmup_no_poison_before_ta():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clients = [_ToyClient("client_03", y), _ToyClient("client_01", y)]
    ctrl = LateCompromiseController(
        LateCompromiseConfig(compromise_round=20, flip_p=0.5, seed=42),
        [c.client_id for c in clients],
    )
    ctrl.sync_round(19, clients)
    assert (clients[0].y_train.values == y).all()
    assert (clients[1].y_train.values == y).all()


def test_poison_at_ta_and_val_untouched():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    c = _ToyClient("client_03", y)
    ctrl = LateCompromiseController(
        LateCompromiseConfig(
            compromise_round=20,
            flip_p=0.5,
            seed=42,
            attacker_client_ids=["client_03"],
        ),
        ["client_03"],
    )
    report = ctrl.sync_round(20, [c])
    assert report["flipped"]["client_03"] == 5
    assert report["attack_active"] is True
    assert not (c.y_train.values == y).all()
    assert (c.y_val.values == y).all()


def test_restore_then_reapply_is_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    c = _ToyClient("client_07", y)
    ctrl = LateCompromiseController(
        LateCompromiseConfig(
            compromise_round=10,
            flip_p=0.5,
            seed=7,
            attacker_client_ids=["client_07"],
        ),
        ["client_07"],
    )
    ctrl.sync_round(10, [c])
    y1 = c.y_train.copy()
    c.restore_training_data_from_original()
    assert (c.y_train.values == y).all()
    ctrl.sync_round(10, [c])
    assert (c.y_train.values == y1.values).all()


def test_on_off_recovers_labels():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    c = _ToyClient("client_03", y)
    ctrl = LateCompromiseController(
        LateCompromiseConfig(
            compromise_round=5,
            poison_mode="on_off",
            on_off_attack_rounds=2,
            flip_p=0.5,
            seed=42,
            attacker_client_ids=["client_03"],
        ),
        ["client_03"],
    )
    r5 = ctrl.sync_round(5, [c])
    assert r5["attack_active"] and not (c.y_train.values == y).all()
    r6 = ctrl.sync_round(6, [c])
    assert r6["attack_active"]
    r7 = ctrl.sync_round(7, [c])
    assert r7["recovery"] and (c.y_train.values == y).all()


def test_sign_flip_update():
    ctrl = LateCompromiseController(
        LateCompromiseConfig(
            compromise_round=1,
            poison_mode="sign_flip",
            attacker_client_ids=["client_03"],
        ),
        ["client_03", "client_01"],
    )
    update = {"client_id": "client_03", "parameters": {"coef_": np.array([1.0, -2.0]), "intercept_": 3.0}}
    out = ctrl.maybe_sign_flip_update("client_03", update, round_num=1)
    assert out["sign_flipped"] is True
    assert np.allclose(out["parameters"]["coef_"], [-1.0, 2.0])
    assert out["parameters"]["intercept_"] == -3.0
    # Benign client untouched
    clean = ctrl.maybe_sign_flip_update("client_01", update, round_num=1)
    assert "sign_flipped" not in clean


def test_adversary_fraction_10_to_40():
    ids = [f"client_{i:02d}" for i in range(1, 13)]
    for frac, expect_n in [(0.1, 1), (0.2, 2), (0.25, 3), (0.3, 4), (0.4, 5)]:
        cfg = LateCompromiseConfig(adversary_fraction=frac, attacker_client_ids=[])
        got = resolve_attacker_ids(ids, cfg)
        assert len(got) == expect_n, (frac, got)
        assert fraction_tag(frac, len(got), 12) == f"poison{int(round(100*frac))}"


def test_trust_recovery_metric():
    logs = []
    for r, t_att, attack, recovery in [
        (1, 0.8, False, False),
        (5, 0.3, True, False),
        (6, 0.25, True, False),
        (7, 0.55, False, True),
        (8, 0.70, False, True),
    ]:
        logs.append({
            "round": r,
            "attack_active": attack,
            "recovery_phase": recovery,
            "compromise_active": r >= 5,
            "client_trust": {
                "client_01": {"T": 0.8, "is_attacker": False},
                "client_03": {"T": t_att, "is_attacker": True},
            },
        })
    out = compute_trust_recovery(logs, compromise_round=5, attacker_ids=["client_03"])
    assert out["available"]
    assert out["recovery_delta"] > 0


def test_real_client_poison_drops_clean_val_f1():
    clients_dir = ROOT / "data" / "CSVs" / "iomt_natural" / "clients"
    val_ref = ROOT / "data" / "CSVs" / "iomt_natural" / "iomt_val_ref.csv"
    if not clients_dir.exists() or not val_ref.exists():
        return
    path = clients_dir / "client_03.csv"
    client = FederatedClient(
        client_id="client_03",
        data_path=str(path),
        model_type="logistic_regression",
        clean_validation_source=str(val_ref),
        use_shared_val_ref=True,
        performance_signal="f1_score",
        uniform_contextual_prior=True,
    )
    client.load_data()
    client.train()
    client.evaluate()
    f1_clean = float(client.val_metrics["f1_score"])

    ctrl = LateCompromiseController(
        LateCompromiseConfig(
            compromise_round=1,
            flip_p=0.5,
            seed=42,
            attacker_client_ids=["client_03"],
        ),
        ["client_03"],
    )
    ctrl.sync_round(1, [client])
    client.train()
    client.evaluate()
    f1_poison = float(client.val_metrics["f1_score"])
    assert f1_poison < f1_clean - 0.02, (f1_clean, f1_poison)


def test_train_invalidates_stale_val_metrics():
    """Regression: train() must drop cached V so next evaluate is the new model."""
    clients_dir = ROOT / "data" / "CSVs" / "iomt_natural" / "clients"
    val_ref = ROOT / "data" / "CSVs" / "iomt_natural" / "iomt_val_ref.csv"
    if not clients_dir.exists() or not val_ref.exists():
        return
    client = FederatedClient(
        client_id="client_03",
        data_path=str(clients_dir / "client_03.csv"),
        model_type="logistic_regression",
        clean_validation_source=str(val_ref),
        use_shared_val_ref=True,
        performance_signal="f1_score",
        uniform_contextual_prior=True,
    )
    client.load_data()
    client.train()
    client.evaluate()
    planted = dict(client.val_metrics)
    planted["f1_score"] = 0.01
    planted["accuracy"] = 0.01
    client.val_metrics = planted
    client.train()
    assert client.val_metrics is None
    signals = client.compute_multi_signal_trust_signals()
    assert signals["V"] > 0.5
    assert abs(signals["V"] - 0.01) > 0.5


def test_experiment_runner_does_not_overwrite_natural_dynamic_adversary():
    """Ctor used to set dynamic_adversary=False then overwrite it to True."""
    clients = ROOT / "data" / "CSVs" / "iomt_natural" / "clients"
    if not clients.exists():
        return
    sys.path.insert(0, str(ROOT))
    from experiment import ExperimentRunner

    runner = ExperimentRunner(
        data_dir=str(clients),
        model_type="logistic_regression",
        random_state=42,
        num_rounds=8,
        use_multi_signal=False,
        dynamic_adversary=True,
    )
    assert runner.use_shared_val_ref is True
    assert runner.dynamic_adversary is False


def test_mid_training_always_syncs_late_compromise():
    clients = ROOT / "data" / "CSVs" / "iomt_natural" / "clients"
    if not clients.exists():
        return
    sys.path.insert(0, str(ROOT))
    from experiment import ExperimentRunner

    runner = ExperimentRunner(
        data_dir=str(clients),
        model_type="logistic_regression",
        random_state=42,
        num_rounds=8,
        use_multi_signal=False,
        dynamic_adversary=True,
    )
    seen = []

    class _Ctrl:
        t_a = 4
        cfg = type("C", (), {"poison_mode": "label_flip"})()

        def sync_round(self, round_num, _clients):
            seen.append(round_num)
            return {"attack_active": True, "flipped": {}}

    runner.late_compromise = _Ctrl()
    runner.dynamic_adversary = True  # even if someone flips it back
    runner._apply_mid_training_adversary(4)
    assert seen == [4]

def test_benign_straggler_disjoint_and_probabilistic():
    import json

    raw = json.loads(
        (ROOT / "config" / "iomt_natural_adversary_path_strag.json").read_text()
    )
    cfg = LateCompromiseConfig.from_dict(raw)
    cfg.seed = 42
    assert cfg.benign_straggler_enabled
    assert cfg.benign_n_stragglers == 3
    assert cfg.benign_skip_prob_q == 0.5
    ids = [f"client_{i:02d}" for i in range(1, 13)]
    ctrl = LateCompromiseController(cfg, ids)
    assert sorted(ctrl.attacker_ids) == ["client_01", "client_06", "client_12"]
    assert sorted(ctrl.straggler_ids) == ["client_09", "client_10", "client_11"]
    assert ctrl.attacker_ids.isdisjoint(ctrl.straggler_ids)

    for cid in ids:
        assert ctrl.participation_skip_reason(cid, 19) is None

    for cid in ctrl.attacker_ids:
        assert ctrl.participation_skip_reason(cid, 20) == "attacker_comm_skip"

    assert ctrl.participation_skip_reason("client_02", 20) is None

    r1 = ctrl.participation_skip_reason("client_09", 20)
    r2 = ctrl.participation_skip_reason("client_09", 20)
    assert r1 == r2
    assert r1 in (None, "benign_straggler")

    skips = sum(
        1
        for rnd in range(20, 30)
        if ctrl.participation_skip_reason("client_09", rnd) == "benign_straggler"
    )
    assert 1 <= skips <= 9


def test_benign_straggler_off_by_default():
    cfg = LateCompromiseConfig(
        poison_mode="comm_skip",
        adversary_fraction=0.25,
        compromise_round=20,
    )
    ids = [f"client_{i:02d}" for i in range(1, 13)]
    ctrl = LateCompromiseController(cfg, ids)
    assert not ctrl.straggler_ids
    assert ctrl.participation_skip_reason("client_09", 20) is None
    assert ctrl.participation_skip_reason("client_01", 20) == "attacker_comm_skip"


def test_benign_skip_reason_memoized():
    import json
    raw = json.loads((ROOT / "config" / "iomt_natural_adversary_path_strag.json").read_text())
    cfg = LateCompromiseConfig.from_dict(raw)
    cfg.seed = 42
    ids = [f"client_{i:02d}" for i in range(1, 13)]
    ctrl = LateCompromiseController(cfg, ids)
    ctrl.sync_round(20, [])  # clear cache
    a = ctrl.participation_skip_reason("client_09", 20)
    b = ctrl.participation_skip_reason("client_09", 20)
    assert a == b
    assert ("client_09", 20) in ctrl._skip_reason_cache

