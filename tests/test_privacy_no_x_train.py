"""Locality: privacy-mode updates must not carry training matrices."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from federated_client import FederatedClient
from privacy.exceptions import PlaintextLeakError
from privacy.weighted_secagg import assert_no_plaintext


def test_get_model_update_can_omit_training_data():
    client = FederatedClient.__new__(FederatedClient)
    client.client_id = "hospital_01"
    client.model_type = "logistic_regression"
    client.trust_score = 0.5
    client.val_metrics = {
        "accuracy": 0.9,
        "f1_score": 0.9,
        "precision": 0.9,
        "recall": 0.9,
    }
    client.current_round = 1
    client.performance_history = [{"round": 1}]
    client.train_metrics = None
    client.X_train = np.ones((4, 3))
    client.y_train = np.array([0, 1, 0, 1])
    dummy = MagicMock()
    dummy.coef_ = np.zeros((1, 3))
    dummy.intercept_ = np.zeros(1)
    dummy.classes_ = np.array([0, 1])
    client.model = dummy

    update = FederatedClient.get_model_update(client, round_num=1, include_data=False)
    assert "X_train" not in update
    assert "y_train" not in update
    assert "parameters" in update


def test_assert_no_plaintext_on_include_data_true():
    update = {"client_id": "h", "X_train": np.zeros((2, 2)), "y_train": np.zeros(2)}
    with pytest.raises(PlaintextLeakError):
        assert_no_plaintext(update)
