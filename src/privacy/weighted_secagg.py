"""Weighted secure aggregation: mask v_i = α_i θ_i so the server sees only the sum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from privacy.exceptions import (
    DimensionError,
    EmptyCohortError,
    PlaintextLeakError,
    ShareError,
)
from privacy.pairwise_mask import (
    pairwise_mask_for,
    pairwise_seed,
    prg_mask,
    public_key,
    shared_secret,
)
from privacy.quantize import (
    DEFAULT_SCALE,
    dequantize,
    pack_lr_parameters,
    quantize,
    unpack_lr_parameters,
)
from privacy.secret_share import (
    bind_shares_to_ids,
    random_secret,
    reconstruct_secret,
    split_secret,
)

DEFAULT_THRESHOLD_N12 = 7


def shamir_threshold(n: int) -> int:
    """``e=7`` for N≥7 (honest-majority 10/12); smaller n for unit tests."""
    if n <= 1:
        return 1
    if n >= 7:
        return DEFAULT_THRESHOLD_N12
    return max(2, n // 2 + 1)


def assert_no_plaintext(update: Dict[str, Any]) -> None:
    if "X_train" in update or "y_train" in update:
        raise PlaintextLeakError("training matrix in privacy payload")
    if "parameters" in update:
        raise PlaintextLeakError("plaintext parameters in privacy payload")
    if update.get("model") is not None:
        raise PlaintextLeakError("sklearn model object in privacy payload")


@dataclass
class ClientMaskState:
    client_id: str
    secret: int
    public: int
    shares_held: Dict[str, Tuple[int, int]] = field(default_factory=dict)


class WeightedSecAggRound:
    """One communication round of pairwise masking + Shamir dropout."""

    def __init__(
        self,
        client_ids: Sequence[str],
        dim: int,
        *,
        threshold: Optional[int] = None,
        scale: int = DEFAULT_SCALE,
    ):
        self.client_ids: List[str] = [str(c) for c in client_ids]
        self.n = len(self.client_ids)
        if self.n == 0:
            raise EmptyCohortError("empty roster")
        self.dim = int(dim)
        self.threshold = int(threshold if threshold is not None else shamir_threshold(self.n))
        self.scale = int(scale)
        self._states: Dict[str, ClientMaskState] = {}
        self._public: Dict[str, int] = {}
        self._ready = False

    def setup_keys(self) -> None:
        n = self.n
        t = self.threshold
        raw_shares: Dict[str, List[Tuple[int, int]]] = {}
        for cid in self.client_ids:
            secret = random_secret()
            pub = public_key(secret)
            st = ClientMaskState(client_id=cid, secret=secret, public=pub)
            self._states[cid] = st
            self._public[cid] = pub
            raw_shares[cid] = split_secret(secret, n, t)

        for owner in self.client_ids:
            by_id = bind_shares_to_ids(self.client_ids, raw_shares[owner])
            for holder, xy in by_id.items():
                if holder == owner:
                    continue
                self._states[holder].shares_held[owner] = xy
        self._ready = True

    def mask_vector(self, client_id: str, v_float: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise ShareError("setup_keys() was not called")
        cid = str(client_id)
        vec = np.asarray(v_float, dtype=np.float64).ravel()
        if vec.size != self.dim:
            raise DimensionError(f"{cid}: expected dim {self.dim}, got {vec.size}")
        q = quantize(vec, self.scale)
        mask = pairwise_mask_for(
            cid, self.client_ids, self._states[cid].secret, self._public, self.dim
        )
        return q + mask

    def _reconstruct_secret(self, dropped_id: str, online: Sequence[str]) -> int:
        holders = []
        for oid in online:
            xy = self._states[oid].shares_held.get(dropped_id)
            if xy is not None:
                holders.append(xy)
        if len(holders) < self.threshold:
            raise ShareError(
                f"cannot reconstruct {dropped_id}: {len(holders)} < e={self.threshold}"
            )
        return reconstruct_secret(holders, self.threshold)

    def unmask_sum(
        self,
        masked: Dict[str, np.ndarray],
        *,
        expected_online: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        """Return Σ v_i for clients who submitted a mask (plus dropout correction)."""
        if not masked:
            raise EmptyCohortError("no masked vectors")
        submitted = {str(k): np.asarray(v, dtype=np.int64).ravel() for k, v in masked.items()}
        for cid, vec in submitted.items():
            if vec.size != self.dim:
                raise DimensionError(f"{cid}: expected dim {self.dim}, got {vec.size}")

        roster: Set[str] = set(self.client_ids)
        if expected_online is not None:
            roster = {str(x) for x in expected_online}
        online = list(submitted.keys())
        dropped = [c for c in roster if c not in submitted]

        acc = np.zeros(self.dim, dtype=np.int64)
        for vec in submitted.values():
            acc = acc + vec

        # Dropout: cancel leftover pairwise terms using reconstructed a_k only
        # (not the online clients' secrets).
        for kid in dropped:
            a_k = self._reconstruct_secret(kid, online)
            leftover = np.zeros(self.dim, dtype=np.int64)
            for oid in online:
                dh = shared_secret(a_k, self._public[oid])
                seed = pairwise_seed(oid, kid, dh)
                mask = prg_mask(seed, self.dim)
                if str(oid) < str(kid):
                    leftover = leftover + mask
                else:
                    leftover = leftover - mask
            acc = acc - leftover
        return acc

    def dequantize_sum(self, qsum: np.ndarray) -> np.ndarray:
        return dequantize(qsum, self.scale)

    def share_blob_bytes(self) -> int:
        """Approximate Shamir share payload per client (two ints × (N-1))."""
        return max(self.n - 1, 0) * 32


def weighted_payload(
    params: Dict[str, Any],
    alpha: float,
) -> Tuple[np.ndarray, int]:
    packed = pack_lr_parameters(params)
    return packed * float(alpha), int(packed.size)


def privacy_update_from_masked(
    *,
    client_id: str,
    trust: float,
    aggregation_weight: float,
    masked: np.ndarray,
    share_blob_bytes: int,
    n_features: int,
    classes: np.ndarray,
    collaboration_share: bool,
    round_num: Optional[int],
) -> Dict[str, Any]:
    """Aggregator-facing dict: no θ, no X, no sklearn model."""
    mv = np.asarray(masked, dtype=np.int64)
    return {
        "client_id": client_id,
        "trust": float(trust),
        "aggregation_weight": float(aggregation_weight),
        "masked_vector": mv,
        "share_blob_bytes": int(share_blob_bytes),
        "n_features": int(n_features),
        "lr_classes": np.asarray(classes),
        "collaboration_share": bool(collaboration_share),
        "masked": True,
        "round": round_num,
        "model": None,
    }


class InstalledLogisticRegression:
    """Averaged LR coefficients without ``fit()`` (numpy predict; sklearn-shaped attrs)."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        if self.coef_.ndim == 1:
            self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = np.asarray(intercept, dtype=np.float64).ravel()
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(self.coef_.shape[1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float64)
        w = np.asarray(self.coef_, dtype=np.float64).reshape(-1)
        b = float(np.asarray(self.intercept_, dtype=np.float64).reshape(-1)[0])
        return (Xn * w).sum(axis=1) + b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(self.decision_function(X), dtype=np.float64).ravel()
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]


def install_logistic_from_avg(
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
) -> InstalledLogisticRegression:
    return InstalledLogisticRegression(coef, intercept, classes)
