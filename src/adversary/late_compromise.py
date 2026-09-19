"""
Late compromise schedule for natural-partition FL (Trusted-plan §7–8 / WP4+WP7).

Warm-up: all clients train benign for rounds 1 .. t_a−1.
At t_a: designated attackers receive poison (label_flip / sign_flip / on_off).
Val/test refs are never poisoned.

Poison is always derived from immutable train originals so restore+re-apply is safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import hashlib

import numpy as np
import pandas as pd


DEFAULT_ATTACKER_IDS = ("client_03", "client_07", "client_11")
POISON_MODES = (
    "label_flip",
    "sign_flip",
    "on_off",
    "label_flip_on_off",
    "comm_skip",  # WP-A A3: attackers withhold uploads (C↓); train labels stay clean
)


@dataclass
class LateCompromiseConfig:
    compromise_round: int = 20
    attacker_client_ids: List[str] = field(default_factory=lambda: list(DEFAULT_ATTACKER_IDS))
    poison_mode: str = "label_flip"
    flip_p: float = 0.5
    adversary_fraction: Optional[float] = None  # if set, overrides explicit ids by fraction of N
    seed: int = 42
    enabled: bool = True
    # WP7 on–off: after t_a, attack for `on_off_attack_rounds`, then recover (benign)
    # If on_off_period > 0, alternate attack/benign blocks of that length instead.
    on_off_attack_rounds: int = 5
    on_off_period: int = 0  # 0 ⇒ single attack window then permanent recovery
    # WP7 sign-flip scale on uploaded parameters
    sign_flip_scale: float = -1.0
    # Path STRAG (T1): honest intermittent misses (disjoint from attackers)
    benign_straggler_enabled: bool = False
    benign_n_stragglers: int = 0
    benign_skip_prob_q: float = 0.5
    benign_selection: str = "last_non_attackers_sorted"
    benign_rng_salt: str = "path_strag_benign"

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "LateCompromiseConfig":
        if not raw:
            return cls(enabled=False)
        ids = raw.get("attacker_client_ids") or list(DEFAULT_ATTACKER_IDS)
        mode = str(raw.get("poison_mode", "label_flip"))
        bs = raw.get("benign_straggler") or {}
        rng = bs.get("rng") or {}
        return cls(
            compromise_round=int(raw.get("compromise_round", 20)),
            attacker_client_ids=[str(x) for x in ids],
            poison_mode=mode,
            flip_p=float(raw.get("flip_p", 0.5)),
            adversary_fraction=(
                float(raw["adversary_fraction"])
                if raw.get("adversary_fraction") is not None
                else None
            ),
            seed=int(raw.get("seed", 42)),
            enabled=bool(raw.get("enabled", True)),
            on_off_attack_rounds=int(raw.get("on_off_attack_rounds", 5)),
            on_off_period=int(raw.get("on_off_period", 0)),
            sign_flip_scale=float(raw.get("sign_flip_scale", -1.0)),
            benign_straggler_enabled=bool(bs.get("enabled", False)),
            benign_n_stragglers=int(bs.get("n_stragglers", 0) or 0),
            benign_skip_prob_q=float(bs.get("skip_prob_q", 0.5)),
            benign_selection=str(bs.get("selection", "last_non_attackers_sorted")),
            benign_rng_salt=str(rng.get("salt", bs.get("rng_salt", "path_strag_benign"))),
        )


def resolve_attacker_ids(
    client_ids: Sequence[str],
    cfg: LateCompromiseConfig,
) -> List[str]:
    """Pick attacker set from fraction or explicit IDs (must exist in client_ids)."""
    ordered = sorted(client_ids)
    if cfg.adversary_fraction is not None:
        frac = max(0.0, min(1.0, float(cfg.adversary_fraction)))
        k = int(round(frac * len(ordered)))
        if k <= 0:
            return []
        if k >= len(ordered):
            return list(ordered)
        idxs = np.linspace(0, len(ordered) - 1, num=k, dtype=int)
        return [ordered[i] for i in sorted(set(int(i) for i in idxs.tolist()))]

    wanted = set(cfg.attacker_client_ids)
    found = [c for c in ordered if c in wanted]
    missing = wanted - set(found)
    if missing:
        remapped = []
        for m in list(missing):
            if m.isdigit():
                remapped.append(f"client_{int(m):02d}")
            elif m.startswith("client_"):
                remapped.append(m)
        found2 = [c for c in ordered if c in set(remapped)]
        found = sorted(set(found) | set(found2))
        still = set(remapped) - set(found) if remapped else missing - set(found)
        if still:
            print(f"[LateCompromise] warning: attacker ids not found: {sorted(still)}")
    return found


def resolve_straggler_ids(
    client_ids: Sequence[str],
    attacker_ids: Iterable[str],
    n_stragglers: int,
    selection: str = "last_non_attackers_sorted",
) -> List[str]:
    """Honest stragglers: disjoint from attackers; deterministic given client set."""
    attackers = set(attacker_ids)
    non = [c for c in sorted(client_ids) if c not in attackers]
    n = max(0, int(n_stragglers))
    if n <= 0 or not non:
        return []
    if selection != "last_non_attackers_sorted":
        raise ValueError(
            f"benign_straggler.selection={selection!r}; "
            "only 'last_non_attackers_sorted' is supported"
        )
    return list(non[-min(n, len(non)) :])


def fraction_tag(frac: Optional[float], n_attackers: int, n_clients: int = 12) -> str:
    """Stable run-id suffix for adversary burden."""
    if frac is not None:
        pct = int(round(100 * float(frac)))
        return f"poison{pct}"
    if n_attackers <= 0:
        return "benign"
    pct = int(round(100 * n_attackers / max(n_clients, 1)))
    return f"poison{pct}"


class LateCompromiseController:
    """Applies / clears train poison and optional update sign-flip per schedule."""

    def __init__(self, cfg: LateCompromiseConfig, client_ids: Sequence[str]):
        self.cfg = cfg
        mode = (cfg.poison_mode or "label_flip").strip().lower()
        if mode not in POISON_MODES:
            raise ValueError(f"poison_mode={mode!r}; expected one of {POISON_MODES}")
        self.cfg.poison_mode = mode
        self.attacker_ids: Set[str] = set(resolve_attacker_ids(client_ids, cfg))
        self.straggler_ids: Set[str] = set()
        if cfg.benign_straggler_enabled and cfg.benign_n_stragglers > 0:
            self.straggler_ids = set(
                resolve_straggler_ids(
                    client_ids,
                    self.attacker_ids,
                    cfg.benign_n_stragglers,
                    cfg.benign_selection,
                )
            )
            overlap = self.straggler_ids & self.attacker_ids
            if overlap:
                raise RuntimeError(
                    f"benign stragglers overlap attackers: {sorted(overlap)}"
                )
        self._poisoned: Set[str] = set()
        self._flip_index_cache: Dict[str, np.ndarray] = {}
        self._skip_reason_cache: Dict[tuple, Optional[str]] = {}
        self.n_clients = len(client_ids)

    @property
    def t_a(self) -> int:
        return int(self.cfg.compromise_round)

    def is_attacker(self, client_id: str) -> bool:
        return client_id in self.attacker_ids

    def uses_label_poison(self) -> bool:
        return self.cfg.poison_mode in ("label_flip", "on_off", "label_flip_on_off")

    def uses_sign_flip(self) -> bool:
        return self.cfg.poison_mode in ("sign_flip",)

    def uses_on_off(self) -> bool:
        return self.cfg.poison_mode in ("on_off", "label_flip_on_off")

    def is_attack_active(self, round_num: int) -> bool:
        """True when attackers should be actively poisoning this round."""
        if not self.cfg.enabled or round_num < self.t_a:
            return False
        if not self.uses_on_off():
            return True
        # on–off schedule
        offset = int(round_num - self.t_a)
        period = int(self.cfg.on_off_period)
        if period > 0:
            # Alternate: [0, period) attack, [period, 2*period) benign, ...
            return (offset % (2 * period)) < period
        # Single window then recovery
        return offset < int(self.cfg.on_off_attack_rounds)

    def is_compromised_round(self, round_num: int) -> bool:
        """Backward-compatible: any post-t_a round counts as compromise era for GT."""
        return bool(self.cfg.enabled) and round_num >= self.t_a

    def is_recovery_round(self, round_num: int) -> bool:
        """Post-t_a round where attackers are currently benign (on–off recovery phase)."""
        return self.is_compromised_round(round_num) and not self.is_attack_active(round_num)

    def _flip_indices(self, client_id: str, n: int) -> np.ndarray:
        cached = self._flip_index_cache.get(client_id)
        if cached is not None and (len(cached) == 0 or (len(cached) > 0 and cached.max() < n)):
            return cached
        digest = hashlib.sha256(f"{self.cfg.seed}|{client_id}|label_flip".encode()).digest()
        seed_mat = int.from_bytes(digest[:8], "little") % (2**31 - 1)
        rng = np.random.default_rng(seed_mat)
        n_flip = int(round(self.cfg.flip_p * n))
        n_flip = max(0, min(n, n_flip))
        idx = rng.choice(n, size=n_flip, replace=False) if n_flip > 0 else np.array([], dtype=int)
        self._flip_index_cache[client_id] = idx
        return idx

    def apply_label_flip(self, client) -> int:
        """Restore clean train labels then flip a deterministic subset. Returns #flipped."""
        if client.y_train_original is None or client.X_train_original is None:
            return 0
        client.X_train = client.X_train_original.copy()
        y = client.y_train_original.copy()
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        else:
            y = y.copy()
        idx = self._flip_indices(client.client_id, len(y))
        if len(idx) > 0:
            y.iloc[idx] = 1 - y.iloc[idx].astype(int)
        client.y_train = y
        client.val_metrics = None
        self._poisoned.add(client.client_id)
        return int(len(idx))

    def clear_poison(self, client) -> None:
        if client.restore_training_data_from_original():
            self._poisoned.discard(client.client_id)

    def sync_round(self, round_num: int, clients: Iterable) -> Dict[str, Any]:
        """
        Call at the start of each FL round.

        Returns dict with flipped counts and phase flags.
        """
        report: Dict[str, Any] = {
            "flipped": {},
            "attack_active": False,
            "recovery": False,
            "phase": "warmup",
        }
        self._skip_reason_cache.clear()
        if not self.cfg.enabled:
            report["phase"] = "disabled"
            return report

        if round_num < self.t_a:
            for client in clients:
                if self.is_attacker(client.client_id):
                    self.clear_poison(client)
            report["phase"] = "warmup"
            return report

        active = self.is_attack_active(round_num)
        report["attack_active"] = active
        report["recovery"] = (not active) and self.uses_on_off()
        report["phase"] = "attack" if active else ("recovery" if self.uses_on_off() else "attack")

        for client in clients:
            if not self.is_attacker(client.client_id):
                continue
            if active and self.uses_label_poison():
                n = self.apply_label_flip(client)
                report["flipped"][client.client_id] = n
            else:
                # sign_flip-only: keep clean train; recovery / warmup: clear labels
                self.clear_poison(client)
        return report

    def maybe_sign_flip_update(self, client_id: str, update: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        """Negate uploaded parameters for sign_flip attackers during attack phases."""
        if not self.uses_sign_flip():
            return update
        if not self.is_attacker(client_id) or not self.is_attack_active(round_num):
            return update
        params = update.get("parameters")
        if not isinstance(params, dict):
            return update
        scale = float(self.cfg.sign_flip_scale)
        flipped = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                flipped[k] = (v * scale).astype(v.dtype, copy=False)
            elif isinstance(v, (int, float)):
                flipped[k] = type(v)(v * scale)
            else:
                flipped[k] = v
        out = dict(update)
        out["parameters"] = flipped
        out["sign_flipped"] = True
        return out

    def should_skip_participation(self, client_id: str, round_num: int) -> bool:
        """Withhold participation so communication reliability C falls.

        ``comm_skip``: after ``t_a``, attackers skip every active attack round
        (clean local labels; no upload). This is the WP-A A3 probe: B1 (V-only)
        cannot see C, while B2 can.

        Path STRAG: additionally, designated honest stragglers skip i.i.d. with
        probability ``benign_skip_prob_q`` on post-``t_a`` rounds (seeded).
        Stragglers are never attackers and are not GT-positive for AUROC.
        """
        reason = self.participation_skip_reason(client_id, round_num)
        return reason is not None

    def participation_skip_reason(
        self, client_id: str, round_num: int
    ) -> Optional[str]:
        """Return skip cause for logging, or None if the client participates.

        Memoized per (client_id, round) so train-loop checks and round-log
        fields share one Bernoulli draw.
        """
        key = (str(client_id), int(round_num))
        if key in self._skip_reason_cache:
            return self._skip_reason_cache[key]
        reason: Optional[str] = None
        if self.cfg.enabled:
            if (
                self.cfg.poison_mode == "comm_skip"
                and self.is_attacker(client_id)
                and self.is_attack_active(round_num)
            ):
                reason = "attacker_comm_skip"
            elif self._benign_straggler_skips(client_id, round_num):
                reason = "benign_straggler"
        self._skip_reason_cache[key] = reason
        return reason

    def _benign_straggler_skips(self, client_id: str, round_num: int) -> bool:
        if not self.cfg.benign_straggler_enabled:
            return False
        if client_id not in self.straggler_ids:
            return False
        if int(round_num) < self.t_a:
            return False
        q = float(self.cfg.benign_skip_prob_q)
        if q <= 0.0:
            return False
        if q >= 1.0:
            return True
        digest = hashlib.sha256(
            f"{self.cfg.seed}|{self.cfg.benign_rng_salt}|{client_id}|{int(round_num)}".encode()
        ).digest()
        seed_mat = int.from_bytes(digest[:8], "little") % (2**31 - 1)
        rng = np.random.default_rng(seed_mat)
        return bool(rng.random() < q)

    def summarize(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "compromise_round": self.t_a,
            "poison_mode": self.cfg.poison_mode,
            "flip_p": self.cfg.flip_p,
            "attacker_client_ids": sorted(self.attacker_ids),
            "adversary_fraction": self.cfg.adversary_fraction,
            "n_attackers": len(self.attacker_ids),
            "on_off_attack_rounds": self.cfg.on_off_attack_rounds,
            "on_off_period": self.cfg.on_off_period,
            "sign_flip_scale": self.cfg.sign_flip_scale,
            "currently_poisoned": sorted(self._poisoned),
            "fraction_tag": fraction_tag(
                self.cfg.adversary_fraction, len(self.attacker_ids), self.n_clients or 12
            ),
            "benign_straggler": {
                "enabled": self.cfg.benign_straggler_enabled,
                "n_stragglers": len(self.straggler_ids),
                "straggler_client_ids": sorted(self.straggler_ids),
                "skip_prob_q": self.cfg.benign_skip_prob_q,
                "selection": self.cfg.benign_selection,
                "rng_salt": self.cfg.benign_rng_salt,
            },
        }
