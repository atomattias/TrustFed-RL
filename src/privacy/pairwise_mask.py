"""Diffie–Hellman pairwise seeds and cancelling PRG masks."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from privacy.secret_share import PRIME

DH_GENERATOR = 3


def public_key(secret: int) -> int:
    return pow(DH_GENERATOR, int(secret), PRIME)


def shared_secret(my_secret: int, peer_public: int) -> int:
    return pow(int(peer_public), int(my_secret), PRIME)


def pairwise_seed(client_i: str, client_j: str, dh_secret: int) -> bytes:
    """Symmetric seed for the unordered pair ``{i,j}``."""
    a, b = sorted((str(client_i), str(client_j)))
    material = f"{dh_secret}|{a}|{b}".encode("utf-8")
    return hashlib.sha256(material).digest()


def prg_mask(seed: bytes, dim: int) -> np.ndarray:
    """Deterministic ``int64`` mask in a range that sums without wrap for N≤32."""
    raw = hashlib.shake_256(seed).digest(int(dim) * 4)
    return np.frombuffer(raw, dtype=np.int32).astype(np.int64)


def pairwise_mask_for(
    client_id: str,
    peer_ids: Sequence[str],
    my_secret: int,
    public_keys: Dict[str, int],
    dim: int,
) -> np.ndarray:
    """Bonawitz-style signed pairwise mask for ``client_id``."""
    acc = np.zeros(int(dim), dtype=np.int64)
    for peer in peer_ids:
        if peer == client_id:
            continue
        dh = shared_secret(my_secret, public_keys[peer])
        seed = pairwise_seed(client_id, peer, dh)
        mask = prg_mask(seed, dim)
        if str(client_id) < str(peer):
            acc = acc + mask
        else:
            acc = acc - mask
    return acc
