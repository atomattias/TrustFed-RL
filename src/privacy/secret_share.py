"""Shamir secret sharing over a prime field (mask-seed reconstruction)."""

from __future__ import annotations

import secrets
from typing import Dict, Iterable, List, Sequence, Tuple

from privacy.exceptions import ShareError

# 127-bit Mersenne prime: fast for tests/sim; not a production modulus.
PRIME = (1 << 127) - 1


def _mod(x: int) -> int:
    return x % PRIME


def _modinv(a: int) -> int:
    return pow(_mod(a), PRIME - 2, PRIME)


def random_secret() -> int:
    """Uniform secret in ``[1, PRIME-1]``."""
    return secrets.randbelow(PRIME - 1) + 1


def split_secret(secret: int, n: int, threshold: int) -> List[Tuple[int, int]]:
    """Return ``n`` shares ``(x, y)`` with reconstruction threshold ``threshold``."""
    if not (1 <= threshold <= n):
        raise ShareError(f"invalid Shamir parameters n={n} t={threshold}")
    secret = _mod(int(secret))
    coeffs = [secret] + [secrets.randbelow(PRIME) for _ in range(threshold - 1)]

    def eval_poly(x: int) -> int:
        acc = 0
        xp = 1
        for c in coeffs:
            acc = _mod(acc + c * xp)
            xp = _mod(xp * x)
        return acc

    return [(i, eval_poly(i)) for i in range(1, n + 1)]


def reconstruct_secret(shares: Sequence[Tuple[int, int]], threshold: int) -> int:
    """Lagrange interpolate at 0. Requires at least ``threshold`` distinct shares."""
    if len(shares) < threshold:
        raise ShareError(f"need {threshold} shares, got {len(shares)}")
    pts = list(shares[:threshold])
    xs = [p[0] for p in pts]
    if len(set(xs)) < len(xs):
        raise ShareError("duplicate share x-coordinates")
    secret = 0
    for i, (xi, yi) in enumerate(pts):
        num, den = 1, 1
        for j, (xj, _) in enumerate(pts):
            if i == j:
                continue
            num = _mod(num * (-xj))
            den = _mod(den * (xi - xj))
        secret = _mod(secret + yi * num * _modinv(den))
    return secret


def bind_shares_to_ids(
    client_ids: Sequence[str], shares: Sequence[Tuple[int, int]]
) -> Dict[str, Tuple[int, int]]:
    """Map Shamir x-index shares onto ordered client ids."""
    if len(client_ids) != len(shares):
        raise ShareError("share count must match client roster")
    return {cid: shares[i] for i, cid in enumerate(client_ids)}


def shares_from_holders(
    holders: Iterable[Tuple[str, Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    return [xy for _, xy in holders]
