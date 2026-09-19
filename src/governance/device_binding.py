"""
Device-type aliases, participant roster, and device_id resolution (WP1–WP2).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

_DEFAULT_ALIASES = (
    Path(__file__).resolve().parents[2] / "config" / "governance_device_aliases.json"
)
_DEFAULT_ROSTER = (
    Path(__file__).resolve().parents[2] / "config" / "governance_device_roster.json"
)


def load_device_aliases(path: Optional[PathLike] = None) -> Dict[str, str]:
    """detection device_type (lowercase) → colleague device_type (UPPER)."""
    p = Path(path) if path else _DEFAULT_ALIASES
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("detection_to_device_type") or {}
    return {str(k).lower(): str(v).upper() for k, v in raw.items()}


def load_device_roster(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """
    Optional explicit participant → device_identifier overrides.
    Default resolver: alias type + hash(participant) % candidates.
    """
    p = Path(path) if path else _DEFAULT_ROSTER
    if not p.exists():
        return {"participant_to_device_identifier": {}, "fallback": "hash_by_type"}
    return json.loads(p.read_text(encoding="utf-8"))


def alias_detection_type(
    detection_device_type: str,
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    table = aliases if aliases is not None else load_device_aliases()
    key = str(detection_device_type or "").strip().lower()
    if not key:
        return "WORKSTATION"
    return table.get(key, key.upper())


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def resolve_device_profile(
    *,
    profiles_by_identifier: Mapping[str, Mapping[str, Any]],
    profiles_by_type: Mapping[str, Sequence[Mapping[str, Any]]],
    all_active_profiles: Sequence[Mapping[str, Any]],
    participant_id: str = "",
    detection_device_type: str = "",
    aliases: Optional[Mapping[str, str]] = None,
    roster: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve a medical_device_profiles row for this participant/detection.

    Order: roster override → alias(type) candidates (hash pick) →
    any ACTIVE MEDIUM non-life-support fallback → first ACTIVE profile.
    """
    roster = roster or {}
    overrides = roster.get("participant_to_device_identifier") or {}
    if participant_id and participant_id in overrides:
        ident = overrides[participant_id]
        row = profiles_by_identifier.get(ident)
        if row:
            return dict(row)

    dtype = alias_detection_type(detection_device_type, aliases)
    candidates = list(profiles_by_type.get(dtype) or [])
    # Prefer ACTIVE; never auto-bind QUARANTINED/INACTIVE when ACTIVE exists
    active = [
        c
        for c in candidates
        if str(c.get("device_status", "ACTIVE")).upper() == "ACTIVE"
    ]
    pool = active
    if not pool:
        # last resort: non-quarantined of this type
        pool = [
            c
            for c in candidates
            if str(c.get("device_status", "ACTIVE")).upper() not in {"QUARANTINED"}
        ] or candidates
    if pool:
        idx = _stable_index(f"{participant_id}|{dtype}", len(pool))
        return dict(pool[idx])

    # Unknown type fallback: MEDIUM, non-life-support, ACTIVE only
    fallbacks = [
        dict(p)
        for p in all_active_profiles
        if str(p.get("device_status", "ACTIVE")).upper() == "ACTIVE"
        and str(p.get("criticality", "")).upper() == "MEDIUM"
        and not int(p.get("life_support_status") or 0)
    ]
    if fallbacks:
        idx = _stable_index(participant_id or dtype or "default", len(fallbacks))
        return fallbacks[idx]

    active_any = [
        dict(p)
        for p in all_active_profiles
        if str(p.get("device_status", "ACTIVE")).upper() == "ACTIVE"
    ]
    if active_any:
        return active_any[0]
    return None
