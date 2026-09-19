#!/usr/bin/env python3
"""WP1: create/verify canonical governance SQLite DB from colleague seed JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from governance.canonical_schema import EXPECTED_SEED_COUNTS, SCHEMA_VERSION  # noqa: E402
from governance.device_binding import load_device_aliases, load_device_roster  # noqa: E402
from governance.seed_import import ensure_canonical_db  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=Path,
        default=ROOT / "results/trustfed_agent/governance/policy_repo_iomt.sqlite",
    )
    p.add_argument(
        "--seed",
        type=Path,
        default=ROOT / "data/governance/colleague_seed_v1.json",
    )
    p.add_argument("--replace-seed", action="store_true")
    p.add_argument(
        "--backup",
        action="store_true",
        help="If DB exists, copy to *.pre_canonical.bak before recreate",
    )
    args = p.parse_args()

    if args.backup and args.db.exists():
        bak = args.db.with_suffix(args.db.suffix + ".pre_canonical.bak")
        shutil.copy2(args.db, bak)
        print(f"backed up -> {bak}")
        args.db.unlink()

    summary = ensure_canonical_db(
        args.db, args.seed, replace_seed=args.replace_seed or not args.db.exists()
    )
    aliases = load_device_aliases()
    roster = load_device_roster()
    roster_ids = set((roster.get("participant_to_device_identifier") or {}).values())
    # Validate roster identifiers exist in DB
    import sqlite3

    conn = sqlite3.connect(str(args.db))
    known = {
        r[0]
        for r in conn.execute("SELECT device_identifier FROM medical_device_profiles")
    }
    conn.close()
    missing = sorted(roster_ids - known)
    if missing:
        raise SystemExit(f"roster device_identifier missing from seed: {missing}")

    out = {
        **summary,
        "expected": EXPECTED_SEED_COUNTS,
        "aliases": len(aliases),
        "roster_overrides": len(roster_ids),
        "ok": True,
    }
    print(json.dumps(out, indent=2))
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["counts"] == EXPECTED_SEED_COUNTS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
