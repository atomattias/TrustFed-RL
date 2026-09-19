#!/usr/bin/env python3
"""Check that IoMT experiment datasets are reachable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment_config.json"


def _count_csv(path: Path) -> int:
    return len(list(path.glob("*.csv"))) if path.exists() else 0


def main() -> int:
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    ds = cfg.get("dataset", {})

    iomt_dir = (ROOT / ds.get("iomt_dir", "data/CSVs/iomt_clients")).resolve()
    iomt_test = (ROOT / ds.get("iomt_test_csv", "data/CSVs/iomt_test_set.csv")).resolve()

    print("IoMT dataset check\n" + "-" * 40)
    n_csv = _count_csv(iomt_dir)
    print(f"  iomt       {iomt_dir}")
    print(f"             -> {n_csv} CSV files" if n_csv else "             -> MISSING or empty")
    if n_csv == 0:
        print("             -> Run: bash scripts/setup_iomt_data.sh --demo")
    print(f"  iomt_test  {iomt_test}")
    print(f"             -> {'OK' if iomt_test.exists() else 'MISSING'}")

    ok = n_csv > 0 and iomt_test.exists()
    if n_csv > 0 and not iomt_test.exists():
        print("             -> iomt clients exist but test set missing")

    if not ok:
        print("\nPrepare IoMT data: bash scripts/setup_iomt_data.sh --demo")
        print("Or see docs/IOMT_DATASET_SETUP.md")
        return 1
    print("\nData paths OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
