#!/usr/bin/env python3
"""
Status checker for TrustFed-RL IoMT master matrix (EXPERIMENT_PLAN §3 / §5).

Usage:
  python scripts/check_trustfed_rl_iomt_status.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
CONFIG = ROOT / "config" / "experiment_config.json"

# Master matrix IDs only (no B0m / B5−λ).
PHASES = [
    ("B0", "run_trustfed_fedavg_iomt_seed_*.json", "FedAvg equal-weight"),
    ("B1", "run_trustfed_b1_iomt_seed_*.json", "6-signal trust-weighted FL"),
    ("B1-U", "run_trustfed_b1_uniform_iomt_seed_*.json", "B1 with uniform b_i=0.10 (no oracle prior)"),
    ("B3", "run_trustfed_governance_iomt_seed_*.json", "Governance only"),
    ("B4", "run_trustfed_rl_only_iomt_seed_*.json", "RL only (no governance)"),
    ("B5−S", "run_trustfed_agent_no_cr_iomt_seed_*.json", "Full stack without C,R"),
    ("B5", "run_trustfed_agent_iomt_seed_*.json", "Full TrustFed-RL (static)"),
    ("B5†", "run_trustfed_agent_co_adaptive_iomt_seed_*.json", "Full + co-adaptive"),
    ("B4†", "run_trustfed_rl_only_co_adaptive_iomt_seed_*.json", "RL-only + co-adaptive"),
    ("B5-P", "run_trustfed_agent_privacy_iomt_seed_*.json", "Full stack + weighted update hiding"),
]


def main() -> int:
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    if CONFIG.exists():
        seeds = list(json.loads(CONFIG.read_text()).get("seeds", seeds))
    n = len(seeds)

    print("TrustFed-RL IoMT matrix status (master)")
    print(f"Metrics dir: {METRICS}")
    print(f"Expected seeds: {n} ({seeds[0]}..{seeds[-1]})")
    print("-" * 72)

    all_ok = True
    for name, pattern, desc in PHASES:
        files = sorted(METRICS.glob(pattern)) if METRICS.exists() else []
        if name == "B5":
            files = [
                p for p in files
                if "no_cr" not in p.name
                and "co_adaptive" not in p.name
                and "no_lambda5" not in p.name
            ]
        if name == "B4":
            files = [p for p in files if "co_adaptive" not in p.name]
        found = {int(p.stem.split("_")[-1]) for p in files if p.stem.split("_")[-1].isdigit()}
        missing = [s for s in seeds if s not in found]
        status = "DONE" if not missing else f"MISSING {missing}"
        if missing:
            all_ok = False
        print(f"{name:<6} {len(found):>2}/{n}  {desc}")
        print(f"       -> {status}")

    print("-" * 72)
    summary = ROOT / "results" / "trustfed_agent" / "summary.json"
    print(f"summary.json: {'OK' if summary.exists() else 'MISSING (run: python run_experiments.py analyze)'}")
    print("Overall:", "READY FOR PAPER TABLES" if all_ok else "INCOMPLETE — resume matrix script")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
