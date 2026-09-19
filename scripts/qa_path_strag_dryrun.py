#!/usr/bin/env python3
"""QA-check Path STRAG confirmatory dry-run log (T1.3)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"

REQUIRED = [
    "--dataset iomt_natural",
    "--num-rounds 30",
    "--compromise-round 20",
    "--adversary-fraction 0.25",
    "--poison-mode comm_skip",
    "--late-compromise-config config/iomt_natural_adversary_path_strag.json",
    "--benign-skip-prob 0.5",
    "--uniform-b-prior",
    "--metrics-suffix path_strag",
]
FORBIDDEN = [
    "--metrics-suffix path_agg",
    "--metrics-suffix path1_c_only",
    "--metrics-suffix ecu_repl",
    "--metrics-suffix path_strag_explore",
    "--metrics-suffix path_strag_smoke",
    "--seed 42",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=REV / "path_strag_t1_3_dryrun.log")
    ap.add_argument("--expect-jobs", type=int, default=15)
    args = ap.parse_args()
    text = args.log.read_text()
    dry = [ln for ln in text.splitlines() if ln.startswith("DRY:")]
    issues: list[str] = []
    if len(dry) != args.expect_jobs:
        issues.append(f"dry lines={len(dry)} expected={args.expect_jobs}")

    arms = {"b1": 0, "bc": 0, "b2": 0}
    seeds: set[int] = set()
    for ln in dry:
        for req in REQUIRED:
            if req not in ln:
                issues.append(f"missing {req!r} in: {ln[:120]}")
                break
        for bad in FORBIDDEN:
            if bad in ln:
                issues.append(f"forbidden {bad!r} in: {ln[:120]}")
        m_base = re.search(r"--baseline (\S+)", ln)
        m_seed = re.search(r"--seed (\d+)", ln)
        if not m_base or not m_seed:
            issues.append(f"parse fail: {ln[:120]}")
            continue
        base = m_base.group(1)
        seed = int(m_seed.group(1))
        if base not in arms:
            issues.append(f"unexpected baseline {base}")
        else:
            arms[base] += 1
        if seed == 42:
            issues.append("seed 42 present")
        seeds.add(seed)

    if args.expect_jobs == 15:
        if seeds != {43, 44, 45, 46, 47}:
            issues.append(f"seeds={sorted(seeds)} expected 43-47")
        if arms != {"b1": 5, "bc": 5, "b2": 5}:
            issues.append(f"arm counts={arms} expected 5 each")

    out = {
        "log": str(args.log),
        "n_dry": len(dry),
        "expect_jobs": args.expect_jobs,
        "arms": arms,
        "seeds": sorted(seeds),
        "issues": issues,
        "pass": len(issues) == 0,
    }
    REV.mkdir(parents=True, exist_ok=True)
    path = REV / "path_strag_t1_3_dryrun_qa.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    if issues:
        print("T1.3 dry-run QA FAIL", file=sys.stderr)
        raise SystemExit(1)
    print("T1.3 dry-run QA PASS")


if __name__ == "__main__":
    main()
