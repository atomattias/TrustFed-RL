#!/usr/bin/env python3
"""
Analyze natural-partition TrustFed metrics (WP5).

Computes / prints:
  - AUROC / AUPRC of (1−T) vs attacker GT
  - detection delay, false distrust
  - mean α for malicious vs benign
  - ΔF1 vs FedAvg when both runs are present

Usage:
  python scripts/analyze_iomt_natural_trust.py \\
    --input results/trustfed_agent/metrics \\
    --export results/trustfed_agent/iomt_natural_trust_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import (
    compute_natural_trust_discrimination,
    compute_trust_recovery,
    delta_f1_robust,
)


def load_runs(metrics_dir: Path, dataset: str = "iomt_natural") -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(metrics_dir.glob("run_*.json")):
        with open(path) as f:
            rec = json.load(f)
        if dataset and rec.get("dataset") not in (None, dataset):
            # also accept path-encoded dataset
            if dataset not in path.name and rec.get("dataset") != dataset:
                continue
        rec["_path"] = str(path)
        rows.append(rec)
    return rows


def analyze_one(rec: Dict[str, Any]) -> Dict[str, Any]:
    late = rec.get("late_compromise") or {}
    t_a = int(late.get("compromise_round") or 20)
    attackers = late.get("attacker_client_ids") or []
    disc = rec.get("trust_discrimination")
    if not disc or not disc.get("available"):
        disc = compute_natural_trust_discrimination(
            rec.get("round_logs") or [],
            compromise_round=t_a,
            attacker_ids=attackers,
        )
    recovery = rec.get("trust_recovery")
    if (not recovery or not recovery.get("available")) and late.get("poison_mode") in (
        "on_off", "label_flip_on_off",
    ):
        recovery = compute_trust_recovery(
            rec.get("round_logs") or [],
            compromise_round=t_a,
            attacker_ids=attackers,
        )
    return {
        "path": rec.get("_path"),
        "run_id": rec.get("run_id"),
        "approach": rec.get("approach"),
        "seed": rec.get("seed"),
        "f1": (rec.get("detection") or {}).get("f1"),
        "include_R_in_T": rec.get("include_R_in_T"),
        "late_compromise": late,
        "trust_discrimination": disc,
        "trust_recovery": recovery,
    }


def pair_delta_f1(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Match FedAvg vs trust arms on same seed for ΔF1."""
    by_seed: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        seed = r.get("seed")
        by_seed.setdefault(seed, {})
        approach = r.get("approach") or r.get("run_id") or ""
        key = "fedavg" if "fedavg" in str(approach).lower() or "b0" in str(r.get("run_id", "")).lower() else "trust"
        # Prefer explicit fedavg
        if "fedavg" in str(r.get("run_id", "")).lower():
            by_seed[seed]["fedavg"] = r
        elif key == "trust":
            by_seed[seed].setdefault("trust_candidates", []).append(r)

    out = []
    for seed, block in by_seed.items():
        fed = block.get("fedavg")
        for trust in block.get("trust_candidates", []):
            out.append({
                "seed": seed,
                "trust_run_id": trust.get("run_id"),
                "fedavg_run_id": (fed or {}).get("run_id"),
                "f1_trust": (trust.get("detection") or {}).get("f1"),
                "f1_fedavg": (fed.get("detection") or {}).get("f1") if fed else None,
                "delta_f1": delta_f1_robust(
                    (trust.get("detection") or {}).get("f1"),
                    (fed.get("detection") or {}).get("f1") if fed else None,
                ),
            })
    return out


def gate_check(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight gate signals for the natural redesign."""
    aurocs = [
        s["trust_discrimination"].get("auroc")
        for s in summaries
        if s.get("trust_discrimination", {}).get("available")
        and s["trust_discrimination"].get("auroc") is not None
    ]
    fd = [
        s["trust_discrimination"].get("false_distrust_rate")
        for s in summaries
        if s.get("trust_discrimination", {}).get("available")
    ]
    return {
        "n_with_auroc": len(aurocs),
        "mean_auroc": float(sum(aurocs) / len(aurocs)) if aurocs else None,
        "auroc_gt_0_5": all(a > 0.5 for a in aurocs) if aurocs else None,
        "mean_false_distrust_rate": float(sum(fd) / len(fd)) if fd else None,
        "note": "Benign F1 parity vs FedAvg requires paired runs (see delta_f1).",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze iomt_natural trust discrimination")
    p.add_argument("--input", type=Path, default=ROOT / "results" / "trustfed_agent" / "metrics")
    p.add_argument("--dataset", default="iomt_natural")
    p.add_argument("--export", type=Path, default=None)
    args = p.parse_args()

    if not args.input.exists():
        print(f"No metrics dir: {args.input}")
        return 1

    runs = load_runs(args.input, dataset=args.dataset)
    if not runs:
        print(f"No {args.dataset} runs under {args.input}")
        return 1

    summaries = [analyze_one(r) for r in runs]
    deltas = pair_delta_f1(runs)
    gate = gate_check(summaries)

    print(f"\n=== iomt_natural trust analysis ({len(summaries)} runs) ===\n")
    for s in summaries:
        d = s["trust_discrimination"]
        print(
            f"{s['run_id']} seed={s['seed']} F1={s['f1']} "
            f"AUROC={d.get('auroc')} AUPRC={d.get('auprc')} "
            f"delay={d.get('detection_delay')} "
            f"false_distrust={d.get('false_distrust_ids')} "
            f"α_mal={d.get('mean_alpha_malicious')} α_ben={d.get('mean_alpha_benign')}"
            f"{(' recovery_Δ='+str((s.get('trust_recovery') or {}).get('recovery_delta'))) if (s.get('trust_recovery') or {}).get('available') else ''}"
        )
    if deltas:
        print("\nΔF1 (Trust − FedAvg):")
        for row in deltas:
            print(
                f"  seed={row['seed']} {row['trust_run_id']}: "
                f"ΔF1={row['delta_f1']} (trust={row['f1_trust']} fedavg={row['f1_fedavg']})"
            )
    print(f"\nGate: {gate}")

    payload = {"summaries": summaries, "delta_f1": deltas, "gate": gate}
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
