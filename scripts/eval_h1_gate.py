#!/usr/bin/env python3
"""
Evaluate H1 gate (supervisor plan v3): multi-signal B2 vs validation-only B1.

Pass (confirmatory) if all hold on the provided seeds:
  - mean AUROC_B1 < 1.0
  - mean(AUROC_B2 - AUROC_B1) >= min_delta (default 0.05)
  - Δ > 0 on at least min_sign_frac of seeds (default 0.8 → 4/5)

Usage:
  python scripts/eval_h1_gate.py \\
    --metrics results/trustfed_agent/metrics \\
    --seeds 43 44 45 46 47 \\
    --poison-mode sign_flip \\
    --adversary-fraction 0.25 \\
    --export results/trustfed_agent/rev_sup/wp_a_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import compute_natural_trust_discrimination  # noqa: E402


def _auroc(rec: Dict[str, Any]) -> Optional[float]:
    disc = rec.get("trust_discrimination")
    if not disc or not disc.get("available"):
        late = rec.get("late_compromise") or {}
        disc = compute_natural_trust_discrimination(
            rec.get("round_logs") or [],
            compromise_round=int(late.get("compromise_round") or 20),
            attacker_ids=late.get("attacker_client_ids") or [],
        )
    a = disc.get("auroc") if disc else None
    return float(a) if a is not None else None


def _match_mode(rec: Dict[str, Any], mode: str, frac: float) -> bool:
    late = rec.get("late_compromise") or {}
    if not late.get("enabled", True):
        return False
    pm = str(late.get("poison_mode") or "label_flip")
    if pm != mode:
        # also accept run_id tag
        rid = str(rec.get("run_id") or "")
        if mode not in rid:
            return False
    af = late.get("adversary_fraction")
    if af is not None and abs(float(af) - float(frac)) > 1e-6:
        # fraction may be encoded only in run_id poison25
        rid = str(rec.get("run_id") or "")
        pct = int(round(100 * float(frac)))
        if f"poison{pct}" not in rid and f"_{pct}" not in rid:
            return False
    return True


def _is_b1(rec: Dict[str, Any]) -> bool:
    rid = str(rec.get("run_id") or "")
    ap = str(rec.get("approach") or "")
    return "b1v" in rid.lower() or "b1" in rid.lower() or ap in ("b1", "trustfed_b1v")


def _is_b2(rec: Dict[str, Any]) -> bool:
    rid = str(rec.get("run_id") or "")
    ap = str(rec.get("approach") or "")
    return "b2" in rid.lower() or ap in ("b2", "trustfed_b2")


def collect_pairs(
    metrics_dir: Path,
    seeds: List[int],
    mode: str,
    frac: float,
    dataset: str = "iomt_natural",
) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(metrics_dir.glob("run_*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("dataset") not in (None, dataset) and dataset not in path.name:
            continue
        seed = int(rec.get("seed") if rec.get("seed") is not None else -1)
        if seed not in seeds:
            continue
        if not _match_mode(rec, mode, frac):
            continue
        rows.append(rec)

    by_seed: Dict[int, Dict[str, Any]] = {}
    for rec in rows:
        seed = int(rec["seed"])
        slot = by_seed.setdefault(seed, {})
        if _is_b1(rec):
            slot["b1"] = rec
        elif _is_b2(rec):
            slot["b2"] = rec
    pairs = []
    for seed in seeds:
        slot = by_seed.get(seed, {})
        b1, b2 = slot.get("b1"), slot.get("b2")
        if not b1 or not b2:
            pairs.append({"seed": seed, "complete": False, "b1_auroc": None, "b2_auroc": None, "delta": None})
            continue
        a1, a2 = _auroc(b1), _auroc(b2)
        delta = (a2 - a1) if a1 is not None and a2 is not None else None
        pairs.append({
            "seed": seed,
            "complete": True,
            "b1_run_id": b1.get("run_id"),
            "b2_run_id": b2.get("run_id"),
            "b1_auroc": a1,
            "b2_auroc": a2,
            "delta": delta,
            "b1_f1": (b1.get("detection") or {}).get("f1"),
            "b2_f1": (b2.get("detection") or {}).get("f1"),
        })
    return pairs


def evaluate(
    pairs: List[Dict[str, Any]],
    min_delta: float = 0.05,
    min_sign_frac: float = 0.8,
) -> Dict[str, Any]:
    complete = [p for p in pairs if p.get("complete") and p.get("delta") is not None]
    if not complete:
        return {"pass": False, "reason": "no_complete_pairs", "pairs": pairs}

    b1_mean = sum(p["b1_auroc"] for p in complete) / len(complete)
    b2_mean = sum(p["b2_auroc"] for p in complete) / len(complete)
    delta_mean = sum(p["delta"] for p in complete) / len(complete)
    n_pos = sum(1 for p in complete if p["delta"] > 0)
    sign_ok = (n_pos / len(complete)) >= min_sign_frac

    checks = {
        "b1_not_perfect": b1_mean < 1.0 - 1e-12,
        "delta_ge_min": delta_mean >= min_delta,
        "sign_consistency": sign_ok,
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "n_complete": len(complete),
        "mean_auroc_b1": b1_mean,
        "mean_auroc_b2": b2_mean,
        "mean_delta": delta_mean,
        "n_delta_positive": n_pos,
        "min_delta": min_delta,
        "min_sign_frac": min_sign_frac,
        "checks": checks,
        "pairs": pairs,
        "reason": "ok" if passed else "failed_checks",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="H1 gate B2 vs B1 (plan v3)")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "trustfed_agent" / "metrics")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--poison-mode", default="sign_flip")
    ap.add_argument("--adversary-fraction", type=float, default=0.25)
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--min-sign-frac", type=float, default=0.8)
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--role", choices=["exploration", "confirmatory"], default="confirmatory")
    args = ap.parse_args()

    pairs = collect_pairs(args.metrics, args.seeds, args.poison_mode, args.adversary_fraction)
    result = evaluate(pairs, min_delta=args.min_delta, min_sign_frac=args.min_sign_frac)
    result["role"] = args.role
    result["poison_mode"] = args.poison_mode
    result["adversary_fraction"] = args.adversary_fraction
    result["seeds"] = args.seeds

    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
    print("\nPer-seed:")
    for p in result["pairs"]:
        print(
            f"  seed={p['seed']} complete={p['complete']} "
            f"B1={p['b1_auroc']} B2={p['b2_auroc']} Δ={p['delta']}"
        )
    print(f"\nH1 pass={result['pass']} ({result['reason']})")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.export}")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
