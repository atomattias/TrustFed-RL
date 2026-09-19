#!/usr/bin/env python3
"""Path AGG confirmatory gate: B0 / B2 / Rm under frozen label_flip (seeds 43-47)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"

ARM_GLOB = {
    "B0": "run_trustfed_fedavg_iomt_natural_poison25_path_agg_seed_{seed}.json",
    "B2": "run_trustfed_b2_uniform_iomt_natural_poison25_path_agg_seed_{seed}.json",
    "Rm": "run_trustfed_rm_iomt_natural_poison25_path_agg_seed_{seed}.json",
}


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _stats(vals: List[float]) -> Dict[str, Any]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "values": [float(x) for x in a],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="43,44,45,46,47")
    ap.add_argument("--suffix", default="path_agg")
    ap.add_argument("--delta-alpha-min", type=float, default=0.02)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    per_seed: Dict[str, Any] = {}
    alpha: Dict[str, List[float]] = {"B0": [], "B2": [], "Rm": []}
    f1: Dict[str, List[float]] = {"B0": [], "B2": [], "Rm": []}
    missing: List[str] = []

    for seed in seeds:
        row: Dict[str, Any] = {"seed": seed}
        for arm, tmpl in ARM_GLOB.items():
            path = METRICS / tmpl.format(seed=seed)
            d = _load(path)
            if d is None:
                missing.append(path.name)
                row[arm] = {"missing": True}
                continue
            disc = d.get("trust_discrimination") or {}
            det = d.get("detection") or {}
            poison = (d.get("late_compromise") or {}).get("poison_mode")
            a_mal = disc.get("mean_alpha_malicious")
            f1v = det.get("f1")
            row[arm] = {
                "run_id": d.get("run_id"),
                "poison_mode": poison,
                "mean_alpha_malicious": a_mal,
                "f1": f1v,
                "auroc": disc.get("auroc"),
            }
            if poison != "label_flip":
                missing.append(f"{path.name}:poison={poison}")
            if a_mal is not None:
                alpha[arm].append(float(a_mal))
            if f1v is not None:
                f1[arm].append(float(f1v))
        if "B0" in row and "B2" in row and not row["B0"].get("missing") and not row["B2"].get("missing"):
            a0 = row["B0"].get("mean_alpha_malicious")
            a2 = row["B2"].get("mean_alpha_malicious")
            if a0 is not None and a2 is not None:
                row["delta_alpha_B0_minus_B2"] = float(a0) - float(a2)
                row["B2_alpha_lt_B0"] = float(a2) < float(a0)
        per_seed[str(seed)] = row

    mean_a = {k: _stats(v) for k, v in alpha.items()}
    mean_f = {k: _stats(v) for k, v in f1.items()}
    deltas = [
        per_seed[str(s)].get("delta_alpha_B0_minus_B2")
        for s in seeds
        if per_seed.get(str(s), {}).get("delta_alpha_B0_minus_B2") is not None
    ]
    n_b2_lt = sum(
        1
        for s in seeds
        if per_seed.get(str(s), {}).get("B2_alpha_lt_B0") is True
    )
    delta_stats = _stats([float(x) for x in deltas])
    p1 = bool(
        (delta_stats["mean"] is not None and delta_stats["mean"] >= args.delta_alpha_min)
        or n_b2_lt >= 4
    )
    f0 = mean_f["B0"]["mean"]
    f2 = mean_f["B2"]["mean"]
    if f0 is None or f2 is None:
        p2_status = "MISSING"
    elif f2 + 1e-6 >= f0:
        # ceiling if both near 0.98
        if f0 >= 0.97 and abs(f2 - f0) < 1e-3:
            p2_status = "PASS_CEILING"
        else:
            p2_status = "PASS"
    else:
        p2_status = "FAIL"

    gate = {
        "path": "AGG",
        "mode": "label_flip",
        "fraction": 0.25,
        "suffix": args.suffix,
        "seeds": seeds,
        "missing": missing,
        "per_seed": per_seed,
        "mean_alpha_malicious": mean_a,
        "mean_f1": mean_f,
        "delta_alpha_B0_minus_B2": delta_stats,
        "n_seeds_B2_alpha_lt_B0": n_b2_lt,
        "gates": {
            "P1_meaningful_alpha": {
                "pass": p1,
                "rule": f"mean_delta>={args.delta_alpha_min} or B2<B0 on >=4/5 seeds",
                "mean_delta": delta_stats["mean"],
                "n_B2_lt_B0": n_b2_lt,
            },
            "P2_f1": {"status": p2_status, "mean_B0": f0, "mean_B2": f2},
            "P3_Rm_context": {
                "mean_f1_Rm": mean_f["Rm"]["mean"],
                "mean_alpha_Rm": mean_a["Rm"]["mean"],
                "note": "Descriptive only; B2 need not beat Rm",
            },
        },
        "overall_pass": bool(p1) and p2_status in ("PASS", "PASS_CEILING") and not missing,
        "disclaimer": "Path AGG aggregation evidence under label_flip; not H1/Path1 comm_skip discrimination.",
    }
    REV.mkdir(parents=True, exist_ok=True)
    out = REV / "path_agg_label_flip_f0.25_gate.json"
    out.write_text(json.dumps(gate, indent=2))
    print(json.dumps({k: gate[k] for k in ("overall_pass", "gates", "missing", "mean_alpha_malicious", "mean_f1", "delta_alpha_B0_minus_B2")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
