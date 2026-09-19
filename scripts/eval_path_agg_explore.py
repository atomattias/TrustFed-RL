#!/usr/bin/env python3
"""Summarize Path AGG seed-42 explore metrics (B0/B2/Rm)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"


def _load(pattern: str) -> dict | None:
    hits = sorted(METRICS.glob(pattern))
    if not hits:
        return None
    return json.loads(hits[-1].read_text())


def row(d: dict | None) -> dict:
    if not d:
        return {"missing": True}
    disc = d.get("trust_discrimination") or {}
    det = d.get("detection") or {}
    return {
        "approach": d.get("approach"),
        "run_id": d.get("run_id"),
        "poison_mode": (d.get("late_compromise") or {}).get("poison_mode"),
        "f1": det.get("f1"),
        "accuracy": det.get("accuracy"),
        "mean_alpha_malicious": disc.get("mean_alpha_malicious"),
        "mean_alpha_benign": disc.get("mean_alpha_benign"),
        "auroc": disc.get("auroc"),
        "auprc": disc.get("auprc"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="path_agg_explore")
    ap.add_argument("--mode", default="sign_flip")
    args = ap.parse_args()
    tag = f"poison25_{args.mode}_{args.suffix}"
    arms = {
        "B0": _load(f"run_trustfed_fedavg_iomt_natural_{tag}_seed_{args.seed}.json"),
        "B2": _load(f"run_trustfed_b2_uniform_iomt_natural_{tag}_seed_{args.seed}.json"),
        "Rm": _load(f"run_trustfed_rm_iomt_natural_{tag}_seed_{args.seed}.json"),
    }
    summary = {k: row(v) for k, v in arms.items()}
    b0, b2, rm = summary["B0"], summary["B2"], summary["Rm"]

    def f(x, key):
        return None if not x or x.get("missing") else x.get(key)

    a0, a2 = f(b0, "mean_alpha_malicious"), f(b2, "mean_alpha_malicious")
    f0, f2, fm = f(b0, "f1"), f(b2, "f1"), f(rm, "f1")
    p1 = None if a0 is None or a2 is None else (a2 < a0 - 1e-6)
    p2 = None if f0 is None or f2 is None else (f2 + 1e-6 >= f0)
    p3_note = {
        "f1_B2_vs_Rm": None if f2 is None or fm is None else f2 - fm,
        "f1_Rm_vs_B0": None if fm is None or f0 is None else fm - f0,
    }
    decision = {
        "P1_B2_alpha_lt_B0": p1,
        "P2_B2_f1_ge_B0": p2,
        "P3_context": p3_note,
        "freeze_candidate": bool(p1) or (
            f0 is not None and fm is not None and abs(fm - f0) > 0.01
        ) or (
            f2 is not None and fm is not None and abs(f2 - fm) > 0.01
        ),
    }
    out = {
        "seed": args.seed,
        "suffix": args.suffix,
        "mode": args.mode,
        "arms": summary,
        "decision": decision,
    }
    REV.mkdir(parents=True, exist_ok=True)
    path = REV / f"path_agg_explore_{args.mode}_f0.25_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
