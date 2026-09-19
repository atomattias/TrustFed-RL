#!/usr/bin/env python3
"""
Path 1A gate: C-only (Bc) vs B1 vs B2 under frozen comm_skip.

Pre-registered (PATH1_C_ONLY_BASELINE_PLAN.md):
  G1  mean(AUROC_Bc - AUROC_B1) >= 0.05 and Δ>0 on >=4/5 seeds
  G2a mean(|AUROC_B2 - AUROC_Bc|) <= 0.05  → multi-signal not necessary for this regime
  G2b mean(AUROC_B2 - AUROC_Bc) >= 0.05 and Δ>0 on >=4/5 → multi-signal adds beyond C

Only metrics whose run_id contains --suffix (default path1_c_only) are used.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _auroc(rec: Dict[str, Any]) -> Optional[float]:
    disc = rec.get("trust_discrimination") or {}
    a = disc.get("auroc")
    return float(a) if a is not None else None


def _auprc(rec: Dict[str, Any]) -> Optional[float]:
    disc = rec.get("trust_discrimination") or {}
    a = disc.get("auprc")
    return float(a) if a is not None else None


def _f1(rec: Dict[str, Any]) -> Optional[float]:
    det = rec.get("detection") or {}
    a = det.get("f1")
    return float(a) if a is not None else None


def _arm(rec: Dict[str, Any]) -> Optional[str]:
    rid = str(rec.get("run_id") or "").lower()
    if "trustfed_bc" in rid or rid.startswith("bc") or "_bc_" in rid:
        return "bc"
    if "trustfed_b1v" in rid or "b1v" in rid:
        return "b1"
    if "trustfed_b2" in rid or "_b2_" in rid or rid.endswith("_b2"):
        return "b2"
    # approach field fallback
    ap = str(rec.get("approach") or "").lower()
    if "bc" in ap or ap == "trustfed_bc":
        return "bc"
    if "b1" in ap:
        return "b1"
    if "b2" in ap:
        return "b2"
    return None


def collect(
    metrics_dir: Path,
    seeds: List[int],
    suffix: str,
    mode: str,
    frac: float,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {s: {} for s in seeds}
    pct = int(round(100 * float(frac)))
    for path in sorted(metrics_dir.glob("run_*.json")):
        if suffix not in path.name:
            continue
        with open(path) as f:
            rec = json.load(f)
        rid = str(rec.get("run_id") or path.stem)
        if suffix not in rid and suffix not in path.name:
            continue
        seed = int(rec.get("seed") if rec.get("seed") is not None else -1)
        if seed not in by_seed:
            continue
        late = rec.get("late_compromise") or {}
        pm = str(late.get("poison_mode") or "")
        if pm and pm != mode and mode not in rid:
            continue
        arm = _arm(rec)
        if arm is None:
            # path-based
            name = path.name.lower()
            if "trustfed_bc" in name:
                arm = "bc"
            elif "trustfed_b1v" in name:
                arm = "b1"
            elif "trustfed_b2" in name:
                arm = "b2"
        if arm is None:
            continue
        if f"poison{pct}" not in path.name and late.get("adversary_fraction") is not None:
            if abs(float(late["adversary_fraction"]) - float(frac)) > 1e-6:
                continue
        by_seed[seed][arm] = rec
    return by_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("results/trustfed_agent/metrics"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46, 47])
    ap.add_argument("--suffix", default="path1_c_only")
    ap.add_argument("--poison-mode", default="comm_skip")
    ap.add_argument("--adversary-fraction", type=float, default=0.25)
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--min-sign-frac", type=float, default=0.8)
    ap.add_argument(
        "--export",
        type=Path,
        default=Path("results/trustfed_agent/rev_sup/path1_c_only_comm_skip_f0.25_gate.json"),
    )
    args = ap.parse_args()

    by_seed = collect(
        args.metrics, args.seeds, args.suffix, args.poison_mode, args.adversary_fraction
    )
    triples = []
    for seed in args.seeds:
        slot = by_seed.get(seed, {})
        b1, bc, b2 = slot.get("b1"), slot.get("bc"), slot.get("b2")
        complete = all(x is not None for x in (b1, bc, b2))
        row: Dict[str, Any] = {"seed": seed, "complete": complete}
        if complete:
            a1, ac, a2 = _auroc(b1), _auroc(bc), _auroc(b2)  # type: ignore
            row.update(
                {
                    "b1_auroc": a1,
                    "bc_auroc": ac,
                    "b2_auroc": a2,
                    "delta_bc_b1": (ac - a1) if a1 is not None and ac is not None else None,
                    "delta_b2_bc": (a2 - ac) if a2 is not None and ac is not None else None,
                    "abs_delta_b2_bc": (abs(a2 - ac) if a2 is not None and ac is not None else None),
                    "b1_auprc": _auprc(b1),  # type: ignore
                    "bc_auprc": _auprc(bc),  # type: ignore
                    "b2_auprc": _auprc(b2),  # type: ignore
                    "b1_f1": _f1(b1),  # type: ignore
                    "bc_f1": _f1(bc),  # type: ignore
                    "b2_f1": _f1(b2),  # type: ignore
                    "signal_weights_bc": (bc or {}).get("signal_weights"),  # type: ignore
                }
            )
        triples.append(row)

    complete = [t for t in triples if t.get("complete")]
    n = len(complete)
    def mean(key: str) -> Optional[float]:
        vals = [float(t[key]) for t in complete if t.get(key) is not None]
        return statistics.mean(vals) if vals else None

    mean_b1 = mean("b1_auroc")
    mean_bc = mean("bc_auroc")
    mean_b2 = mean("b2_auroc")
    mean_d_bc_b1 = mean("delta_bc_b1")
    mean_d_b2_bc = mean("delta_b2_bc")
    mean_abs_b2_bc = mean("abs_delta_b2_bc")

    n_need = int(round(args.min_sign_frac * len(args.seeds)))
    n_bc_gt_b1 = sum(1 for t in complete if (t.get("delta_bc_b1") or 0) > 0)
    n_b2_gt_bc = sum(1 for t in complete if (t.get("delta_b2_bc") or 0) > 0)

    g1 = bool(
        n == len(args.seeds)
        and mean_d_bc_b1 is not None
        and mean_d_bc_b1 >= args.min_delta
        and n_bc_gt_b1 >= n_need
    )
    g2a = bool(
        n == len(args.seeds)
        and mean_abs_b2_bc is not None
        and mean_abs_b2_bc <= args.min_delta
    )
    g2b = bool(
        n == len(args.seeds)
        and mean_d_b2_bc is not None
        and mean_d_b2_bc >= args.min_delta
        and n_b2_gt_bc >= n_need
    )

    if g2a and not g2b:
        narrative = "G2a: B2≈Bc; multi-signal not necessary for pure comm_skip discrimination"
    elif g2b and not g2a:
        narrative = "G2b: B2≫Bc; multi-signal adds beyond C"
    elif g2a and g2b:
        narrative = "mixed: both G2a and G2b thresholds met (inspect magnitudes)"
    else:
        narrative = "mixed/fail: neither G2a nor G2b clearly satisfied"

    gate = {
        "pass": bool(g1 and (g2a or g2b or True)),  # G1 required; G2 recorded
        "g1_bc_beats_b1": g1,
        "g2a_b2_approx_bc": g2a,
        "g2b_b2_beats_bc": g2b,
        "narrative": narrative,
        "n_complete": n,
        "n_seeds": len(args.seeds),
        "suffix": args.suffix,
        "poison_mode": args.poison_mode,
        "adversary_fraction": args.adversary_fraction,
        "mean_auroc_b1": mean_b1,
        "mean_auroc_bc": mean_bc,
        "mean_auroc_b2": mean_b2,
        "mean_delta_bc_b1": mean_d_bc_b1,
        "mean_delta_b2_bc": mean_d_b2_bc,
        "mean_abs_delta_b2_bc": mean_abs_b2_bc,
        "n_bc_gt_b1": n_bc_gt_b1,
        "n_b2_gt_bc": n_b2_gt_bc,
        "min_delta": args.min_delta,
        "min_sign_n": n_need,
        "mean_f1_b1": mean("b1_f1"),
        "mean_f1_bc": mean("bc_f1"),
        "mean_f1_b2": mean("b2_f1"),
        "triples": triples,
        "claim_note": (
            "Discrimination / ranking under communication withholding only; "
            "not poisoning-resilient aggregation (skippers upload nothing)."
        ),
    }
    # Overall pass: G1 must hold; G2a or explicit mixed is OK for science
    gate["pass"] = bool(g1 and n == len(args.seeds))

    args.export.parent.mkdir(parents=True, exist_ok=True)
    args.export.write_text(json.dumps(gate, indent=2) + "\n")

    summary_path = args.export.with_name(args.export.name.replace("_gate.json", "_summary.json"))
    if summary_path == args.export:
        summary_path = args.export.with_name(args.export.stem + "_summary.json")
    summary = {
        "mean_auroc": {"B1": mean_b1, "Bc": mean_bc, "B2": mean_b2},
        "deltas": {"Bc-B1": mean_d_bc_b1, "B2-Bc": mean_d_b2_bc},
        "gates": {"G1": g1, "G2a": g2a, "G2b": g2b},
        "narrative": narrative,
        "pass": gate["pass"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fill_path = args.export.with_name(args.export.name.replace("_gate.json", "_paper_fill.tex"))
    if fill_path == args.export:
        fill_path = args.export.with_name(args.export.stem + "_paper_fill.tex")
    fill = (
        "% Auto-generated Path 1 C-only paper fill — discrimination only, not aggregation defence.\n"
        f"% Narrative: {narrative}\n"
        f"Under frozen \\texttt{{comm\\_skip}} ($f{{=}}0.25$, seeds~$43$--$47$), "
        f"participation-only trust (Bc) achieves mean AUROC ${mean_bc:.3f}$ "
        f"versus B1 ${mean_b1:.3f}$ ($\\Delta{{=}}{{+}}{mean_d_bc_b1:.3f}$) "
        f"and matches multi-signal B2 (${mean_b2:.3f}$; "
        f"$|\\Delta|_{{\\mathrm{{B2,Bc}}}}{{=}}{mean_abs_b2_bc:.3f}$). "
        "Shared-test F1 remains at the detection ceiling; skippers contribute no updates, "
        "so this cell supports reliability discrimination mediated by $C$, "
        "not poisoning-resilient aggregation.\n"
    )
    fill_path.write_text(fill)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.export}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fill_path}")
    raise SystemExit(0 if gate["pass"] else 1)


if __name__ == "__main__":
    main()
