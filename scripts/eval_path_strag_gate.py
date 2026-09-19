#!/usr/bin/env python3
"""Path STRAG confirmatory gate: B1 / Bc / B2 under benign straggler + comm_skip (seeds 43–47).

Pre-registered (PATH_STRAGGLER_PLAN.md):
  G1  mean(AUROC_Bc - AUROC_B1) >= 0.05 OR Bc>B1 on >=4/5
  G2a mean(AUROC_B2 - AUROC_Bc) >= 0.05 OR B2>Bc on >=4/5
  G2b abs(mean(AUROC_B2 - AUROC_Bc)) < 0.05

Primary story = exactly one of G2a / G2b. G2b is an honest null, not Path 1 failure.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"
HARD = ROOT / "config" / "iomt_natural_adversary_hard.json"

ATT: Set[str] = {"client_01", "client_06", "client_12"}
STRAG: Set[str] = {"client_09", "client_10", "client_11"}
EXPECTED_W = {
    "B1": {"V": 1.0, "S": 0.0, "D": 0.0, "U": 0.0, "C": 0.0, "R": 0.0},
    "Bc": {"V": 0.0, "S": 0.0, "D": 0.0, "U": 0.0, "C": 1.0, "R": 0.0},
    "B2": {"V": 0.2, "S": 0.2, "D": 0.2, "U": 0.2, "C": 0.2, "R": 0.0},
}
ARM_GLOB = {
    "B1": "run_trustfed_b1v_uniform_iomt_natural_poison25_comm_skip_strag_path_strag_seed_{seed}.json",
    "Bc": "run_trustfed_bc_uniform_iomt_natural_poison25_comm_skip_strag_path_strag_seed_{seed}.json",
    "B2": "run_trustfed_b2_uniform_iomt_natural_poison25_comm_skip_strag_path_strag_seed_{seed}.json",
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


def _w_ok(got: dict | None, exp: dict) -> bool:
    if not got:
        return False
    return all(abs(float(got.get(k, -1)) - float(v)) < 1e-9 for k, v in exp.items())


def _audit_file(arm: str, seed: int, q_expected: float) -> Dict[str, Any]:
    path = METRICS / ARM_GLOB[arm].format(seed=seed)
    rec: Dict[str, Any] = {
        "path": path.name,
        "missing": False,
        "issues": [],
        "auroc": None,
        "auprc": None,
        "f1": None,
        "mean_alpha_malicious": None,
    }
    d = _load(path)
    if d is None:
        rec["missing"] = True
        rec["issues"].append("missing")
        return rec

    lc = d.get("late_compromise") or {}
    bs = lc.get("benign_straggler") or {}
    disc = d.get("trust_discrimination") or {}
    det = d.get("detection") or {}
    att = set(lc.get("attacker_client_ids") or [])
    strag = set(bs.get("straggler_client_ids") or [])
    disc_att = set(disc.get("attacker_ids") or [])
    poison = lc.get("poison_mode")
    q = bs.get("skip_prob_q")
    rid = str(d.get("run_id") or "")

    rec.update(
        {
            "run_id": rid,
            "poison_mode": poison,
            "skip_prob_q": q,
            "auroc": disc.get("auroc"),
            "auprc": disc.get("auprc"),
            "f1": det.get("f1"),
            "mean_alpha_malicious": disc.get("mean_alpha_malicious"),
            "attacker_ids": sorted(att),
            "straggler_ids": sorted(strag),
            "disc_attacker_ids": sorted(disc_att),
        }
    )

    if d.get("num_rounds") != 30:
        rec["issues"].append("rounds")
    if lc.get("compromise_round") != 20:
        rec["issues"].append("ta")
    if poison != "comm_skip":
        rec["issues"].append(f"poison={poison}")
    if q is None or abs(float(q) - float(q_expected)) > 1e-12 or not bs.get("enabled"):
        rec["issues"].append(f"q={q}")
    if att != ATT:
        rec["issues"].append(f"att={sorted(att)}")
    if strag != STRAG:
        rec["issues"].append(f"strag={sorted(strag)}")
    if not att.isdisjoint(strag):
        rec["issues"].append("overlap")
    if disc_att != att:
        rec["issues"].append("disc_att")
    if strag & disc_att:
        rec["issues"].append("strag_in_positives")
    if not _w_ok(d.get("signal_weights"), EXPECTED_W[arm]):
        rec["issues"].append("weights")
    if "path_strag" not in rid or "explore" in rid or "smoke" in rid:
        rec["issues"].append("run_id")
    if "path_strag" not in path.name or "explore" in path.name or "smoke" in path.name:
        rec["issues"].append("filename")

    logs = d.get("round_logs") or []
    post = [r for r in logs if int(r.get("round", 0)) >= 20]
    reasons: Counter = Counter()
    bad_att = 0
    benign = 0
    miss_f = 0
    for r in post:
        for cid, row in (r.get("client_trust") or {}).items():
            if "skip_reason" not in row or "is_straggler" not in row:
                miss_f += 1
            reason = row.get("skip_reason")
            reasons[reason] += 1
            if cid in att and (
                reason != "attacker_comm_skip" or float(row.get("alpha") or 0) != 0
            ):
                bad_att += 1
            if reason == "benign_straggler":
                benign += 1
                if (
                    cid not in strag
                    or row.get("is_attacker")
                    or float(row.get("alpha") or 0) != 0
                ):
                    rec["issues"].append(f"bad_benign:{cid}:r{r.get('round')}")
    exp_att = len(att) * len(post)
    rec["skip_audit"] = {
        "post_ta": len(post),
        "benign_skips": benign,
        "bad_att": bad_att,
        "missing_fields": miss_f,
        "attacker_skips": reasons.get("attacker_comm_skip", 0),
        "expected_attacker_skips": exp_att,
    }
    if not post:
        rec["issues"].append("no_post")
    if bad_att:
        rec["issues"].append(f"bad_att={bad_att}")
    if benign < 1:
        rec["issues"].append("no_benign")
    if miss_f:
        rec["issues"].append(f"miss_f={miss_f}")
    if reasons.get("attacker_comm_skip", 0) != exp_att:
        rec["issues"].append(
            f"att_skip {reasons.get('attacker_comm_skip')}!={exp_att}"
        )
    rec["pass"] = len(rec["issues"]) == 0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="43,44,45,46,47")
    ap.add_argument("--suffix", default="path_strag")
    ap.add_argument("--q", type=float, default=0.5)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.suffix != "path_strag":
        raise SystemExit(f"refusing suffix={args.suffix!r}; locked to path_strag")
    if abs(float(args.q) - 0.5) > 1e-12:
        raise SystemExit(f"refusing q={args.q}; locked to 0.5")
    if 42 in seeds:
        raise SystemExit("refusing seed 42 in confirmatory gate (explore only)")

    audits: Dict[str, Dict[str, Any]] = {}
    per_seed: Dict[str, Any] = {}
    auroc: Dict[str, List[float]] = {"B1": [], "Bc": [], "B2": []}
    auprc: Dict[str, List[float]] = {"B1": [], "Bc": [], "B2": []}
    f1: Dict[str, List[float]] = {"B1": [], "Bc": [], "B2": []}
    alpha_mal: Dict[str, List[float]] = {"B1": [], "Bc": [], "B2": []}
    audit_issues: List[str] = []

    for seed in seeds:
        row: Dict[str, Any] = {"seed": seed}
        for arm in ("B1", "Bc", "B2"):
            a = _audit_file(arm, seed, args.q)
            audits[f"{arm}:{seed}"] = a
            if not a.get("pass"):
                audit_issues.append(f"{arm} s{seed}: {a.get('issues')}")
            arm_row = {
                "run_id": a.get("run_id"),
                "poison_mode": a.get("poison_mode"),
                "skip_prob_q": a.get("skip_prob_q"),
                "auroc": a.get("auroc"),
                "auprc": a.get("auprc"),
                "f1": a.get("f1"),
                "mean_alpha_malicious": a.get("mean_alpha_malicious"),
                "attacker_ids": a.get("attacker_ids"),
                "straggler_ids": a.get("straggler_ids"),
                "skip_audit": a.get("skip_audit"),
                "audit_pass": a.get("pass"),
                "audit_issues": a.get("issues"),
            }
            row[arm] = arm_row
            if a.get("auroc") is not None:
                auroc[arm].append(float(a["auroc"]))
            if a.get("auprc") is not None:
                auprc[arm].append(float(a["auprc"]))
            if a.get("f1") is not None:
                f1[arm].append(float(a["f1"]))
            if a.get("mean_alpha_malicious") is not None:
                alpha_mal[arm].append(float(a["mean_alpha_malicious"]))

        if all(row.get(a, {}).get("auroc") is not None for a in ("B1", "Bc", "B2")):
            a1 = float(row["B1"]["auroc"])
            ac = float(row["Bc"]["auroc"])
            a2 = float(row["B2"]["auroc"])
            row["delta_Bc_minus_B1"] = ac - a1
            row["delta_B2_minus_Bc"] = a2 - ac
            row["Bc_gt_B1"] = ac > a1
            row["B2_gt_Bc"] = a2 > ac
        per_seed[str(seed)] = row

    mean_au = {k: _stats(v) for k, v in auroc.items()}
    mean_ap = {k: _stats(v) for k, v in auprc.items()}
    mean_f = {k: _stats(v) for k, v in f1.items()}
    mean_am = {k: _stats(v) for k, v in alpha_mal.items()}

    d_bc_b1 = [
        per_seed[str(s)]["delta_Bc_minus_B1"]
        for s in seeds
        if per_seed.get(str(s), {}).get("delta_Bc_minus_B1") is not None
    ]
    d_b2_bc = [
        per_seed[str(s)]["delta_B2_minus_Bc"]
        for s in seeds
        if per_seed.get(str(s), {}).get("delta_B2_minus_Bc") is not None
    ]
    n_bc_gt = sum(1 for s in seeds if per_seed.get(str(s), {}).get("Bc_gt_B1") is True)
    n_b2_gt = sum(1 for s in seeds if per_seed.get(str(s), {}).get("B2_gt_Bc") is True)

    st_bc_b1 = _stats([float(x) for x in d_bc_b1])
    st_b2_bc = _stats([float(x) for x in d_b2_bc])

    g1 = bool(
        (st_bc_b1["mean"] is not None and st_bc_b1["mean"] >= 0.05) or n_bc_gt >= 4
    )
    g2a = bool(
        (st_b2_bc["mean"] is not None and st_b2_bc["mean"] >= 0.05) or n_b2_gt >= 4
    )
    g2b = bool(st_b2_bc["mean"] is not None and abs(st_b2_bc["mean"]) < 0.05)

    if g2a and not g2b:
        primary = "G2a"
        narrative = (
            "Multi-signal helps under confounding (B2 > Bc); "
            "directional claim only — not aggregation defence"
        )
    elif g2b and not g2a:
        primary = "G2b"
        narrative = (
            "C still dominates under this benign-straggler schedule (B2 ≈ Bc); "
            "document null — not Path 1 failure; not multi-signal necessity"
        )
    elif g2a and g2b:
        primary = None
        narrative = "Ambiguous G2a∩G2b — refuse exclusive primary; recheck thresholds"
    else:
        primary = None
        narrative = "Neither G2a nor G2b; mixed B2 vs Bc — do not claim multi-signal necessity"

    hard = _load(HARD) or {}
    hard_ok = (
        hard.get("status") == "frozen_primary_hard_attack"
        and "benign_straggler" not in hard
    )
    h1_ok = (
        METRICS / "run_trustfed_b2_uniform_iomt_natural_poison25_comm_skip_seed_43.json"
    ).exists()
    path1_ok = bool(list(METRICS.glob("run_*path1_c_only*seed_43.json")))
    path_agg_ok = bool(list(METRICS.glob("run_*path_agg*seed_43.json"))) or bool(
        list(METRICS.glob("run_*label_flip*path_agg*seed_43.json"))
    )

    locks = {
        "suffix_locked": args.suffix == "path_strag",
        "q_locked": abs(float(args.q) - 0.5) < 1e-12,
        "no_seed_42": 42 not in seeds,
        "hard_untouched": hard_ok,
        "h1_seed43_present": h1_ok,
        "path1_seed43_present": path1_ok,
        "path_agg_seed43_present": path_agg_ok,
        "n_files": sum(1 for a in audits.values() if not a.get("missing")),
        "all_audits_pass": len(audit_issues) == 0,
        "audit_issues": audit_issues,
    }
    exclusive = primary in ("G2a", "G2b") and not (g2a and g2b)
    overall = bool(
        g1
        and exclusive
        and locks["all_audits_pass"]
        and locks["hard_untouched"]
        and locks["h1_seed43_present"]
        and locks["path1_seed43_present"]
        and locks["n_files"] == 15
    )

    gate = {
        "path": "STRAG",
        "mode": "comm_skip",
        "benign_skip_prob_q": args.q,
        "fraction": 0.25,
        "suffix": args.suffix,
        "seeds": seeds,
        "attacker_ids_locked": sorted(ATT),
        "straggler_ids_locked": sorted(STRAG),
        "gt_positives": "attackers_only",
        "per_seed": per_seed,
        "mean_auroc": mean_au,
        "mean_auprc": mean_ap,
        "mean_f1": mean_f,
        "mean_alpha_malicious": mean_am,
        "delta_Bc_minus_B1": st_bc_b1,
        "delta_B2_minus_Bc": st_b2_bc,
        "n_seeds_Bc_gt_B1": n_bc_gt,
        "n_seeds_B2_gt_Bc": n_b2_gt,
        "gates": {
            "G1_participation_informative": {
                "pass": g1,
                "rule": "mean(AUROC_Bc - AUROC_B1) >= 0.05 OR Bc>B1 on >=4/5 seeds",
                "mean_delta": st_bc_b1["mean"],
                "n_Bc_gt_B1": n_bc_gt,
            },
            "G2a_multisignal_helps": {
                "pass": g2a,
                "rule": "mean(AUROC_B2 - AUROC_Bc) >= 0.05 OR B2>Bc on >=4/5 seeds",
                "mean_delta": st_b2_bc["mean"],
                "n_B2_gt_Bc": n_b2_gt,
            },
            "G2b_C_still_dominates": {
                "pass": g2b,
                "rule": "abs(mean(AUROC_B2 - AUROC_Bc)) < 0.05",
                "mean_delta": st_b2_bc["mean"],
            },
        },
        "primary_story": primary,
        "exclusive_g2": exclusive,
        "narrative": narrative,
        "locks": locks,
        "overall_pass": overall,
        "claim_allowed": [
            "Directional B2 vs Bc under confounded missingness",
            "G2b null: C still ranks attackers under this q=0.5 schedule",
        ],
        "claim_forbidden": [
            "Aggregation defence",
            "Path 1 wrong",
            "Multi-hospital ECU",
            "AUROC=1.0 mandate",
            "Multi-signal necessity from this cell",
        ],
        "disclaimer": (
            "Path STRAG: malicious comm_skip plus honest intermittent misses. "
            "G2b is an honest null (C still enough), not Path 1 failure. "
            "Not aggregation defence; not multi-hospital."
        ),
    }
    REV.mkdir(parents=True, exist_ok=True)
    out = REV / "path_strag_comm_skip_f0.25_gate.json"
    out.write_text(json.dumps(gate, indent=2) + "\n")

    summary = {
        "overall_pass": overall,
        "primary_story": primary,
        "narrative": narrative,
        "gates": {k: v["pass"] for k, v in gate["gates"].items()},
        "mean_auroc": {k: mean_au[k]["mean"] for k in mean_au},
        "delta_Bc_minus_B1": st_bc_b1["mean"],
        "delta_B2_minus_Bc": st_b2_bc["mean"],
        "locks_ok": locks["all_audits_pass"] and locks["hard_untouched"],
        "audit_issue_count": len(audit_issues),
    }
    (REV / "path_strag_comm_skip_f0.25_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Draft numbers for T1.6 (red insert); not a paper claim yet
    m1 = mean_au["B1"]["mean"]
    mc = mean_au["Bc"]["mean"]
    m2 = mean_au["B2"]["mean"]
    s1 = mean_au["B1"]["std"] or 0.0
    fill = (
        "% Auto-generated Path STRAG paper fill — G2b null; not aggregation defence.\n"
        f"% Narrative: {narrative}\n"
        f"% Primary: {primary}; do not claim multi-signal necessity from this cell.\n"
        f"Under malicious \\texttt{{comm\\_skip}} ($f{{=}}0.25$, $t_a{{=}}20$) plus "
        f"honest intermittent misses ($q{{=}}0.5$ on clients 09/10/11; seeds~$43$--$47$), "
        f"participation-only trust (Bc) mean AUROC ${mc:.3f}$ versus B1 "
        f"${m1:.3f}\\pm{s1:.3f}$ "
        f"($\\Delta_{{\\mathrm{{Bc,B1}}}}{{=}}{{+}}{st_bc_b1['mean']:.3f}$; G1). "
        f"Multi-signal B2 matches Bc (${m2:.3f}$; "
        f"$\\Delta_{{\\mathrm{{B2,Bc}}}}{{=}}{st_b2_bc['mean']:.3f}$; G2b null). "
        "Ground-truth positives remain malicious attackers only; stragglers are benign. "
        "This cell does not overturn Path~1 and is not an aggregation-defence claim.\n"
    )
    (REV / "path_strag_comm_skip_f0.25_paper_fill.tex").write_text(fill)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {REV / 'path_strag_comm_skip_f0.25_summary.json'}")
    print(f"wrote {REV / 'path_strag_comm_skip_f0.25_paper_fill.tex'}")
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
