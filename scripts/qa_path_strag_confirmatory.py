#!/usr/bin/env python3
"""QA-audit Path STRAG confirmatory metrics (T1.4): 15 files, skips, weights, locks."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"
HARD = ROOT / "config" / "iomt_natural_adversary_hard.json"

EXPECTED_W = {
    "B1": {"V": 1.0, "S": 0.0, "D": 0.0, "U": 0.0, "C": 0.0, "R": 0.0},
    "Bc": {"V": 0.0, "S": 0.0, "D": 0.0, "U": 0.0, "C": 1.0, "R": 0.0},
    "B2": {"V": 0.2, "S": 0.2, "D": 0.2, "U": 0.2, "C": 0.2, "R": 0.0},
}
PAT = {
    "B1": "run_trustfed_b1v_uniform_iomt_natural_poison25_comm_skip_strag_path_strag_seed_{s}.json",
    "Bc": "run_trustfed_bc_uniform_iomt_natural_poison25_comm_skip_strag_path_strag_seed_{s}.json",
    "B2": "run_trustfed_b2_uniform_iomt_natural_poison25_comm_skip_strag_path_strag_seed_{s}.json",
}
ATT = {"client_01", "client_06", "client_12"}
STRAG = {"client_09", "client_10", "client_11"}


def w_ok(got: dict | None, exp: dict) -> bool:
    if not got:
        return False
    return all(abs(float(got.get(k, -1)) - float(v)) < 1e-9 for k, v in exp.items())


def audit_one(arm: str, seed: int) -> dict:
    p = METRICS / PAT[arm].format(s=seed)
    rec: dict = {"path": p.name, "pass": False, "issues": []}
    if not p.exists():
        rec["issues"].append("missing")
        return rec
    d = json.loads(p.read_text())
    lc = d.get("late_compromise") or {}
    bs = lc.get("benign_straggler") or {}
    disc = d.get("trust_discrimination") or {}
    det = d.get("detection") or {}
    att = set(lc.get("attacker_client_ids") or [])
    strag = set(bs.get("straggler_client_ids") or [])
    disc_att = set(disc.get("attacker_ids") or [])
    rec.update(
        {
            "auroc": disc.get("auroc"),
            "auprc": disc.get("auprc"),
            "f1": det.get("f1"),
            "mean_alpha_mal": disc.get("mean_alpha_malicious"),
            "q": bs.get("skip_prob_q"),
            "run_id": d.get("run_id"),
        }
    )
    if d.get("num_rounds") != 30:
        rec["issues"].append("rounds")
    if lc.get("compromise_round") != 20:
        rec["issues"].append("ta")
    if lc.get("poison_mode") != "comm_skip":
        rec["issues"].append("mode")
    if bs.get("skip_prob_q") != 0.5 or not bs.get("enabled"):
        rec["issues"].append("q")
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
    if not w_ok(d.get("signal_weights"), EXPECTED_W[arm]):
        rec["issues"].append("weights")
    rid = d.get("run_id") or ""
    if "path_strag" not in rid or "explore" in rid or "smoke" in rid:
        rec["issues"].append("run_id")

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
                    rec["issues"].append(f"bad_benign {cid} r{r.get('round')}")
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
            f"att_skip_count {reasons.get('attacker_comm_skip')}!={exp_att}"
        )
    rec["pass"] = len(rec["issues"]) == 0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="43,44,45,46,47")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    arms = {arm: {s: audit_one(arm, s) for s in seeds} for arm in PAT}
    issues = [
        f"{arm} s{s}: {arms[arm][s]['issues']}"
        for arm in arms
        for s in seeds
        if not arms[arm][s].get("pass")
    ]
    means = {
        arm: sum(float(arms[arm][s]["auroc"]) for s in seeds) / len(seeds)
        for arm in arms
    }
    deltas = {
        "Bc_minus_B1": means["Bc"] - means["B1"],
        "B2_minus_Bc": means["B2"] - means["Bc"],
    }
    g1 = deltas["Bc_minus_B1"] >= 0.05 or (
        sum(1 for s in seeds if arms["Bc"][s]["auroc"] > arms["B1"][s]["auroc"]) >= 4
    )
    g2a = deltas["B2_minus_Bc"] >= 0.05 or (
        sum(1 for s in seeds if arms["B2"][s]["auroc"] > arms["Bc"][s]["auroc"]) >= 4
    )
    g2b = abs(deltas["B2_minus_Bc"]) < 0.05

    hard = json.loads(HARD.read_text())
    hard_ok = (
        hard.get("status") == "frozen_primary_hard_attack"
        and "benign_straggler" not in hard
    )
    h1 = (
        METRICS / "run_trustfed_b2_uniform_iomt_natural_poison25_comm_skip_seed_43.json"
    ).exists()
    path1 = list(METRICS.glob("run_*path1_c_only*seed_43.json"))

    out = {
        "n_files": sum(
            1 for arm in arms for s in seeds if "missing" not in arms[arm][s]["issues"]
        ),
        "means": means,
        "deltas": deltas,
        "gates_preview": {
            "G1": g1,
            "G2a": g2a,
            "G2b": g2b,
            "primary_story_hint": (
                "G2b" if g2b and not g2a else ("G2a" if g2a else "mixed")
            ),
        },
        "hard_untouched": hard_ok,
        "h1_seed43_present": h1,
        "path1_seed43_present": bool(path1),
        "all_arms_pass": all(arms[a][s].get("pass") for a in arms for s in seeds),
        "issues": issues,
        "arms": {
            a: {
                str(s): {
                    k: arms[a][s].get(k)
                    for k in (
                        "path",
                        "auroc",
                        "auprc",
                        "f1",
                        "q",
                        "pass",
                        "issues",
                        "skip_audit",
                    )
                }
                for s in seeds
            }
            for a in arms
        },
        "pass": len(issues) == 0 and hard_ok and all(
            arms[a][s].get("pass") for a in arms for s in seeds
        ),
    }
    REV.mkdir(parents=True, exist_ok=True)
    path = REV / "path_strag_t1_4_qa.json"
    path.write_text(json.dumps(out, indent=2))
    # keep thin inventory pointer
    (REV / "path_strag_t1_4_inventory.json").write_text(
        json.dumps(
            {
                "n_files": out["n_files"],
                "means": means,
                "deltas": deltas,
                "qa_detail": str(path),
                "pass": out["pass"],
            },
            indent=2,
        )
    )
    print(json.dumps({k: out[k] for k in ("means", "deltas", "gates_preview", "pass", "issues")}, indent=2))
    print(f"wrote {path}")
    if not out["pass"]:
        print("T1.4 QA FAIL", file=sys.stderr)
        raise SystemExit(1)
    print("T1.4 QA PASS")


if __name__ == "__main__":
    main()
