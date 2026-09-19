#!/usr/bin/env python3
"""Summarize Path STRAG seed-42 explore (B1/Bc/B2) + QA audit; decide freeze vs one q-retune."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"
HARD = ROOT / "config" / "iomt_natural_adversary_hard.json"

ARM_PATTERNS = {
    "B1": "run_trustfed_b1v_uniform_iomt_natural_{tag}_seed_{seed}.json",
    "Bc": "run_trustfed_bc_uniform_iomt_natural_{tag}_seed_{seed}.json",
    "B2": "run_trustfed_b2_uniform_iomt_natural_{tag}_seed_{seed}.json",
}

EXPECTED_WEIGHTS = {
    "B1": {"V": 1.0, "S": 0.0, "D": 0.0, "U": 0.0, "C": 0.0, "R": 0.0},
    "Bc": {"V": 0.0, "S": 0.0, "D": 0.0, "U": 0.0, "C": 1.0, "R": 0.0},
    "B2": {"V": 0.2, "S": 0.2, "D": 0.2, "U": 0.2, "C": 0.2, "R": 0.0},
}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _weights_ok(got: dict | None, exp: dict) -> bool:
    if not got:
        return False
    for k, v in exp.items():
        if abs(float(got.get(k, -1)) - float(v)) > 1e-9:
            return False
    return True


def audit_arm(name: str, d: dict | None) -> dict:
    out: dict = {"arm": name, "pass": False, "issues": []}
    if not d:
        out["issues"].append("missing_metrics")
        return out
    lc = d.get("late_compromise") or {}
    bs = lc.get("benign_straggler") or {}
    disc = d.get("trust_discrimination") or {}
    det = d.get("detection") or {}
    att = set(lc.get("attacker_client_ids") or [])
    strag = set(bs.get("straggler_client_ids") or [])
    disc_att = set(disc.get("attacker_ids") or [])

    out.update(
        {
            "approach": d.get("approach"),
            "run_id": d.get("run_id"),
            "auroc": disc.get("auroc"),
            "auprc": disc.get("auprc"),
            "f1": det.get("f1"),
            "mean_alpha_malicious": disc.get("mean_alpha_malicious"),
            "skip_prob_q": bs.get("skip_prob_q"),
            "benign_enabled": bs.get("enabled"),
            "attacker_ids": sorted(att),
            "straggler_ids": sorted(strag),
            "signal_weights": d.get("signal_weights"),
            "num_rounds": d.get("num_rounds"),
            "compromise_round": lc.get("compromise_round"),
        }
    )

    if d.get("num_rounds") != 30:
        out["issues"].append(f"num_rounds={d.get('num_rounds')}")
    if lc.get("compromise_round") != 20:
        out["issues"].append(f"ta={lc.get('compromise_round')}")
    if not bs.get("enabled"):
        out["issues"].append("benign_straggler disabled")
    if bs.get("skip_prob_q") != 0.5 and out.get("skip_prob_q") is not None:
        # allow retune suffixes later; default explore expects 0.5 unless --q set
        pass
    if att != {"client_01", "client_06", "client_12"}:
        out["issues"].append(f"unexpected attackers {sorted(att)}")
    if strag != {"client_09", "client_10", "client_11"}:
        out["issues"].append(f"unexpected stragglers {sorted(strag)}")
    if not att.isdisjoint(strag):
        out["issues"].append("attacker/straggler overlap")
    if disc_att != att:
        out["issues"].append("disc attacker_ids mismatch")
    if strag & disc_att:
        out["issues"].append("stragglers in AUROC positives")
    if not _weights_ok(d.get("signal_weights"), EXPECTED_WEIGHTS[name]):
        out["issues"].append(f"signal_weights != {EXPECTED_WEIGHTS[name]}")

    logs = d.get("round_logs") or []
    ta = int(lc.get("compromise_round") or 20)
    post = [r for r in logs if int(r.get("round", 0)) >= ta]
    reasons: Counter = Counter()
    bad_att = 0
    benign_skips = 0
    missing_fields = 0
    for r in post:
        for cid, row in (r.get("client_trust") or {}).items():
            if "is_straggler" not in row or "skip_reason" not in row:
                missing_fields += 1
            reason = row.get("skip_reason")
            reasons[reason] += 1
            if cid in att:
                if reason != "attacker_comm_skip" or float(row.get("alpha") or 0) != 0.0:
                    bad_att += 1
            if reason == "benign_straggler":
                benign_skips += 1
                if cid not in strag or row.get("is_attacker"):
                    out["issues"].append(f"bad benign skip {cid} r{r.get('round')}")
                if float(row.get("alpha") or 0) != 0.0:
                    out["issues"].append(f"benign skip alpha!=0 {cid}")

    out["skip_audit"] = {
        "post_ta_rounds": len(post),
        "benign_straggler_skips": benign_skips,
        "attacker_skip_failures": bad_att,
        "missing_skip_fields": missing_fields,
        "reason_counts": {str(k): int(v) for k, v in reasons.items()},
        "expected_attacker_skips": len(att) * len(post),
    }
    if not post:
        out["issues"].append("no post-ta rounds")
    if bad_att:
        out["issues"].append(f"attacker_skip_failures={bad_att}")
    if benign_skips < 1:
        out["issues"].append("no benign_straggler skips")
    if missing_fields:
        out["issues"].append(f"missing_skip_fields={missing_fields}")
    # 3 attackers × n post rounds
    exp_att = len(att) * len(post)
    if reasons.get("attacker_comm_skip", 0) != exp_att:
        out["issues"].append(
            f"attacker_comm_skip count {reasons.get('attacker_comm_skip')} != {exp_att}"
        )

    out["pass"] = len(out["issues"]) == 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="path_strag_explore")
    ap.add_argument("--q", type=float, default=0.5)
    args = ap.parse_args()

    tag = f"poison25_comm_skip_strag_{args.suffix}"
    audits = {}
    arms = {}
    for name, pat in ARM_PATTERNS.items():
        path = METRICS / pat.format(tag=tag, seed=args.seed)
        raw = _load(path)
        aud = audit_arm(name, raw)
        aud["path"] = path.name
        if raw and aud.get("skip_prob_q") is not None and abs(float(aud["skip_prob_q"]) - float(args.q)) > 1e-12:
            aud["issues"].append(f"q={aud['skip_prob_q']} != expected {args.q}")
            aud["pass"] = False
        audits[name] = aud
        arms[name] = {
            k: aud.get(k)
            for k in (
                "approach",
                "run_id",
                "auroc",
                "auprc",
                "f1",
                "mean_alpha_malicious",
                "attacker_ids",
                "straggler_ids",
                "skip_prob_q",
                "benign_enabled",
                "path",
            )
        }
        if aud.get("missing") or aud["issues"] == ["missing_metrics"]:
            arms[name] = {"missing": True, "path": path.name}

    def auroc(arm: str):
        r = arms[arm]
        return None if r.get("missing") else r.get("auroc")

    a1, ac, a2 = auroc("B1"), auroc("Bc"), auroc("B2")
    missing = any(v is None for v in (a1, ac, a2))
    g1 = None if missing else ((ac - a1) >= 0.05)
    g2a = None if missing else ((a2 - ac) >= 0.05)
    g2b = None if missing else (abs(a2 - ac) < 0.05)
    all_sat = None if missing else all(x >= 0.99 for x in (a1, ac, a2))
    # Pre-registered retune trigger only (do NOT retune merely because B2≈Bc)
    need_retune = (not missing) and ((g1 is False) or bool(all_sat))

    if missing:
        story = "incomplete"
    elif g2a:
        story = "G2a_candidate"
    elif g2b:
        story = "G2b_null_candidate"
    else:
        story = "mixed_B2_vs_Bc"

    hard = json.loads(HARD.read_text()) if HARD.exists() else {}
    hard_ok = hard.get("status") == "frozen_primary_hard_attack" and "benign_straggler" not in hard

    qa = {
        "arms_pass": {k: bool(v.get("pass")) for k, v in audits.items()},
        "all_arms_pass": all(bool(v.get("pass")) for v in audits.values()),
        "hard_adversary_untouched": hard_ok,
        "arm_details": audits,
        "notes": [
            "Retune only if G1 fails or all three AUROCs ≥ 0.99.",
            "B2≈Bc with G1 pass is a valid G2b null explore outcome — freeze q, do not retune.",
        ],
    }

    out = {
        "seed": args.seed,
        "suffix": args.suffix,
        "q": args.q,
        "arms": arms,
        "deltas": {
            "Bc_minus_B1": None if missing else ac - a1,
            "B2_minus_Bc": None if missing else a2 - ac,
        },
        "explore_checks": {
            "G1_Bc_ge_B1_plus_0.05": g1,
            "G2a_B2_ge_Bc_plus_0.05": g2a,
            "G2b_abs_B2_Bc_lt_0.05": g2b,
            "all_auroc_ge_0.99": all_sat,
        },
        "need_retune": need_retune,
        "retune_policy": {
            "max_attempts": 1,
            "q_candidates_if_needed": [0.3, 0.7],
            "trigger": "G1 fail OR all AUROC >= 0.99 on explore seed 42",
            "non_trigger": "G2b (B2≈Bc) alone does not force retune",
        },
        "story_hint": story,
        "freeze_q": (None if need_retune or not qa["all_arms_pass"] else args.q),
        "qa": qa,
    }
    REV.mkdir(parents=True, exist_ok=True)
    path = REV / f"path_strag_explore_q{args.q}_f0.25_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    if not qa["all_arms_pass"] or not hard_ok:
        print("QA_FAIL")
        raise SystemExit(1)
    if need_retune:
        print("RETUNE_RECOMMENDED: try Q=0.3 then Q=0.7 once (seed 42 only)")
    else:
        print(f"QA_PASS FREEZE_Q={args.q} story_hint={story}")


if __name__ == "__main__":
    main()
