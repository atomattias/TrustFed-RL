#!/usr/bin/env python3
"""QA-audit Path STRAG T1.5 gate JSON vs plan locks and confirmatory metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"
METRICS = ROOT / "results" / "trustfed_agent" / "metrics"
HARD = ROOT / "config" / "iomt_natural_adversary_hard.json"
GATE = REV / "path_strag_comm_skip_f0.25_gate.json"
T14 = REV / "path_strag_t1_4_qa.json"

ATT = {"client_01", "client_06", "client_12"}
STRAG = {"client_09", "client_10", "client_11"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=Path, default=GATE)
    args = ap.parse_args()
    issues: list[str] = []
    checks: dict[str, bool] = {}

    if not args.gate.exists():
        print("missing gate JSON", file=sys.stderr)
        raise SystemExit(1)
    g = json.loads(args.gate.read_text())

    def ok(cid: str, cond: bool, detail: str = "") -> None:
        checks[cid] = bool(cond)
        if not cond:
            issues.append(f"{cid}: {detail}" if detail else cid)

    # T1.5.1 — file + protocol locks
    ok("T1.5.1_exists", True)
    ok("T1.5.1_suffix", g.get("suffix") == "path_strag", str(g.get("suffix")))
    ok("T1.5.1_q", abs(float(g.get("benign_skip_prob_q", -1)) - 0.5) < 1e-12)
    ok("T1.5.1_mode", g.get("mode") == "comm_skip")
    ok("T1.5.1_frac", abs(float(g.get("fraction", -1)) - 0.25) < 1e-12)
    ok("T1.5.1_seeds", g.get("seeds") == [43, 44, 45, 46, 47], str(g.get("seeds")))
    ok("T1.5.1_no42", 42 not in (g.get("seeds") or []))
    ok(
        "T1.5.1_att",
        set(g.get("attacker_ids_locked") or []) == ATT,
        str(g.get("attacker_ids_locked")),
    )
    ok(
        "T1.5.1_strag",
        set(g.get("straggler_ids_locked") or []) == STRAG,
        str(g.get("straggler_ids_locked")),
    )
    ok("T1.5.1_gt", g.get("gt_positives") == "attackers_only")

    # T1.5.2 — gates + exclusive primary
    gates = g.get("gates") or {}
    g1 = bool((gates.get("G1_participation_informative") or {}).get("pass"))
    g2a = bool((gates.get("G2a_multisignal_helps") or {}).get("pass"))
    g2b = bool((gates.get("G2b_C_still_dominates") or {}).get("pass"))
    primary = g.get("primary_story")
    ok("T1.5.2_G1", g1)
    ok("T1.5.2_G2a_false", g2a is False, f"G2a={g2a}")
    ok("T1.5.2_G2b", g2b)
    ok("T1.5.2_primary_G2b", primary == "G2b", str(primary))
    ok("T1.5.2_exclusive", g.get("exclusive_g2") is True and not (g2a and g2b))
    ok("T1.5.2_overall", g.get("overall_pass") is True)

    d_bc = (g.get("delta_Bc_minus_B1") or {}).get("mean")
    d_b2 = (g.get("delta_B2_minus_Bc") or {}).get("mean")
    ok("T1.5.2_delta_G1", d_bc is not None and float(d_bc) >= 0.05, str(d_bc))
    ok("T1.5.2_delta_G2b", d_b2 is not None and abs(float(d_b2)) < 0.05, str(d_b2))
    ok("T1.5.2_n_Bc_gt", g.get("n_seeds_Bc_gt_B1") == 5)
    ok("T1.5.2_n_B2_gt", g.get("n_seeds_B2_gt_Bc") == 0)

    # T1.5.3 — means + co-logs present
    mau = g.get("mean_auroc") or {}
    ok(
        "T1.5.3_means",
        abs(float((mau.get("B1") or {}).get("mean") or -1) - 0.5555555555555556) < 1e-9
        and float((mau.get("Bc") or {}).get("mean") or -1) == 1.0
        and float((mau.get("B2") or {}).get("mean") or -1) == 1.0,
        str({k: (mau.get(k) or {}).get("mean") for k in ("B1", "Bc", "B2")}),
    )
    ok("T1.5.3_auprc", bool(g.get("mean_auprc")))
    ok("T1.5.3_f1", bool(g.get("mean_f1")))
    ok("T1.5.3_alpha", bool(g.get("mean_alpha_malicious")))

    # T1.5.4 — per-seed audits embedded
    per = g.get("per_seed") or {}
    detail_4: list[str] = []
    n_ok = 0
    for s in ("43", "44", "45", "46", "47"):
        row = per.get(s) or {}
        for arm in ("B1", "Bc", "B2"):
            a = row.get(arm) or {}
            if a.get("audit_pass") is True:
                n_ok += 1
            else:
                detail_4.append(f"{arm} s{s}: {a.get('audit_issues')}")
            sa = a.get("skip_audit") or {}
            if sa.get("attacker_skips") != 33 or sa.get("expected_attacker_skips") != 33:
                detail_4.append(f"att_skip:{arm} s{s}: {sa}")
            if int(sa.get("benign_skips") or 0) < 1:
                detail_4.append(f"benign:{arm} s{s}")
            if set(a.get("attacker_ids") or []) != ATT:
                detail_4.append(f"att_ids:{arm} s{s}")
            if set(a.get("straggler_ids") or []) != STRAG:
                detail_4.append(f"strag_ids:{arm} s{s}")
    ok("T1.5.4_audits", n_ok == 15 and not detail_4, "; ".join(detail_4[:5]))
    issues.extend(f"T1.5.4:{x}" for x in detail_4)

    # T1.5.5 — locks vs hard / H1 / Path1 / Path AGG
    locks = g.get("locks") or {}
    hard = json.loads(HARD.read_text())
    hard_ok = (
        hard.get("status") == "frozen_primary_hard_attack"
        and "benign_straggler" not in hard
    )
    ok("T1.5.5_hard", locks.get("hard_untouched") is True and hard_ok)
    ok("T1.5.5_h1", locks.get("h1_seed43_present") is True)
    ok("T1.5.5_path1", locks.get("path1_seed43_present") is True)
    ok(
        "T1.5.5_h1_file",
        (
            METRICS
            / "run_trustfed_b2_uniform_iomt_natural_poison25_comm_skip_seed_43.json"
        ).exists(),
    )
    ok(
        "T1.5.5_path1_file",
        bool(list(METRICS.glob("run_*path1_c_only*seed_43.json"))),
    )

    # T1.5.6 — consistent with T1.4 QA preview
    if T14.exists():
        t14 = json.loads(T14.read_text())
        ok("T1.5.6_t14_pass", t14.get("pass") is True)
        hint = (t14.get("gates_preview") or {}).get("primary_story_hint")
        ok("T1.5.6_t14_hint", hint == "G2b", str(hint))
        m14 = t14.get("means") or {}
        ok(
            "T1.5.6_means_match",
            abs(float(m14.get("B1", -1)) - float((mau.get("B1") or {}).get("mean")))
            < 1e-12
            and abs(float(m14.get("Bc", -1)) - float((mau.get("Bc") or {}).get("mean")))
            < 1e-12
            and abs(float(m14.get("B2", -1)) - float((mau.get("B2") or {}).get("mean")))
            < 1e-12,
        )
    else:
        ok("T1.5.6_t14_pass", False, "missing t1_4_qa.json")

    # T1.5.7 — claim hygiene + companion artifacts
    forbidden = set(g.get("claim_forbidden") or [])
    needed_f = {
        "Aggregation defence",
        "Path 1 wrong",
        "Multi-hospital ECU",
        "AUROC=1.0 mandate",
        "Multi-signal necessity from this cell",
    }
    ok("T1.5.7_forbidden", needed_f <= forbidden, str(forbidden))
    narr = str(g.get("narrative") or "")
    disc = str(g.get("disclaimer") or "")
    ok(
        "T1.5.7_null_language",
        "null" in narr.lower() or "null" in disc.lower(),
        narr[:80],
    )
    ok(
        "T1.5.7_not_path1_fail",
        "not Path 1 failure" in disc or "not Path 1 failure" in narr,
    )
    ok(
        "T1.5.7_summary",
        (REV / "path_strag_comm_skip_f0.25_summary.json").exists(),
    )
    ok(
        "T1.5.7_paper_fill",
        (REV / "path_strag_comm_skip_f0.25_paper_fill.tex").exists(),
    )
    fill = ""
    fp = REV / "path_strag_comm_skip_f0.25_paper_fill.tex"
    if fp.exists():
        fill = fp.read_text()
    ok(
        "T1.5.7_fill_no_agg",
        "not an aggregation-defence" in fill.lower()
        or "not aggregation" in fill.lower(),
    )
    ok("T1.5.7_fill_G2b", "G2b" in fill)

    all_pass = all(checks.values())
    fail_issues = [k for k, v in checks.items() if not v]
    if detail_4:
        fail_issues.extend(f"T1.5.4:{x}" for x in detail_4[:10])
    out = {
        "gate": str(args.gate),
        "checks": checks,
        "issues": fail_issues,
        "primary_story": primary,
        "gates": {"G1": g1, "G2a": g2a, "G2b": g2b},
        "mean_auroc": {
            k: (mau.get(k) or {}).get("mean") for k in ("B1", "Bc", "B2")
        },
        "pass": all_pass,
    }
    REV.mkdir(parents=True, exist_ok=True)
    path = REV / "path_strag_t1_5_qa.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in ("pass", "primary_story", "gates", "mean_auroc", "issues")
            },
            indent=2,
        )
    )
    print(f"wrote {path}")
    if fail_issues:
        print("FAIL checks:", fail_issues, file=sys.stderr)
    if not all_pass:
        print("T1.5 QA FAIL", file=sys.stderr)
        raise SystemExit(1)
    print("T1.5 QA PASS")


if __name__ == "__main__":
    main()
