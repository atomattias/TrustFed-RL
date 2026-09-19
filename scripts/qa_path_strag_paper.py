#!/usr/bin/env python3
"""QA-audit Path STRAG T1.6 paper insert vs gate + Path 1 locks."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "docs" / "TrustFed_RL-4.tex"
GATE = ROOT / "results" / "trustfed_agent" / "rev_sup" / "path_strag_comm_skip_f0.25_gate.json"
REV = ROOT / "results" / "trustfed_agent" / "rev_sup"
T15 = REV / "path_strag_t1_5_qa.json"

# Locked Path 1 table numbers (must remain byte-identical in tab:path1_c_only)
PATH1_ROWS = [
    r"B1 ($V$ only) & $0.556\pm0.037$ & $0.549\pm0.023$ & $0.982\pm0.000$ &",
    r"Bc ($C$ only) & $1.000\pm0.000$ & $1.000\pm0.000$ & $0.982\pm0.000$ &",
    r"B2 (behavioural) & $1.000\pm0.000$ & $1.000\pm0.000$ & $0.982\pm0.000$ &",
]
PATH1_FOOT = [
    r"$|\Delta|_{\mathrm{B2,Bc}}{=}0$ on $5/5$ seeds;",
    r"$\Delta_{\mathrm{Bc{-}B1}}{=}{+}0.444$.",
]


def _extract_env(tex: str, label: str) -> str:
    """Return table body after \\label{label} until \\end{table}."""
    m = re.search(rf"\\label\{{{re.escape(label)}\}}(.*?)\\end\{{table\}}", tex, re.S)
    return m.group(1) if m else ""


def _section(tex: str, label: str, next_labels: list[str]) -> str:
    m = re.search(rf"\\label\{{{re.escape(label)}\}}", tex)
    if not m:
        return ""
    start = m.start()
    end = len(tex)
    for nl in next_labels:
        nm = re.search(rf"\\label\{{{re.escape(nl)}\}}", tex[start + 1 :])
        if nm:
            end = min(end, start + 1 + nm.start())
    return tex[start:end]


def main() -> None:
    checks: dict[str, bool] = {}
    issues: list[str] = []

    def ok(cid: str, cond: bool, detail: str = "") -> None:
        checks[cid] = bool(cond)
        if not cond:
            issues.append(f"{cid}: {detail}" if detail else cid)

    tex = TEX.read_text()
    gate = json.loads(GATE.read_text())
    mau = gate.get("mean_auroc") or {}

    # T1.6.1 — Path 1 table untouched
    p1 = _extract_env(tex, "tab:path1_c_only")
    ok("T1.6.1_path1_label", "\\label{tab:path1_c_only}" in tex)
    for i, row in enumerate(PATH1_ROWS):
        ok(f"T1.6.1_row{i}", row in p1, row[:40])
    for i, ft in enumerate(PATH1_FOOT):
        ok(f"T1.6.1_foot{i}", ft in p1, ft[:40])
    ok("T1.6.1_not_strag_suffix", "path_strag" not in p1)

    # T1.6.2 — STRAG subsection + table present (red)
    ok("T1.6.2_subsec", "\\label{subsec:res_strag}" in tex)
    ok("T1.6.2_tab", "\\label{tab:path_strag}" in tex)
    strag = _section(
        tex,
        "subsec:res_strag",
        ["subsec:res_agg", "subsec:res_ecu"],
    )
    ok("T1.6.2_red", r"{\color{red}" in strag or r"\color{red}" in strag)
    ok("T1.6.2_q", r"q{=}0.5" in strag or "q{=}0.5" in strag or "$q{=}0.5$" in strag)
    ok(
        "T1.6.2_att",
        ("client\\_01" in strag or "client_01" in strag)
        and ("client\\_06" in strag or "client_06" in strag)
        and ("client\\_12" in strag or "client_12" in strag),
    )
    ok(
        "T1.6.2_strag",
        ("client\\_09" in strag or "client_09" in strag)
        and ("client\\_10" in strag or "client_10" in strag)
        and ("client\\_11" in strag or "client_11" in strag),
    )
    ok("T1.6.2_gt", "attackers only" in strag.lower() or "malicious attackers only" in strag)
    ok(
        "T1.6.2_gate_name",
        "path\\_strag\\_comm\\_skip\\_f0.25\\_gate.json" in strag
        or "path_strag_comm_skip_f0.25_gate.json" in strag,
    )
    ok(
        "T1.6.2_config",
        "iomt\\_natural\\_adversary\\_path\\_strag.json" in strag
        or "iomt_natural_adversary_path_strag.json" in strag,
    )
    ok("T1.6.2_separate_metrics", "not reused Path~1" in strag or "separate" in strag)

    # T1.6.3 — numbers match gate (rounded paper form)
    tab = _extract_env(tex, "tab:path_strag")
    for i, row in enumerate(PATH1_ROWS):  # same rounded means as Path1 by outcome
        ok(f"T1.6.3_tab_row{i}", row in tab, row[:40])
    b1m = float((mau.get("B1") or {}).get("mean") or -1)
    bcm = float((mau.get("Bc") or {}).get("mean") or -1)
    b2m = float((mau.get("B2") or {}).get("mean") or -1)
    ok("T1.6.3_gate_b1", abs(b1m - 0.5555555555555556) < 1e-9)
    ok("T1.6.3_gate_bc_b2", bcm == 1.0 and b2m == 1.0)
    ok("T1.6.3_g2b_null", r"G2b null" in tab or "G2b null" in strag)
    ok("T1.6.3_delta0", r"\Delta_{\mathrm{B2{-}Bc}}{=}0" in strag or "B2{-}Bc" in strag)
    ok(
        "T1.6.3_primary_g2b",
        gate.get("primary_story") == "G2b" and gate.get("overall_pass") is True,
    )

    # T1.6.4 — claim hygiene in STRAG block
    forbid_hits = []
    low = strag.lower()
    for bad in (
        "aggregation defence proven",
        "path 1 wrong",
        "multi-hospital",
        "multi-signal necessity is established",
        "auroc=1.0 mandate",
    ):
        if bad in low:
            forbid_hits.append(bad)
    ok("T1.6.4_no_forbid", not forbid_hits, str(forbid_hits))
    ok("T1.6.4_honest_null", "honest null" in low or "g2b" in low)
    ok(
        "T1.6.4_not_agg",
        "not an aggregation-defence" in low or "not aggregation defence" in low,
    )
    ok("T1.6.4_not_overturn", "does not overturn Path~1" in strag)

    # T1.6.5 — roadmap / protocol / metrics / repro cross-refs
    ok("T1.6.5_roadmap", "subsec:res_strag" in tex and "benign-straggler Path~STRAG" in tex)
    ok(
        "T1.6.5_hyper_adv",
        "Path~STRAG" in tex
        and ("benign $q{=}0.5$" in tex or "benign $q" in tex),
    )
    ok("T1.6.5_ablation_note", "path_strag" in tex and "Path~STRAG (Section" in tex)
    ok("T1.6.5_metrics_gt", "honest stragglers are labelled benign" in tex)
    ok(
        "T1.6.5_repro",
        ("run\\_path\\_strag.sh" in tex or "run_path_strag.sh" in tex)
        and (
            "iomt\\_natural\\_adversary\\_path\\_strag.json" in tex
            or "iomt_natural_adversary_path_strag.json" in tex
        ),
    )

    # T1.6.6 — Discussion + Limitations + Conclusion
    ok(
        "T1.6.6_disc",
        "Path~STRAG adds honest intermittent misses" in tex
        and "multi-signal necessity claim" in tex,
    )
    ok(
        "T1.6.6_disc_no_deny_benign",
        "not evidence against adaptive attackers or benign network failures" not in tex,
    )
    ok(
        "T1.6.6_lim",
        "Path~STRAG partially addresses benign missingness" in tex
        and "G2b null" in tex,
    )
    ok(
        "T1.6.6_concl",
        "Path~STRAG confounds" in tex and "honest null" in tex,
    )
    ok(
        "T1.6.6_concl_future",
        "beyond the Path~STRAG" in tex or "Path~STRAG $q" in tex,
    )

    # T1.6.7 — gate QA still green; no multi-hospital ECU overclaim near STRAG
    if T15.exists():
        t15 = json.loads(T15.read_text())
        ok("T1.6.7_t15", t15.get("pass") is True)
    else:
        ok("T1.6.7_t15", False, "missing t1_5_qa")
    ok("T1.6.7_no_hospital_strag", "hospital" not in strag.lower())

    all_pass = all(checks.values())
    out = {
        "tex": str(TEX),
        "gate": str(GATE),
        "checks": checks,
        "issues": issues,
        "pass": all_pass,
    }
    REV.mkdir(parents=True, exist_ok=True)
    path = REV / "path_strag_t1_6_qa.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {"pass": all_pass, "n_checks": len(checks), "issues": issues},
            indent=2,
        )
    )
    print(f"wrote {path}")
    if not all_pass:
        print("T1.6 QA FAIL", file=sys.stderr)
        raise SystemExit(1)
    print("T1.6 QA PASS")


if __name__ == "__main__":
    main()
