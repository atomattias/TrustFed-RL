#!/usr/bin/env bash
# =============================================================================
# Paper confirmatory run — hard-regime B5 under Option 3 (skip-symmetric Δτ)
# =============================================================================
#
# Supports TrustFed_RL-4.tex Results table tab:trust_hard (B5 row) and the
# Method/Setup disclosure that RL trust calibration (Δτ) is *enabled* with a
# participation-symmetric skip path (not the older Option-2 Δτ-off workaround).
#
# Science lock (do not change without a new plan revision):
#   dataset:     iomt_natural
#   attack:      comm_skip (frozen hard mode)
#   f:           0.25 , t_a: 20 , rounds: 30
#   seeds:       43 44 45 46 47   (n=5 confirmatory; 42 = exploration only)
#   arm:         B5 only (B1/B2 already locked; do not re-run)
#   Δτ:          trust_calibration.enabled=true + skip-path apply_to_signals
#
# Paper soft gate (H2):
#   mean AUROC_B5 within 0.05 of locked B2 (≈1.0), or ≥0.95
#   expect α_mal ≈ 0 (skippers absent from uploads)
#
# Wall time: ~8–12 h on this machine.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_b5_option3.sh
#   bash scripts/run_b5_option3.sh
#   SMOKE_SEED=43 bash scripts/run_b5_option3.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

# --- Paper protocol knobs (defaults = camera-ready lock) ---------------------
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
SEEDS="${SEEDS:-43 44 45 46 47}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-comm_skip}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
ROUNDS="${ROUNDS:-30}"
RL_CFG="${RL_CFG:-config/iomt_rl_config.json}"
METRICS="${METRICS:-results/trustfed_agent/metrics}"
HARD_ADV="${HARD_ADV:-config/iomt_natural_adversary_hard.json}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_SEED="${SMOKE_SEED:-}"
FORCE="${FORCE:-0}"

mkdir -p "$REV_SUP"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TAG="${POISON_MODE}_f${FRAC}"
LOG="$REV_SUP/b5_hard_option3_${TAG}.log"
MANIFEST="$REV_SUP/b5_hard_option3_${TAG}_manifest.json"
GATE_OUT="$REV_SUP/b5_hard_option3_${TAG}_gate.json"
PAPER_JSON="$REV_SUP/b5_hard_option3_${TAG}_paper_fill.json"
PAPER_TEX="$REV_SUP/b5_hard_option3_${TAG}_paper_fill.tex"
CFG_SNAP="$REV_SUP/b5_hard_option3_${TAG}_iomt_rl_config.snapshot.json"
CLAIM_MD="$REV_SUP/b5_hard_option3_${TAG}_CLAIMS.md"

pct="$( "$PY" -c "print(int(round(100*float('$FRAC'))))" )"

{
  echo ""
  echo "===== Paper Option-3 B5 start $STAMP ====="
  echo "fills: tab:trust_hard B5 row + Method/Setup Δτ-on (skip-symmetric)"
  echo "protocol: mode=$POISON_MODE f=$FRAC t_a=$COMPROMISE_ROUND rounds=$ROUNDS seeds=[$SEEDS]"
  echo "artifacts: $GATE_OUT | $PAPER_JSON | $PAPER_TEX | $MANIFEST"
} | tee -a "$LOG"

# =============================================================================
# Phase A — Preflight (must pass before touching config or metrics)
# =============================================================================
echo "[A] preflight" | tee -a "$LOG"

# A1. Frozen hard-attack config present (documentation lock; CLI still passes mode)
if [[ -f "$HARD_ADV" ]]; then
  "$PY" - "$HARD_ADV" "$POISON_MODE" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
adv = json.loads(Path(sys.argv[1]).read_text())
mode = sys.argv[2]
assert adv.get("poison_mode") == mode, (adv.get("poison_mode"), mode)
assert adv.get("status") == "frozen_primary_hard_attack"
print(f"[ok] frozen hard adversary: {sys.argv[1]} mode={mode}")
PY
else
  echo "[warn] missing $HARD_ADV (continuing; CLI --poison-mode still set)" | tee -a "$LOG"
fi

# A2. Skip/participate Δτ wiring
SKIP_BLOCK="$(awk '/if skip:/{flag=1} flag{print} flag && /continue/{exit}' src/agent_experiment_runner.py)"
if ! grep -q 'apply_to_signals' <<<"$SKIP_BLOCK" \
  || ! grep -q 'participated=False' <<<"$SKIP_BLOCK"; then
  echo "ERROR: skip-path must call apply_to_signals(..., participated=False)" | tee -a "$LOG"
  exit 1
fi
if ! grep -q 'participated=True' src/agent_experiment_runner.py; then
  echo "ERROR: participate path must call apply_to_signals(..., participated=True)" | tee -a "$LOG"
  exit 1
fi
echo "[ok] skip/participate Δτ flags present" | tee -a "$LOG"

# A3. Symmetry regression suite (paper safety net)
echo "[A] pytest tests/test_deltatau_comm_skip_symmetry.py" | tee -a "$LOG"
"$PY" -m pytest tests/test_deltatau_comm_skip_symmetry.py -q | tee -a "$LOG"

# A4. No concurrent B5 / reduced-matrix job (runners only; not our own log tee)
_job_running() {
  local hits
  hits="$(
    ps -ax -o pid=,command= 2>/dev/null \
      | rg 'run_iomt_natural_reduced\.sh|trustfed_agent_runner\.py' \
      | rg -v 'run_b5_option3|builtin eval|rg run_iomt|DRY_RUN=1' \
      || true
  )"
  if [[ -n "$hits" ]]; then
    printf '%s\n' "$hits"
    return 0
  fi
  return 1
}
if [[ "$FORCE" != "1" ]]; then
  if hits="$(_job_running)"; then
    echo "ERROR: another B5 / iomt_natural job appears to be running." | tee -a "$LOG"
    echo "$hits" | head -20 | tee -a "$LOG"
    exit 2
  fi
fi
echo "[ok] no conflicting B5 job" | tee -a "$LOG"

# A5. Locked B1/B2 present (paper H1; we only fill B5)
for s in $SEEDS; do
  b1="$METRICS/run_trustfed_b1v_uniform_iomt_natural_poison${pct}_${POISON_MODE}_seed_${s}.json"
  b2="$METRICS/run_trustfed_b2_uniform_iomt_natural_poison${pct}_${POISON_MODE}_seed_${s}.json"
  if [[ ! -f "$b1" || ! -f "$b2" ]]; then
    echo "ERROR: missing locked H1 artifact for seed $s" | tee -a "$LOG"
    echo "  need: $b1" | tee -a "$LOG"
    echo "  need: $b2" | tee -a "$LOG"
    exit 1
  fi
done
echo "[ok] locked B1/B2 metrics present for seeds [$SEEDS]" | tee -a "$LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] Phase A passed. Would enable Δτ, run B5, write paper fills." | tee -a "$LOG"
  exit 0
fi

# =============================================================================
# Phase B — Config for reported B5 (Δτ on)
# =============================================================================
echo "[B] enable trust_calibration + snapshot (post-enable = as-run)" | tee -a "$LOG"
"$PY" - "$RL_CFG" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
cfg = json.loads(path.read_text())
tc = cfg.setdefault("trust_calibration", {})
prev = bool(tc.get("enabled", False))
tc["enabled"] = True
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"[ok] {path}: trust_calibration.enabled {prev} -> True")
PY
# Snapshot AFTER enable so the artifact matches the reported run.
cp "$RL_CFG" "$CFG_SNAP"
echo "[ok] snapshot → $CFG_SNAP" | tee -a "$LOG"

# =============================================================================
# Phase C — Run B5 only
# =============================================================================
run_b5_seeds() {
  local seed_list="$1"
  INCLUDE_B0=0 INCLUDE_B1=0 INCLUDE_B2=0 INCLUDE_B3=0 INCLUDE_B4=0 INCLUDE_B5=1 \
    RUN_BENIGN=0 SEEDS="$seed_list" POISON_MODE="$POISON_MODE" FRAC="$FRAC" \
    COMPROMISE_ROUND="$COMPROMISE_ROUND" ROUNDS="$ROUNDS" \
    bash scripts/run_iomt_natural_reduced.sh
}

if [[ -n "$SMOKE_SEED" ]]; then
  echo "[C] smoke seed=$SMOKE_SEED (not confirmatory; no paper_fill)" | tee -a "$LOG"
  rm -f "$METRICS/run_trustfed_agent_uniform_iomt_natural_poison${pct}_${POISON_MODE}_seed_${SMOKE_SEED}.json"
  run_b5_seeds "$SMOKE_SEED" 2>&1 | tee -a "$LOG"
  "$PY" - <<PY | tee -a "$LOG"
import json
from pathlib import Path
p = Path("$METRICS") / f"run_trustfed_agent_uniform_iomt_natural_poison${pct}_${POISON_MODE}_seed_${SMOKE_SEED}.json"
d = json.loads(p.read_text())
disc = d.get("trust_discrimination") or {}
print(f"smoke AUROC={disc.get('auroc')} AUPRC={disc.get('auprc')} F1={(d.get('detection') or {}).get('f1')}")
print(f"alpha_mal={disc.get('mean_alpha_malicious')} FD={disc.get('false_distrust_rate')}")
tc = None
for row in reversed(d.get("round_logs") or []):
    if row.get("trust_calibration"):
        tc = row["trust_calibration"]; break
if tc is None:
    tc = (d.get("rl") or {}).get("trust_calibration")
print(f"trust_calibration={tc}")
assert tc is not None and tc.get("enabled") is True, (
    "Δτ must be enabled=true for Option 3 smoke"
)
print("Re-run without SMOKE_SEED for confirmatory n=5 paper fill.")
PY
  exit 0
fi

# Claims checklist only for the confirmatory (n=5) path
cat > "$CLAIM_MD" <<EOF
# Paper claims supported by this run (Option 3)

**Target tex:** \`docs/TrustFed_RL-4.tex\`
**Table:** \`tab:trust_hard\` B5 row (hard \`comm_skip\`, \$f=0.25\$, seeds 43–47)

## What to claim
- Full-stack **B5** under frozen hard attack \`comm_skip\` with **Δτ enabled**.
- Skip path applies the same Δτ channel as participate (\`participated=False\` + peer-mean guard; offset-only Δτ).
- Primary H1 contrast remains **B1 ≪ B2** (locked; not re-run here).
- B5 is a **system arm** (behavioural fusion + RL/gov); soft-check AUROC ≈ B2.

## What NOT to claim
- Do **not** claim Δτ *improves* ranking vs B2.
- Do **not** present Option-2 (Δτ-off) numbers for B5.
- Do **not** retune \$t_a\$, \$f\$, seeds, or attack mode from this run.

## After success — edit tex
1. Replace B5 TBD cells in \`tab:trust_hard\` using \`$PAPER_TEX\`.
2. Method / Setup: Δτ **enabled** for reported IoMT B5; skip-symmetric (not disabled).
3. Discussion / Limitations: remove “Δτ-off workaround”; keep skip-path design note if useful.
4. Reproducibility: point to \`$REV_SUP/b5_hard_option3_*\`.

Started: $STAMP
EOF
echo "[ok] wrote $CLAIM_MD" | tee -a "$LOG"

echo "[C] delete prior hard B5 JSONs (invalid Option-2 / broken Δτ-on)" | tee -a "$LOG"
for s in $SEEDS; do
  rm -f "$METRICS/run_trustfed_agent_uniform_iomt_natural_poison${pct}_${POISON_MODE}_seed_${s}.json"
done

echo "[C] launch confirmatory B5 matrix seeds=[$SEEDS]" | tee -a "$LOG"
run_b5_seeds "$SEEDS" 2>&1 | tee -a "$LOG"

# =============================================================================
# Phase D — Soft gate + paper fills (JSON + LaTeX row)
# =============================================================================
echo "[D] soft gate + paper_fill exports" | tee -a "$LOG"
"$PY" - <<PY | tee -a "$LOG"
import json, statistics as st
from pathlib import Path
from datetime import datetime, timezone

seeds = [int(x) for x in "$SEEDS".split()]
pct = int("$pct")
mode = "$POISON_MODE"
frac = float("$FRAC")
metrics = Path("$METRICS")
rev = Path("$REV_SUP")

def load_disc(path: Path):
    d = json.loads(path.read_text())
    disc = d.get("trust_discrimination") or {}
    det = d.get("detection") or {}
    # Prefer top-level rl.trust_calibration; fall back to last round_log.
    tc = (d.get("rl") or {}).get("trust_calibration")
    if not tc:
        for row in reversed(d.get("round_logs") or []):
            if row.get("trust_calibration"):
                tc = row["trust_calibration"]
                break
    return {
        "auroc": disc.get("auroc"),
        "auprc": disc.get("auprc"),
        "delay": disc.get("detection_delay"),
        "fd": disc.get("false_distrust_rate"),
        "alpha_mal": disc.get("mean_alpha_malicious"),
        "f1": det.get("f1"),
        "trust_calibration": tc,
        "path": str(path),
    }

rows = []
for s in seeds:
    p5 = metrics / f"run_trustfed_agent_uniform_iomt_natural_poison{pct}_{mode}_seed_{s}.json"
    p2 = metrics / f"run_trustfed_b2_uniform_iomt_natural_poison{pct}_{mode}_seed_{s}.json"
    p1 = metrics / f"run_trustfed_b1v_uniform_iomt_natural_poison{pct}_{mode}_seed_{s}.json"
    if not p5.exists():
        raise SystemExit(f"missing B5: {p5}")
    b5 = load_disc(p5)
    b2 = load_disc(p2) if p2.exists() else {}
    b1 = load_disc(p1) if p1.exists() else {}
    if b5["auroc"] is None:
        raise SystemExit(f"no AUROC in {p5}")
    tc = b5.get("trust_calibration") or {}
    # Must be explicitly True — missing/None must not slip through as Option 3.
    if tc.get("enabled") is not True:
        raise SystemExit(
            f"B5 seed {s} missing trust_calibration.enabled=true "
            f"(got {tc!r}) — not Option 3"
        )
    rows.append({
        "seed": s,
        "b5": b5,
        "b2_auroc": b2.get("auroc"),
        "b1_auroc": b1.get("auroc"),
    })

def mean_std(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.pstdev(vals)

au = [r["b5"]["auroc"] for r in rows]
ap = [r["b5"]["auprc"] for r in rows]
f1 = [r["b5"]["f1"] for r in rows]
fd = [r["b5"]["fd"] for r in rows]
am = [r["b5"]["alpha_mal"] for r in rows]
b2a = [r["b2_auroc"] for r in rows if r["b2_auroc"] is not None]

m_au, s_au = mean_std(au)
m_ap, s_ap = mean_std(ap)
m_f1, s_f1 = mean_std(f1)
m_fd, s_fd = mean_std(fd)
m_am, s_am = mean_std(am)
m_b2, _ = mean_std(b2a)

tol = 0.05
soft_pass = (m_b2 is None) or (abs(m_au - m_b2) <= tol) or (m_au >= 0.95)

def fmt(m, s, digits=3):
    if m is None:
        return "TBD"
    return f"{m:.{digits}f}\\pm{s:.{digits}f}"

delta = (m_au - m_b2) if (m_au is not None and m_b2 is not None) else None
delta_s = f"{delta:+.3f}" if delta is not None else "n/a"
_d = "$"
tex_row = (
    "B5 (full; $\\Delta\\tau$ on, skip-symmetric) & "
    f"{_d}{fmt(m_au, s_au)}{_d} & {_d}{fmt(m_ap, s_ap)}{_d} & {_d}{fmt(m_f1, s_f1)}{_d} & "
    f"{_d}\\Delta{_d} vs B2 {_d}{delta_s}{_d} \\\\\n"
)

gate = {
    "role": "paper_option3_b5_hard_skip_symmetric_deltatau",
    "table": "tab:trust_hard",
    "poison_mode": mode,
    "adversary_fraction": frac,
    "compromise_round": int("$COMPROMISE_ROUND"),
    "num_rounds": int("$ROUNDS"),
    "seeds": seeds,
    "trust_calibration_enabled": True,
    "skip_path_symmetric": True,
    "soft_pass": soft_pass,
    "soft_pass_rule": "mean_auroc_b5 within 0.05 of b2 OR >= 0.95",
    "mean_auroc_b5": m_au,
    "std_auroc_b5": s_au,
    "mean_auprc_b5": m_ap,
    "std_auprc_b5": s_ap,
    "mean_f1_b5": m_f1,
    "std_f1_b5": s_f1,
    "mean_fd_b5": m_fd,
    "mean_alpha_mal_b5": m_am,
    "mean_auroc_b2_locked": m_b2,
    "delta_auroc_b5_minus_b2": delta,
    "per_seed": rows,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
Path("$GATE_OUT").write_text(json.dumps(gate, indent=2) + "\n")

paper = {
    "tex_table": "tab:trust_hard",
    "b5_row_latex": tex_row.strip(),
    "caption_note": (
        "B5 uses behavioural fusion with RL/governance and $\\Delta\\tau$ enabled "
        "under a skip-symmetric calibration path (Option 3)."
    ),
    "method_disclosure": (
        "Reported IoMT B5 enables trust_calibration with apply_to_signals on both "
        "participate and skip paths (peer-mean guard; offset-only $\\Delta\\tau$)."
    ),
    "numbers": {
        "auroc": {"mean": m_au, "std": s_au},
        "auprc": {"mean": m_ap, "std": s_ap},
        "f1": {"mean": m_f1, "std": s_f1},
        "false_distrust": {"mean": m_fd, "std": s_fd},
        "alpha_mal": {"mean": m_am, "std": s_am},
    },
    "soft_pass": soft_pass,
    "artifacts": {
        "gate": "$GATE_OUT",
        "log": "$LOG",
        "config_snapshot": "$CFG_SNAP",
        "claims": "$CLAIM_MD",
    },
}
Path("$PAPER_JSON").write_text(json.dumps(paper, indent=2) + "\n")

tex = f"""% Auto-generated by scripts/run_b5_option3.sh — paste into tab:trust_hard
% soft_pass={soft_pass}  seeds={seeds}  mode={mode} f={frac}
{tex_row}
% Suggested caption fragment:
% {paper["caption_note"]}
"""
Path("$PAPER_TEX").write_text(tex)

manifest = {
    "protocol": {
        "dataset": "iomt_natural",
        "poison_mode": mode,
        "adversary_fraction": frac,
        "compromise_round": int("$COMPROMISE_ROUND"),
        "num_rounds": int("$ROUNDS"),
        "seeds": seeds,
        "arm": "B5",
        "option": 3,
        "trust_calibration_enabled": True,
    },
    "reproducibility": {
        "script": "scripts/run_b5_option3.sh",
        "runner": "scripts/run_iomt_natural_reduced.sh",
        "rl_config": "$RL_CFG",
        "rl_config_snapshot": "$CFG_SNAP",
        "hard_adversary_doc": "$HARD_ADV",
        "symmetry_tests": "tests/test_deltatau_comm_skip_symmetry.py",
        "log": "$LOG",
    },
    "paper": {
        "tex": "docs/TrustFed_RL-4.tex",
        "table": "tab:trust_hard",
        "fill_json": "$PAPER_JSON",
        "fill_tex": "$PAPER_TEX",
        "claims": "$CLAIM_MD",
    },
    "gate": "$GATE_OUT",
    "soft_pass": soft_pass,
}
Path("$MANIFEST").write_text(json.dumps(manifest, indent=2) + "\n")

print(json.dumps({"soft_pass": soft_pass, "auroc": m_au, "auprc": m_ap, "f1": m_f1, "delta_vs_b2": delta}, indent=2))
print(f"[ok] gate     → $GATE_OUT")
print(f"[ok] paper    → $PAPER_JSON")
print(f"[ok] tex row  → $PAPER_TEX")
print(f"[ok] manifest → $MANIFEST")
print("--- paste into tab:trust_hard ---")
print(tex_row)
if not soft_pass:
    raise SystemExit("Option 3 soft gate FAILED — do not paste into paper")
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Option 3 B5 finished — paper fills ready" | tee -a "$LOG"
echo "Claims:   $CLAIM_MD"
echo "Gate:     $GATE_OUT"
echo "TeX row:  $PAPER_TEX"
echo "Manifest: $MANIFEST"
