#!/usr/bin/env bash
# =============================================================================
# DO3 RL bake-off: Static (B3) / PPO-unmasked / Maskable PPO (B5)
# =============================================================================
# Same response environment, governance ON for all three arms.
# Easy label-flip @ f=0.25 (does not reopen hard H1 / Option-3 B5).
#
# Pass rule (Maskable M vs unmasked P):
#   invalid_proposal_rate(M) < invalid_proposal_rate(P)
#   AND (precision(M) >= precision(P)-0.02 OR mean_reward(M) >= mean_reward(P))
#
# Usage:
#   DRY_RUN=1 bash scripts/run_do3_rl_bakeoff.sh
#   bash scripts/run_do3_rl_bakeoff.sh
#   SEEDS="42 43" bash scripts/run_do3_rl_bakeoff.sh
#   SKIP_EXISTING=1 bash scripts/run_do3_rl_bakeoff.sh   # reuse B3/B5 if present
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
METRICS="${METRICS:-results/trustfed_agent/metrics}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-30}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-label_flip}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
# Default: re-run so invalid_proposal_rate is instrumented (old B3/B5 JSONs lack it).
RUN_STATIC="${RUN_STATIC:-1}"
RUN_UNMASKED="${RUN_UNMASKED:-1}"
RUN_MASKABLE="${RUN_MASKABLE:-1}"

MASK_CFG="${MASK_CFG:-config/iomt_rl_config.json}"
NOMASK_CFG="${NOMASK_CFG:-config/iomt_rl_config_nomask.json}"
# Keep bake-off metrics out of locked B3/B5 paper filenames.
METRICS_SUFFIX="${METRICS_SUFFIX:-bakeoff}"

mkdir -p "$REV_SUP"
TAG="${POISON_MODE}_f${FRAC}"
LOG="$REV_SUP/do3_rl_bakeoff_${TAG}.log"
GATE="$REV_SUP/do3_rl_bakeoff_${TAG}_gate.json"
SUMMARY="$REV_SUP/do3_rl_bakeoff_${TAG}_summary.json"
PAPER="$REV_SUP/do3_rl_bakeoff_${TAG}_paper_fill.tex"
pct="$("$PY" -c "print(int(round(100*float('$FRAC'))))")"

echo "===== DO3 RL bake-off mode=$POISON_MODE f=$FRAC seeds=[$SEEDS] suffix=$METRICS_SUFFIX =====" | tee -a "$LOG"

# Preflight: masking flag must be live in PolicyOptimizer (source check; avoids numpy import)
if ! grep -q 'self.use_action_masking = bool(rl_config.get("use_action_masking"' src/rl/policy_optimizer.py; then
  echo "ERROR: use_action_masking not wired in PolicyOptimizer" | tee -a "$LOG"
  exit 1
fi
if ! grep -q 'invalid_proposal_rate' src/rl/response_strategy.py; then
  echo "ERROR: invalid_proposal_rate not tracked in ResponseStrategy" | tee -a "$LOG"
  exit 1
fi
echo "[ok] use_action_masking + invalid_proposal_rate wired" | tee -a "$LOG"
late=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
      --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")

metric_path() {
  local approach="$1" seed="$2" nomask="${3:-0}"
  local rid
  case "$approach" in
    trustfed_governance) rid="trustfed_governance_uniform_iomt_natural_poison${pct}" ;;
    trustfed_agent)
      if [[ "$nomask" == "1" ]]; then
        rid="trustfed_agent_uniform_iomt_natural_poison${pct}_nomask"
      else
        rid="trustfed_agent_uniform_iomt_natural_poison${pct}"
      fi
      ;;
    *) rid="${approach}_uniform_iomt_natural_poison${pct}" ;;
  esac
  # label_flip has no mode suffix in run_id
  if [[ "$POISON_MODE" != "label_flip" ]]; then
    rid="${rid}_${POISON_MODE}"
  fi
  if [[ -n "${METRICS_SUFFIX}" ]]; then
    rid="${rid}_${METRICS_SUFFIX}"
  fi
  echo "$METRICS/run_${rid}_seed_${seed}.json"
}

run_one() {
  local seed="$1" approach="$2" rl_cfg="$3" nomask="$4" label="$5"
  local out
  out="$(metric_path "$approach" "$seed" "$nomask")"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "[skip] $label seed=$seed exists: $out" | tee -a "$LOG"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] $label seed=$seed -> $out" | tee -a "$LOG"
    return 0
  fi
  echo "[run] $label seed=$seed" | tee -a "$LOG"
  local suffix_args=()
  if [[ -n "${METRICS_SUFFIX}" ]]; then
    suffix_args=(--metrics-suffix "$METRICS_SUFFIX")
  fi
  "$PY" trustfed_agent_runner.py \
    --dataset iomt_natural \
    --approach "$approach" \
    --random-state "$seed" \
    --num-rounds "$ROUNDS" \
    --uniform-b-prior \
    --rl-config "$rl_cfg" \
    "${suffix_args[@]}" \
    "${late[@]}"
}

for seed in $SEEDS; do
  if [[ "$RUN_STATIC" == "1" ]]; then
    run_one "$seed" trustfed_governance "$MASK_CFG" 0 "S-static"
  fi
  if [[ "$RUN_UNMASKED" == "1" ]]; then
    run_one "$seed" trustfed_agent "$NOMASK_CFG" 1 "P-unmasked"
  fi
  if [[ "$RUN_MASKABLE" == "1" ]]; then
    run_one "$seed" trustfed_agent "$MASK_CFG" 0 "M-maskable"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry] skip gate aggregation" | tee -a "$LOG"
  exit 0
fi

"$PY" - "$METRICS" "$SEEDS" "$pct" "$POISON_MODE" "$GATE" "$SUMMARY" "$PAPER" "$METRICS_SUFFIX" <<'PY' | tee -a "$LOG"
import json, sys, statistics
from pathlib import Path

metrics_dir = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
pct = sys.argv[3]
mode = sys.argv[4]
gate_path, summary_path, paper_path = map(Path, sys.argv[5:8])
suffix = sys.argv[8].strip() if len(sys.argv) > 8 else ""

def load(approach, seed, nomask=False):
    rid = f"{approach}_uniform_iomt_natural_poison{pct}"
    if nomask:
        rid += "_nomask"
    if mode != "label_flip":
        rid += f"_{mode}"
    if suffix:
        rid += f"_{suffix}"
    path = metrics_dir / f"run_{rid}_seed_{seed}.json"
    if not path.exists():
        return None, str(path)
    return json.loads(path.read_text()), str(path)

def extract(rec):
    if not rec:
        return None
    resp = rec.get("response") or {}
    gov = rec.get("governance") or {}
    if "invalid_proposal_rate" not in resp and "invalid_proposal_rate" not in (rec.get("rl") or {}):
        return None
    return {
        "precision": resp.get("precision"),
        "invalid_proposal_rate": resp.get(
            "invalid_proposal_rate",
            (rec.get("rl") or {}).get("invalid_proposal_rate"),
        ),
        "mean_reward": resp.get("mean_reward"),
        "human_review_rate": gov.get("human_review_rate"),
        "violation_rate": gov.get("violation_rate"),
        "intervention_rate": gov.get("intervention_rate", gov.get("violation_rate")),
        "use_action_masking": (rec.get("rl") or {}).get("use_action_masking"),
    }

arms = {"static": [], "ppo_unmasked": [], "maskable_ppo": []}
missing = []
for seed in seeds:
    for key, approach, nomask in (
        ("static", "trustfed_governance", False),
        ("ppo_unmasked", "trustfed_agent", True),
        ("maskable_ppo", "trustfed_agent", False),
    ):
        rec, path = load(approach, seed, nomask)
        row = extract(rec) if rec is not None else None
        if row is None:
            missing.append(path)
        else:
            arms[key].append({"seed": seed, **row})

def mean_field(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)

summary = {"arms": {}, "missing": missing, "seeds": seeds, "poison_mode": mode, "metrics_suffix": suffix}
for name, rows in arms.items():
    summary["arms"][name] = {
        "n": len(rows),
        "rows": rows,
        "precision_mean_std": mean_field(rows, "precision"),
        "invalid_mean_std": mean_field(rows, "invalid_proposal_rate"),
        "reward_mean_std": mean_field(rows, "mean_reward"),
        "human_rev_mean_std": mean_field(rows, "human_review_rate"),
        "intervention_mean_std": mean_field(rows, "intervention_rate"),
    }

m_inv = summary["arms"]["maskable_ppo"]["invalid_mean_std"][0]
p_inv = summary["arms"]["ppo_unmasked"]["invalid_mean_std"][0]
m_prec = summary["arms"]["maskable_ppo"]["precision_mean_std"][0]
p_prec = summary["arms"]["ppo_unmasked"]["precision_mean_std"][0]
m_rew = summary["arms"]["maskable_ppo"]["reward_mean_std"][0]
p_rew = summary["arms"]["ppo_unmasked"]["reward_mean_std"][0]

checks = {
    "has_all_arms": not missing and all(summary["arms"][a]["n"] for a in arms),
    "invalid_m_lt_p": (m_inv is not None and p_inv is not None and m_inv < p_inv),
    "utility_ok": (
        (m_prec is not None and p_prec is not None and m_prec >= p_prec - 0.02)
        or (m_rew is not None and p_rew is not None and m_rew >= p_rew)
    ),
}
passed = bool(checks["has_all_arms"] and checks["invalid_m_lt_p"] and checks["utility_ok"])
gate = {
    "pass": passed,
    "checks": checks,
    "means": {
        "maskable_invalid": m_inv,
        "unmasked_invalid": p_inv,
        "maskable_precision": m_prec,
        "unmasked_precision": p_prec,
        "maskable_reward": m_rew,
        "unmasked_reward": p_rew,
    },
    "reason": "ok" if passed else ("incomplete" if missing else "bakeoff_gate_failed"),
}
summary_path.write_text(json.dumps(summary, indent=2))
gate_path.write_text(json.dumps(gate, indent=2))

def fmt(ms):
    if not ms or ms[0] is None:
        return "---"
    return f"{ms[0]:.3f}$\\pm${ms[1]:.3f}"

lines = [
    "% Auto-filled by scripts/run_do3_rl_bakeoff.sh — tab:rl_bakeoff",
    f"% gate_pass={passed} suffix={suffix}",
]
for label, key in [("Static rule", "static"), ("PPO (unmasked)", "ppo_unmasked"), ("Maskable PPO", "maskable_ppo")]:
    a = summary["arms"][key]
    lines.append(
        f"{label} & {fmt(a['precision_mean_std'])} & {fmt(a['invalid_mean_std'])} & "
        f"{fmt(a['reward_mean_std'])} & {fmt(a['intervention_mean_std'])} & "
        f"{fmt(a['human_rev_mean_std'])} \\\\"
    )
paper_path.write_text("\n".join(lines) + "\n")
print(json.dumps(gate, indent=2))
print(f"Wrote {gate_path}")
print(f"Wrote {summary_path}")
print(f"Wrote {paper_path}")
sys.exit(0 if passed else 1)
PY

echo "Done. Review $GATE $SUMMARY $PAPER" | tee -a "$LOG"
