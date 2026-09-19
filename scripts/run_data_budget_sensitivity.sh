#!/usr/bin/env bash
# =============================================================================
# WP-B: data-budget sensitivity — B1 vs B2 under frozen comm_skip
# =============================================================================
# Rebuilds natural clients at larger per-client row budgets into a SEPARATE
# directory (does not overwrite locked baseline data/CSVs/iomt_natural).
# Note: raising pool caps also changes the 70/15/15 split sizes, so this is a
# "larger client budget / alternate export" sensitivity — not identical D_test.
#
# Pass: same H1 direction (B1 imperfect, B2 ≫ B1).
#
# Usage:
#   DRY_RUN=1 bash scripts/run_data_budget_sensitivity.sh
#   bash scripts/run_data_budget_sensitivity.sh
#   SEEDS="43 45 47" bash scripts/run_data_budget_sensitivity.sh
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
OUT_DATA="${OUT_DATA:-data/CSVs/iomt_natural_budget2x}"
SEEDS="${SEEDS:-43 44 45 46 47}"
ROUNDS="${ROUNDS:-30}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-comm_skip}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
# Grow scarce-family pool caps so mix-faithful sizing can assign ~2x rows/client.
# (Utilization % may fall if the train pool grows faster than assignment;
#  the scientific target is larger per-client budget under the same CLIENT_MIX,
#  not gaming utilization by dropping surplus families from the denominator.)
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-20000}"
MAX_ROWS_BENIGN="${MAX_ROWS_BENIGN:-120000}"
MAX_ROWS_ARP="${MAX_ROWS_ARP:-40000}"
MAX_ROWS_PER_CLIENT="${MAX_ROWS_PER_CLIENT:-50000}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "$REV_SUP"
TAG="f${FRAC}"
LOG="$REV_SUP/budget2x_${TAG}.log"
GATE="$REV_SUP/budget2x_${TAG}_gate.json"
SUMMARY="$REV_SUP/budget2x_${TAG}_summary.json"

echo "===== data-budget sensitivity out=$OUT_DATA mode=$POISON_MODE =====" | tee -a "$LOG"

if [[ "$SKIP_CONVERT" != "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] convert -> $OUT_DATA (caps file=$MAX_ROWS_PER_FILE client=$MAX_ROWS_PER_CLIENT)" | tee -a "$LOG"
  else
    "$PY" scripts/convert_iomt_natural_clients.py \
      --output-dir "$OUT_DATA" \
      --max-rows-per-file "$MAX_ROWS_PER_FILE" \
      --max-rows-per-client "$MAX_ROWS_PER_CLIENT" \
      --max-rows-benign "$MAX_ROWS_BENIGN" \
      --max-rows-arp "$MAX_ROWS_ARP" \
      | tee -a "$LOG"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry] skip B1/B2 runs" | tee -a "$LOG"
  exit 0
fi

if [[ ! -d "$OUT_DATA/clients" ]]; then
  echo "ERROR: missing $OUT_DATA/clients" | tee -a "$LOG"
  exit 1
fi

UTIL="$("$PY" -c "import json; print(json.load(open('$OUT_DATA/manifest.json'))['counts']['train_utilization'])")"
echo "[info] new train_utilization=$UTIL" | tee -a "$LOG"

extra=(--compromise-round "$COMPROMISE_ROUND" --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")

pct_tag="$("$PY" -c "print(int(round(100*float('$FRAC'))))")"
for seed in $SEEDS; do
  b1_m="$METRICS/run_trustfed_b1v_uniform_iomt_natural_poison${pct_tag}_${POISON_MODE}_budget2x_seed_${seed}.json"
  b2_m="$METRICS/run_trustfed_b2_uniform_iomt_natural_poison${pct_tag}_${POISON_MODE}_budget2x_seed_${seed}.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$b1_m" ]]; then
    echo "---- skip B1 seed $seed (exists) ----" | tee -a "$LOG"
  else
    echo "---- B1 seed $seed ----" | tee -a "$LOG"
    "$PY" run_experiments.py regression --dataset iomt_natural --baseline b1 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior \
      --data-dir "$OUT_DATA/clients" \
      --test-csv "$OUT_DATA/iomt_test_set.csv" \
      --metrics-suffix budget2x \
      "${extra[@]}"
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$b2_m" ]]; then
    echo "---- skip B2 seed $seed (exists) ----" | tee -a "$LOG"
  else
    echo "---- B2 seed $seed ----" | tee -a "$LOG"
    "$PY" run_experiments.py regression --dataset iomt_natural --baseline b2 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior \
      --data-dir "$OUT_DATA/clients" \
      --test-csv "$OUT_DATA/iomt_test_set.csv" \
      --metrics-suffix budget2x \
      "${extra[@]}"
  fi
done

# Aggregate AUROC from budget2x-tagged metrics (do not touch locked H1 files).
"$PY" - "$METRICS" "$SEEDS" "$FRAC" "$POISON_MODE" "$GATE" "$SUMMARY" "$OUT_DATA/manifest.json" <<'PY' | tee -a "$LOG"
import json, sys, statistics
from pathlib import Path

metrics_dir = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
frac = float(sys.argv[3])
mode = sys.argv[4]
gate_path, summary_path, man_path = map(Path, sys.argv[5:8])
pct = int(round(100 * frac))
man = json.loads(man_path.read_text())

def auroc(path):
    rec = json.loads(path.read_text())
    disc = rec.get("trust_discrimination") or {}
    if disc.get("auroc") is not None:
        return float(disc["auroc"])
    return None

pairs = []
missing = []
for seed in seeds:
    b1 = metrics_dir / f"run_trustfed_b1v_uniform_iomt_natural_poison{pct}_{mode}_budget2x_seed_{seed}.json"
    b2 = metrics_dir / f"run_trustfed_b2_uniform_iomt_natural_poison{pct}_{mode}_budget2x_seed_{seed}.json"
    if not b1.exists() or not b2.exists():
        missing.append({"seed": seed, "b1": str(b1), "b2": str(b2)})
        continue
    a1, a2 = auroc(b1), auroc(b2)
    pairs.append({"seed": seed, "b1_auroc": a1, "b2_auroc": a2, "delta": (a2 - a1) if a1 is not None and a2 is not None else None})

complete = [p for p in pairs if p["delta"] is not None]
mean_b1 = statistics.mean(p["b1_auroc"] for p in complete) if complete else None
mean_b2 = statistics.mean(p["b2_auroc"] for p in complete) if complete else None
mean_d = statistics.mean(p["delta"] for p in complete) if complete else None
n_pos = sum(1 for p in complete if p["delta"] > 0)
checks = {
    "b1_not_perfect": mean_b1 is not None and mean_b1 < 1.0,
    "delta_ge_min": mean_d is not None and mean_d >= 0.05,
    "sign_consistency": complete and n_pos >= max(1, int(0.8 * len(complete))),
    "or_b2_high": mean_b2 is not None and mean_b2 >= 0.95,
}
passed = bool(
    complete
    and checks["b1_not_perfect"]
    and (checks["delta_ge_min"] or checks["or_b2_high"])
    and checks["sign_consistency"]
)
gate = {
    "pass": passed,
    "n_complete": len(complete),
    "mean_auroc_b1": mean_b1,
    "mean_auroc_b2": mean_b2,
    "mean_delta": mean_d,
    "n_delta_positive": n_pos,
    "checks": checks,
    "pairs": pairs,
    "missing": missing,
    "sensitivity_utilization": man.get("counts", {}).get("train_utilization"),
    "baseline_utilization": 0.25672688491742884,
}
summary = {"gate": gate, "manifest_counts": man.get("counts"), "note": "budget2x metrics suffix; locked H1 untouched"}
gate_path.write_text(json.dumps(gate, indent=2))
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(gate, indent=2))
# Always exit 0 after writing artifacts; gate["pass"] is the scientific verdict.
# (Incomplete runs => pass=false; do not treat missing seeds as success.)
sys.exit(0)
PY

echo "Done. Review $GATE $SUMMARY (see gate['pass'])" | tee -a "$LOG"
