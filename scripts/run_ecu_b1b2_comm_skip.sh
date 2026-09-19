#!/usr/bin/env bash
# =============================================================================
# ECU-IoHT cross-dataset B1 vs B2 rematch (plan WP2)
# =============================================================================
# Directional replication of trust discrimination under frozen comm_skip.
# Does NOT overwrite locked CICIoMT iomt_natural data or H1 gates.
# Clients are constructed Source silos (NOT hospitals).
#
# Usage:
#   bash scripts/run_ecu_b1b2_comm_skip.sh
#   SEEDS="43 44 45" ROUNDS=30 bash scripts/run_ecu_b1b2_comm_skip.sh
#   SKIP_CONVERT=1 bash scripts/run_ecu_b1b2_comm_skip.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
# Re-read plan rule: execute WP → recheck → adjust
# Plan: docs/ECU_CROSS_DATASET_REPLICATION_PLAN.md

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

DATA="${DATA:-data/CSVs/ecu_ioht_clients}"
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
METRICS="${METRICS:-results/trustfed_agent/metrics}"
SEEDS="${SEEDS:-43 44 45}"
ROUNDS="${ROUNDS:-30}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-comm_skip}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
SUFFIX="${SUFFIX:-ecu_repl}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$REV_SUP" "$METRICS"
TAG="${POISON_MODE}_f${FRAC}"
LOG="$REV_SUP/ecu_h1_${TAG}.log"
GATE="$REV_SUP/ecu_h1_${TAG}_gate.json"
SUMMARY="$REV_SUP/ecu_h1_${TAG}_summary.json"

echo "===== ECU B1/B2 rematch data=$DATA mode=$POISON_MODE =====" | tee "$LOG"

if [[ "$SKIP_CONVERT" != "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] convert ECU clients" | tee -a "$LOG"
  else
    "$PY" scripts/convert_ecu_ioht_to_clients.py --out-dir "$DATA" | tee -a "$LOG"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry] skip runs" | tee -a "$LOG"
  exit 0
fi

if [[ ! -d "$DATA/clients" ]]; then
  echo "ERROR: missing $DATA/clients" | tee -a "$LOG"
  exit 1
fi

# Safety: refuse if pointed at locked CICIoMT natural tree
if [[ "$DATA" == *iomt_natural* ]]; then
  echo "ERROR: refusing to use iomt_natural path for ECU rematch" | tee -a "$LOG"
  exit 1
fi

extra=(--compromise-round "$COMPROMISE_ROUND" --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE")
pct_tag="$("$PY" -c "print(int(round(100*float('$FRAC'))))")"

for seed in $SEEDS; do
  b1_m="$METRICS/run_trustfed_b1v_uniform_iomt_natural_poison${pct_tag}_${POISON_MODE}_${SUFFIX}_seed_${seed}.json"
  b2_m="$METRICS/run_trustfed_b2_uniform_iomt_natural_poison${pct_tag}_${POISON_MODE}_${SUFFIX}_seed_${seed}.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$b1_m" ]]; then
    echo "---- skip B1 seed $seed ----" | tee -a "$LOG"
  else
    echo "---- B1 seed $seed ----" | tee -a "$LOG"
    "$PY" run_experiments.py regression --dataset iomt_natural --baseline b1 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior \
      --data-dir "$DATA/clients" \
      --test-csv "$DATA/ecu_test_set.csv" \
      --metrics-suffix "$SUFFIX" \
      "${extra[@]}"
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$b2_m" ]]; then
    echo "---- skip B2 seed $seed ----" | tee -a "$LOG"
  else
    echo "---- B2 seed $seed ----" | tee -a "$LOG"
    "$PY" run_experiments.py regression --dataset iomt_natural --baseline b2 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior \
      --data-dir "$DATA/clients" \
      --test-csv "$DATA/ecu_test_set.csv" \
      --metrics-suffix "$SUFFIX" \
      "${extra[@]}"
  fi
done

"$PY" - "$METRICS" "$SEEDS" "$FRAC" "$POISON_MODE" "$SUFFIX" "$GATE" "$SUMMARY" "$DATA/manifest.json" <<'PY' | tee -a "$LOG"
import json, sys, statistics
from pathlib import Path

metrics_dir = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
frac = float(sys.argv[3])
mode = sys.argv[4]
suffix = sys.argv[5]
gate_path, summary_path, man_path = map(Path, sys.argv[6:9])
pct = int(round(100 * frac))
man = json.loads(man_path.read_text())

def auroc(path):
    rec = json.loads(path.read_text())
    disc = rec.get("trust_discrimination") or {}
    if disc.get("auroc") is not None:
        return float(disc["auroc"])
    return None

pairs, missing = [], []
for seed in seeds:
    b1 = metrics_dir / f"run_trustfed_b1v_uniform_iomt_natural_poison{pct}_{mode}_{suffix}_seed_{seed}.json"
    b2 = metrics_dir / f"run_trustfed_b2_uniform_iomt_natural_poison{pct}_{mode}_{suffix}_seed_{seed}.json"
    if not b1.exists() or not b2.exists():
        missing.append({"seed": seed, "b1": str(b1), "b2": str(b2)})
        continue
    a1, a2 = auroc(b1), auroc(b2)
    pairs.append({
        "seed": seed, "b1_auroc": a1, "b2_auroc": a2,
        "delta": (a2 - a1) if a1 is not None and a2 is not None else None,
    })

complete = [p for p in pairs if p["delta"] is not None]
mean_b1 = statistics.mean(p["b1_auroc"] for p in complete) if complete else None
mean_b2 = statistics.mean(p["b2_auroc"] for p in complete) if complete else None
mean_d = statistics.mean(p["delta"] for p in complete) if complete else None
n_pos = sum(1 for p in complete if p["delta"] > 0)
# Directional pass (plan): not require AUROC=1.0
checks = {
    "b1_weaker_than_b2_mean": mean_b1 is not None and mean_b2 is not None and mean_b1 < mean_b2,
    "delta_positive_majority": complete and n_pos >= max(1, (len(complete) + 1) // 2),
    "mean_delta_positive": mean_d is not None and mean_d > 0,
}
passed = bool(complete and checks["b1_weaker_than_b2_mean"] and checks["delta_positive_majority"] and checks["mean_delta_positive"])
gate = {
    "pass": passed,
    "claim": "independent cross-dataset replication (directional); not multi-hospital validation",
    "n_complete": len(complete),
    "mean_auroc_b1": mean_b1,
    "mean_auroc_b2": mean_b2,
    "std_auroc_b1": statistics.stdev(p["b1_auroc"] for p in complete) if len(complete) > 1 else 0.0,
    "std_auroc_b2": statistics.stdev(p["b2_auroc"] for p in complete) if len(complete) > 1 else 0.0,
    "mean_delta": mean_d,
    "n_delta_positive": n_pos,
    "checks": checks,
    "pairs": pairs,
    "missing": missing,
    "manifest_claim": man.get("claim"),
    "n_clients": man.get("n_clients"),
}
summary = {"gate": gate, "manifest": man, "metrics_suffix": suffix, "note": "locked CICIoMT H1 untouched"}
gate_path.write_text(json.dumps(gate, indent=2))
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(gate, indent=2))
sys.exit(0)
PY

echo "Done. Review $GATE (gate['pass'] = directional verdict)" | tee -a "$LOG"
