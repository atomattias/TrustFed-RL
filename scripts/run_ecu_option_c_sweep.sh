#!/usr/bin/env bash
# =============================================================================
# ECU Option C — partition × attacker B1/B2 sensitivity sweep
# Plan: docs/ECU_OPTION_C_PARTITION_ATTACKER_PLAN.md (WP-C2/C3)
# QA:   docs/ECU_OPTION_C_QA_CHECKLIST.md
# =============================================================================
# Light (default): p0/p1/p2 × a0108/a0207 = 6 cells × 2 arms = 12 runs
# Full:            + a0306 = 9 cells × 2 = 18 runs
#
# Usage:
#   bash scripts/run_ecu_option_c_sweep.sh
#   PROFILE=full bash scripts/run_ecu_option_c_sweep.sh
#   DRY_RUN=1 bash scripts/run_ecu_option_c_sweep.sh
#   SKIP_EXISTING=1 bash scripts/run_ecu_option_c_sweep.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

PROFILE="${PROFILE:-light}"
ROUNDS="${ROUNDS:-30}"
SEED="${SEED:-43}"
POISON_MODE="${POISON_MODE:-comm_skip}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
METRICS="${METRICS:-results/trustfed_agent/metrics}"

mkdir -p "$REV_SUP" "$METRICS"
LOG="$REV_SUP/ecu_option_c_${PROFILE}.log"

PARTITIONS=(p0 p1 p2)
if [[ "$PROFILE" == "light" ]]; then
  ATTACKERS=(a0108 a0207)
elif [[ "$PROFILE" == "full" ]]; then
  ATTACKERS=(a0108 a0207 a0306)
else
  echo "ERROR: PROFILE must be light|full (got $PROFILE)" | tee -a "$LOG"
  exit 1
fi

attacker_ids_for() {
  case "$1" in
    a0108) echo "client_01,client_08" ;;
    a0207) echo "client_02,client_07" ;;
    a0306) echo "client_03,client_06" ;;
    *) echo "ERROR: unknown attacker tag $1" >&2; return 1 ;;
  esac
}

echo "===== ECU Option C sweep profile=$PROFILE seed=$SEED rounds=$ROUNDS =====" | tee "$LOG"

for P in "${PARTITIONS[@]}"; do
  DATA="data/CSVs/ecu_ioht_clients_c_${P}"
  if [[ "$DATA" == *iomt_natural* ]]; then
    echo "ERROR: refusing iomt_natural path" | tee -a "$LOG"
    exit 1
  fi
  if [[ ! -d "$DATA/clients" ]]; then
    echo "ERROR: missing $DATA/clients (run WP-C1 first)" | tee -a "$LOG"
    exit 1
  fi
  for A in "${ATTACKERS[@]}"; do
    IDS="$(attacker_ids_for "$A")"
    SUFF="ecu_c_${P}_${A}"
    if [[ "$SUFF" == *ecu_repl* ]]; then
      echo "ERROR: refusing locked ecu_repl suffix" | tee -a "$LOG"
      exit 1
    fi
    for BASE in b1 b2; do
      # Expected metrics basename fragment (fraction_tag may be poison25 for 2/8)
      echo "---- cell P=$P A=$A arm=$BASE ids=$IDS suffix=$SUFF ----" | tee -a "$LOG"
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] $PY run_experiments.py regression --baseline $BASE --attacker-ids $IDS --metrics-suffix $SUFF --data-dir $DATA/clients" | tee -a "$LOG"
        continue
      fi
      # Skip if any matching metrics file already exists for this suffix+seed+arm
      if [[ "$SKIP_EXISTING" == "1" ]]; then
        shopt -s nullglob
        existing=("$METRICS"/run_trustfed_${BASE}*_iomt_natural_*_${SUFF}_seed_${SEED}.json)
        # b1 writes as b1v
        if [[ "$BASE" == "b1" ]]; then
          existing=("$METRICS"/run_trustfed_b1v*_iomt_natural_*_${SUFF}_seed_${SEED}.json)
        fi
        shopt -u nullglob
        if (( ${#existing[@]} > 0 )); then
          echo "skip existing ${existing[0]}" | tee -a "$LOG"
          continue
        fi
      fi
      # IMPORTANT: do NOT pass --adversary-fraction (would override attacker-ids)
      "$PY" run_experiments.py regression --dataset iomt_natural --baseline "$BASE" \
        --seed "$SEED" --num-rounds "$ROUNDS" --uniform-b-prior \
        --data-dir "$DATA/clients" \
        --test-csv "$DATA/ecu_test_set.csv" \
        --metrics-suffix "$SUFF" \
        --poison-mode "$POISON_MODE" \
        --compromise-round "$COMPROMISE_ROUND" \
        --attacker-ids "$IDS" \
        2>&1 | tee -a "$LOG"
    done
  done
done

echo "===== Option C sweep finished profile=$PROFILE =====" | tee -a "$LOG"
