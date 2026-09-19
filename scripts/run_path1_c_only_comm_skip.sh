#!/usr/bin/env bash
# Path 1A: B1 / Bc (C-only) / B2 under frozen hard attack (comm_skip).
# Never omits --metrics-suffix (default path1_c_only). Does not overwrite bare H1 filenames.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-./.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

ROUNDS="${ROUNDS:-30}"
SEEDS="${SEEDS:-43 44 45 46 47}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
FRAC="${FRAC:-0.25}"
POISON_MODE="${POISON_MODE:-comm_skip}"
DATASET=iomt_natural
SUFFIX="${SUFFIX:-path1_c_only}"
ARMS="${ARMS:-b1 bc b2}"
DRY_RUN="${DRY_RUN:-0}"
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

if [[ -z "$SUFFIX" ]]; then
  echo "ERROR: SUFFIX must be non-empty (refuse bare H1 metric names)" >&2
  exit 1
fi
if [[ "$SUFFIX" == "ecu_repl" ]]; then
  echo "ERROR: refuse ECU primary suffix" >&2
  exit 1
fi

echo "=== Path1 C-only matrix mode=$POISON_MODE f=$FRAC seeds=[$SEEDS] arms=[$ARMS] suffix=$SUFFIX ==="

extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
       --adversary-fraction "$FRAC" --poison-mode "$POISON_MODE"
       --metrics-suffix "$SUFFIX")

n_cmd=0
for seed in $SEEDS; do
  for base in $ARMS; do
    n_cmd=$((n_cmd + 1))
    cmd=("$PY" run_experiments.py regression --dataset "$DATASET" --baseline "$base"
         --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}")
    echo "---- [$n_cmd] seed $seed : $base ----"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY:'; printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
  done
done

echo "Done $n_cmd runs (or dry-run lines). Gate: scripts/eval_path1_c_only_gate.py (WP-P4)."
