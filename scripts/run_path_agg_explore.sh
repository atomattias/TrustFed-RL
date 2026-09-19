#!/usr/bin/env bash
# Path AGG WP-A2: seed-42 explore for B0 / B2 / Rm under continuing-upload poison.
# Does not overwrite H1 / Path1 / locked iomt_natural data.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-30}"
MODE="${MODE:-sign_flip}"
FRAC="${FRAC:-0.25}"
TA="${TA:-20}"
SUFF="${SUFF:-path_agg_explore}"
SCALE="${SIGN_FLIP_SCALE:-}"

echo "=== Path AGG explore seed=$SEED mode=$MODE rounds=$ROUNDS suffix=$SUFF ==="
for BASE in fedavg b2 rm; do
  echo "---- $BASE ----"
  CMD=(
    "$PY" run_experiments.py regression
    --dataset iomt_natural
    --baseline "$BASE"
    --seed "$SEED"
    --num-rounds "$ROUNDS"
    --poison-mode "$MODE"
    --adversary-fraction "$FRAC"
    --compromise-round "$TA"
    --metrics-suffix "$SUFF"
    --uniform-b-prior
  )
  # Optional stronger flip via env (applied only if LateCompromiseConfig supports CLI later;
  # for now scale is in config file — re-write scale into a temp note if needed)
  "${CMD[@]}"
done
echo "=== explore runs finished; summarize with scripts/eval_path_agg_explore.py ==="
