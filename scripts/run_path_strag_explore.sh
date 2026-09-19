#!/usr/bin/env bash
# Path STRAG T1.2: seed-42 explore B1 / Bc / B2 under benign straggler + malicious comm_skip.
# Does not overwrite H1 / Path1 / Path AGG / hard adversary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-30}"
TA="${TA:-20}"
ADV="${ADV:-config/iomt_natural_adversary_path_strag.json}"
SUFF="${SUFF:-path_strag_explore}"
Q="${Q:-}"  # optional override of skip_prob_q

echo "=== Path STRAG explore seed=$SEED rounds=$ROUNDS ta=$TA q=${Q:-config} suffix=$SUFF ==="
for BASE in b1 bc b2; do
  echo "---- $BASE ----"
  CMD=(
    "$PY" run_experiments.py regression
    --dataset iomt_natural
    --baseline "$BASE"
    --seed "$SEED"
    --num-rounds "$ROUNDS"
    --compromise-round "$TA"
    --late-compromise-config "$ADV"
    --uniform-b-prior
    --metrics-suffix "$SUFF"
  )
  if [[ -n "$Q" ]]; then
    CMD+=(--benign-skip-prob "$Q")
  fi
  "${CMD[@]}"
done
echo "=== explore runs finished; summarize with scripts/eval_path_strag_explore.py ==="
