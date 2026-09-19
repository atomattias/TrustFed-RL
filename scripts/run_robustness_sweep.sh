#!/usr/bin/env bash
# Robustness sweep (supervisor plan v3): B2 mandatory; multi-seed confirmatory default.
#
# Usage:
#   bash scripts/run_robustness_sweep.sh
#   SEEDS="43 44 45 46 47" MODES=sign_flip INCLUDE_B2=1 bash scripts/run_robustness_sweep.sh
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
FRACTIONS="${FRACTIONS:-0.1 0.2 0.3 0.4}"
MODES="${MODES:-label_flip}"
ON_OFF_ATTACK_ROUNDS="${ON_OFF_ATTACK_ROUNDS:-5}"
INCLUDE_B0="${INCLUDE_B0:-1}"
INCLUDE_B1="${INCLUDE_B1:-1}"
INCLUDE_B2="${INCLUDE_B2:-1}"
INCLUDE_B3="${INCLUDE_B3:-0}"
INCLUDE_B4="${INCLUDE_B4:-0}"
INCLUDE_B5="${INCLUDE_B5:-1}"
INCLUDE_B5R="${INCLUDE_B5R:-0}"
DATASET=iomt_natural
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

if [[ ! -d data/CSVs/iomt_natural/clients ]]; then
  echo "Missing natural clients — running converter..."
  "$PY" scripts/convert_iomt_natural_clients.py
fi

echo "=== robustness sweep rounds=$ROUNDS t_a=$COMPROMISE_ROUND fracs=[$FRACTIONS] modes=[$MODES] seeds=[$SEEDS] B0=$INCLUDE_B0 B1=$INCLUDE_B1 B2=$INCLUDE_B2 B5=$INCLUDE_B5 ==="

run_pair() {
  local seed="$1" frac="$2" mode="$3"
  local extra=(--compromise-round "$COMPROMISE_ROUND" --flip-p "$FLIP_P"
               --adversary-fraction "$frac" --poison-mode "$mode")
  if [[ "$mode" == "on_off" || "$mode" == "label_flip_on_off" ]]; then
    extra+=(--on-off-attack-rounds "$ON_OFF_ATTACK_ROUNDS")
  fi

  if [[ "$INCLUDE_B0" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B0 FedAvg ----"
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline fedavg \
      --seed "$seed" --num-rounds "$ROUNDS" "${extra[@]}"
  fi

  if [[ "$INCLUDE_B1" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B1 V-only ----"
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b1 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  fi

  if [[ "$INCLUDE_B2" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B2 behavioural ----"
    "$PY" run_experiments.py regression --dataset "$DATASET" --baseline b2 \
      --seed "$seed" --num-rounds "$ROUNDS" --uniform-b-prior "${extra[@]}"
  fi

  if [[ "$INCLUDE_B3" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B3 static+gov ----"
    "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach trustfed_governance \
      --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior --no-rl \
      "${extra[@]}"
  fi

  if [[ "$INCLUDE_B4" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B4 RL-only ----"
    "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach trustfed_rl_only \
      --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior --no-governance \
      "${extra[@]}"
  fi

  if [[ "$INCLUDE_B5" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B5 full ----"
    "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach trustfed_agent \
      --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior \
      "${extra[@]}"
  fi

  if [[ "$INCLUDE_B5R" == "1" ]]; then
    echo "---- seed=$seed frac=$frac mode=$mode : B5-R (R∈T) ----"
    "$PY" trustfed_agent_runner.py --dataset "$DATASET" --approach trustfed_agent \
      --random-state "$seed" --num-rounds "$ROUNDS" --uniform-b-prior --include-r-in-t \
      "${extra[@]}"
  fi
}

for seed in $SEEDS; do
  for frac in $FRACTIONS; do
    for mode in $MODES; do
      run_pair "$seed" "$frac" "$mode"
    done
  done
done

MODE_TAG=$(echo "$MODES" | tr ' ' '_')
echo "=== Analyzing ==="
"$PY" scripts/analyze_iomt_natural_trust.py \
  --input results/trustfed_agent/metrics \
  --export "$REV_SUP/iomt_natural_robustness_summary_${MODE_TAG}.json"

echo "Done. Summary: $REV_SUP/iomt_natural_robustness_summary_${MODE_TAG}.json"
