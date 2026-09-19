#!/usr/bin/env bash
# Path AGG WP-A3/A4: confirmatory B0 / B2 / Rm under frozen M*=label_flip.
# Never omits --metrics-suffix path_agg. Does not overwrite H1 / Path1 / ecu / hard adversary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PYTHON:-python3}"
if [[ -n "${TRUSTFED_PYTHON:-}" ]]; then PY="$TRUSTFED_PYTHON"; fi

ROUNDS="${ROUNDS:-30}"
SEEDS="${SEEDS:-43 44 45 46 47}"
COMPROMISE_ROUND="${COMPROMISE_ROUND:-20}"
FLIP_P="${FLIP_P:-0.8}"
FRAC="${FRAC:-0.25}"
# Frozen M* (WP-A2) — do not change without re-explore + plan update
POISON_MODE="${POISON_MODE:-label_flip}"
DATASET=iomt_natural
SUFFIX="${SUFFIX:-path_agg}"
ARMS="${ARMS:-fedavg b2 rm}"
DRY_RUN="${DRY_RUN:-0}"
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"

if [[ -z "$SUFFIX" ]]; then
  echo "ERROR: SUFFIX must be non-empty (refuse bare metric names)" >&2
  exit 1
fi
if [[ "$SUFFIX" == "ecu_repl" || "$SUFFIX" == "path1_c_only" ]]; then
  echo "ERROR: refuse locked Path1/ECU suffixes for Path AGG" >&2
  exit 1
fi
if [[ "$POISON_MODE" != "label_flip" ]]; then
  echo "ERROR: confirmatory Path AGG requires POISON_MODE=label_flip (frozen M*); got $POISON_MODE" >&2
  exit 1
fi
if [[ "$SUFFIX" != "path_agg" && "$DRY_RUN" != "1" ]]; then
  echo "WARNING: SUFFIX=$SUFFIX (expected path_agg for confirmatory)" >&2
fi

# Refuse confirmatory seed 42 (explore-only)
for s in $SEEDS; do
  if [[ "$s" == "42" ]]; then
    echo "ERROR: seed 42 is explore-only; confirmatory seeds must be 43-47" >&2
    exit 1
  fi
done

echo "=== Path AGG confirmatory mode=$POISON_MODE f=$FRAC t_a=$COMPROMISE_ROUND rounds=$ROUNDS ==="
echo "    seeds=[$SEEDS] arms=[$ARMS] suffix=$SUFFIX dry_run=$DRY_RUN"

n_jobs=0
for seed in $SEEDS; do
  for base in $ARMS; do
    n_jobs=$((n_jobs + 1))
    extra=(
      --dataset "$DATASET"
      --baseline "$base"
      --seed "$seed"
      --num-rounds "$ROUNDS"
      --compromise-round "$COMPROMISE_ROUND"
      --flip-p "$FLIP_P"
      --adversary-fraction "$FRAC"
      --poison-mode "$POISON_MODE"
      --metrics-suffix "$SUFFIX"
    )
    # B2 / trust arms: match H1 uniform prior; harmless on fedavg/rm (warned+ignored)
    if [[ "$base" == "b2" || "$base" == "b1" || "$base" == "bc" ]]; then
      extra+=(--uniform-b-prior)
    fi
    cmd=("$PY" run_experiments.py regression "${extra[@]}")
    echo "---- [$n_jobs] baseline=$base seed=$seed ----"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY:'; printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
  done
done

echo "=== Path AGG matrix finished ($n_jobs jobs) suffix=$SUFFIX ==="
if [[ "$DRY_RUN" != "1" ]]; then
  echo "Next: python3 scripts/eval_path_agg_gate.py"
fi
