#!/usr/bin/env bash
# Path STRAG T1.3/T1.4: confirmatory B1 / Bc / B2 under frozen q=0.5 benign straggler + malicious comm_skip.
# Never omits --metrics-suffix path_strag. Does not overwrite H1 / Path1 / Path AGG / hard adversary / iomt_natural.
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
FRAC="${FRAC:-0.25}"
DATASET=iomt_natural
ADV="${ADV:-config/iomt_natural_adversary_path_strag.json}"
# Frozen T1.2 explore — do not change without re-explore + plan update
Q="${Q:-0.5}"
SUFFIX="${SUFFIX:-path_strag}"
ARMS="${ARMS:-b1 bc b2}"
DRY_RUN="${DRY_RUN:-0}"
REV_SUP="${REV_SUP:-results/trustfed_agent/rev_sup}"
mkdir -p "$REV_SUP"
DRY_LOG="$REV_SUP/path_strag_t1_3_dryrun.log"

if [[ -z "$SUFFIX" ]]; then
  echo "ERROR: SUFFIX must be non-empty (refuse bare metric names)" >&2
  exit 1
fi
if [[ "$SUFFIX" != "path_strag" ]]; then
  echo "ERROR: confirmatory Path STRAG requires SUFFIX=path_strag (got '$SUFFIX')" >&2
  exit 1
fi
if [[ "$ADV" != "config/iomt_natural_adversary_path_strag.json" && "$ADV" != "$ROOT/config/iomt_natural_adversary_path_strag.json" ]]; then
  echo "ERROR: confirmatory requires ADV=config/iomt_natural_adversary_path_strag.json (got '$ADV')" >&2
  exit 1
fi
if [[ ! -f "$ADV" ]]; then
  echo "ERROR: missing adversary config $ADV" >&2
  exit 1
fi
# Locked q for confirmatory
python3 - <<PY
import json, sys
from pathlib import Path
q_env = float("$Q")
if abs(q_env - 0.5) > 1e-12:
    print(f"ERROR: confirmatory Q must be 0.5 (frozen T1.2); got {q_env}", file=sys.stderr)
    sys.exit(1)
raw = json.loads(Path("$ADV").read_text())
bs = raw.get("benign_straggler") or {}
q = float(bs.get("skip_prob_q", -1))
assert abs(q - 0.5) < 1e-12, q
assert bs.get("enabled") is True
assert raw.get("poison_mode") == "comm_skip"
assert float(raw.get("adversary_fraction") or 0) == float("$FRAC")
assert int(raw.get("compromise_round") or 0) == int("$COMPROMISE_ROUND")
hard = json.loads(Path("config/iomt_natural_adversary_hard.json").read_text())
assert hard.get("status") == "frozen_primary_hard_attack"
assert "benign_straggler" not in hard
print(f"config OK: poison=comm_skip f=$FRAC q=0.5 ta=$COMPROMISE_ROUND; hard untouched")
PY

# Refuse explore seed 42; allow only 43-47 for default confirmatory
seed_arr=($SEEDS)
arm_arr=($ARMS)
for s in "${seed_arr[@]}"; do
  if [[ "$s" == "42" ]]; then
    echo "ERROR: seed 42 is explore-only; confirmatory seeds must be 43-47" >&2
    exit 1
  fi
done
for a in "${arm_arr[@]}"; do
  if [[ "$a" != "b1" && "$a" != "bc" && "$a" != "b2" ]]; then
    echo "ERROR: confirmatory arms must be subset of {b1,bc,b2}; got '$a'" >&2
    exit 1
  fi
done

expected_jobs=$(( ${#seed_arr[@]} * ${#arm_arr[@]} ))
echo "=== Path STRAG confirmatory q=0.5 f=$FRAC t_a=$COMPROMISE_ROUND rounds=$ROUNDS ==="
echo "    seeds=[${SEEDS}] arms=[${ARMS}] suffix=$SUFFIX adv=$ADV dry_run=$DRY_RUN expected_jobs=$expected_jobs"

n_jobs=0
if [[ "$DRY_RUN" == "1" ]]; then
  : > "$DRY_LOG"
fi

for seed in "${seed_arr[@]}"; do
  for base in "${arm_arr[@]}"; do
    n_jobs=$((n_jobs + 1))
    cmd=(
      "$PY" run_experiments.py regression
      --dataset "$DATASET"
      --baseline "$base"
      --seed "$seed"
      --num-rounds "$ROUNDS"
      --compromise-round "$COMPROMISE_ROUND"
      --adversary-fraction "$FRAC"
      --poison-mode comm_skip
      --late-compromise-config "$ADV"
      --benign-skip-prob 0.5
      --uniform-b-prior
      --metrics-suffix path_strag
    )
    echo "---- [$n_jobs] baseline=$base seed=$seed ----"
    if [[ "$DRY_RUN" == "1" ]]; then
      dry_line="DRY: $(printf '%q ' "${cmd[@]}")"
      echo "$dry_line"
      echo "$dry_line" >> "$DRY_LOG"
    else
      "${cmd[@]}"
    fi
  done
done

echo "=== Path STRAG matrix finished ($n_jobs jobs) suffix=path_strag ==="
if [[ "$n_jobs" -ne "$expected_jobs" ]]; then
  echo "ERROR: job count $n_jobs != expected $expected_jobs" >&2
  exit 1
fi
if [[ "$SEEDS" == "43 44 45 46 47" && "$ARMS" == "b1 bc b2" && "$n_jobs" -ne 15 ]]; then
  echo "ERROR: default confirmatory matrix must be 15 jobs; got $n_jobs" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  "$PY" scripts/qa_path_strag_dryrun.py --log "$DRY_LOG"
  echo "T1.3 dry-run QA PASS ($n_jobs jobs); log=$DRY_LOG"
else
  echo "Next: python3 scripts/eval_path_strag_gate.py"
fi
