#!/usr/bin/env bash
# Prepare CICIoMT2024 federated hospital client CSVs for TrustFed-Agent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_DIR="${CICIOMT_SOURCE:-$ROOT/data/train}"
TEST_DIR="${CICIOMT_TEST:-$ROOT/data/test}"
OUTPUT_DIR="${IOMT_OUTPUT_DIR:-$ROOT/data/CSVs/iomt_clients}"
TEST_CSV="${IOMT_TEST_CSV:-$ROOT/data/CSVs/iomt_test_set.csv}"

if [[ "${1:-}" == "--demo" ]]; then
  echo "=== Generating synthetic IoMT demo hospital clients ==="
  python3 scripts/convert_iomt_to_hospital_clients.py \
    --demo \
    --output-dir "$OUTPUT_DIR" \
    --test-output "$TEST_CSV"
  exit 0
fi

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "Train directory not found: $TRAIN_DIR"
  echo "Place CICIoMT CSVs in TrustFed-Agent/data/train and data/test"
  exit 1
fi

echo "=== Converting CICIoMT2024 ==="
echo "  train: $TRAIN_DIR"
echo "  test:  $TEST_DIR"

ARGS=(
  --source-dir "$TRAIN_DIR"
  --output-dir "$OUTPUT_DIR"
  --test-output "$TEST_CSV"
  --num-hospitals 12
  --partition protocol
  --max-rows-per-file 5000
  --max-test-rows 50000
  --max-rows-per-hospital 25000
  --seed 42
)
if [[ -d "$TEST_DIR" ]]; then
  ARGS+=(--test-dir "$TEST_DIR")
fi

python3 scripts/convert_iomt_to_hospital_clients.py "${ARGS[@]}"

echo "=== IoMT data ready ==="
python3 scripts/check_data.py
