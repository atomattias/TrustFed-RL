#!/usr/bin/env bash
# Pack IoMT CSVs for Google Drive upload (Colab).
# Usage (from TrustFed-Agent root):
#   bash scripts/pack_iomt_data_bundle.sh
# Upload the resulting iomt_data_bundle.zip to My Drive/trustfed/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CLIENTS="data/CSVs/iomt_clients"
TEST="data/CSVs/iomt_test_set.csv"
OUT="${1:-iomt_data_bundle.zip}"

if [[ ! -d "$CLIENTS" ]]; then
  echo "ERROR: missing $CLIENTS — run scripts/setup_iomt_data.sh first"
  exit 1
fi
if [[ ! -f "$TEST" ]]; then
  echo "ERROR: missing $TEST"
  exit 1
fi

n=$(ls "$CLIENTS"/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "Packing $n hospital CSVs + test set → $OUT"
rm -f "$OUT"
zip -r -q "$OUT" "$CLIENTS" "$TEST"
ls -lh "$OUT"
echo "Upload to: Google Drive → My Drive/trustfed/$OUT"
echo "Then open: colab/Run_TrustFed_RL_IoMT_Matrix.ipynb"
