#!/usr/bin/env bash
# Pack TrustFed-Agent source for Colab (private-repo friendly).
# Excludes venv, results, raw CSVs, large caches.
# Usage (from TrustFed-Agent root OR parent):
#   bash TrustFed-Agent/scripts/pack_repo_for_colab.sh
# Upload TrustFed-Agent-colab.zip to My Drive/trustfed/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/TrustFed-Agent-colab.zip}"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"

cd "$PARENT"
rm -f "$OUT"
echo "Packing $NAME → $OUT (code + governance seed; IoMT CSVs separate)..."
# Kept: data/governance/*.json, config/iomt_governance_config.json,
#        config/governance_device_*.json (colleague_v1 Table VI for B3/B5).
zip -r -q "$OUT" "$NAME" \
  -x "$NAME/.venv/*" \
  -x "$NAME/.venv312/*" \
  -x "$NAME/venv/*" \
  -x "$NAME/**/__pycache__/*" \
  -x "$NAME/**/.ipynb_checkpoints/*" \
  -x "$NAME/results/**" \
  -x "$NAME/data/CSVs/**" \
  -x "$NAME/data/train/**" \
  -x "$NAME/data/test/**" \
  -x "$NAME/iomt_data_bundle.zip" \
  -x "$NAME/iomt_metrics_latest.zip" \
  -x "$NAME/TrustFed-Agent-colab.zip" \
  -x "$NAME/*.zip" \
  -x "$NAME/**/*.pyc" \
  -x "$NAME/.git/*"

ls -lh "$OUT"
echo ""
echo "Upload BOTH to Google Drive → My Drive/trustfed/ :"
echo "  1) $(basename "$OUT")          # code + governance seed/config"
echo "  2) iomt_data_bundle.zip        # data (bash scripts/pack_iomt_data_bundle.sh)"
echo "Then re-run notebook cell 1 (Drive zip is preferred over git clone)."
echo "See colab/COLAB_INSTRUCTIONS.md (governance zip note)."
