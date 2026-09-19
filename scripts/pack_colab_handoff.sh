#!/usr/bin/env bash
# Build everything needed to finish B5 / B5† / B4† on Google Colab.
# Usage (from TrustFed-Agent root):
#   bash scripts/pack_colab_handoff.sh
#
# Produces in repo root (upload all three to My Drive/trustfed/):
#   TrustFed-Agent-colab.zip   — code + colleague_v1 governance (fresh pack)
#   iomt_data_bundle.zip       — IoMT CSVs
#   iomt_metrics_latest.zip    — local metrics for resume/skip
#
# Then open/upload: colab/Run_TrustFed_RL_IoMT_Matrix.ipynb
# Colab run cell:
#   !python scripts/colab_run_experiments.py --task trustfed_rl_matrix --phases B5,B5star,B4star --download
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== TrustFed-RL Colab handoff pack ==="
echo "Root: $ROOT"
echo ""

# Status snapshot
if [[ -x "$ROOT/.venv312/bin/python" ]]; then
  PY="$ROOT/.venv312/bin/python"
else
  PY="${PYTHON:-python3}"
fi
"$PY" scripts/check_trustfed_rl_iomt_status.py || true
echo ""

# Warn if a local matrix is still writing B5 (Colab must not also run B5)
if pgrep -f "run_trustfed_rl_iomt_matrix|trustfed_agent_runner.*trustfed_agent" >/dev/null 2>&1; then
  echo "NOTE: a local matrix / agent runner is still active."
  echo "      Keep it running. On Colab use PARALLEL-SAFE phases only:"
  echo "        --phases B5star,B4star"
  echo "      Do NOT start Colab --phases B5,... until local B5 is finished."
  echo "      Re-run this pack later to refresh iomt_metrics_latest.zip."
  echo ""
fi

echo "--- Packing IoMT data ---"
bash scripts/pack_iomt_data_bundle.sh "$ROOT/iomt_data_bundle.zip"

echo ""
echo "--- Packing code (+ governance seed/config) ---"
bash scripts/pack_repo_for_colab.sh "$ROOT/TrustFed-Agent-colab.zip"

echo ""
echo "--- Packing metrics resume zip ---"
METRICS="$ROOT/results/trustfed_agent/metrics"
OUT_METRICS="$ROOT/iomt_metrics_latest.zip"
if [[ ! -d "$METRICS" ]]; then
  echo "ERROR: missing $METRICS"
  exit 1
fi
n=$(ls "$METRICS"/run_*.json 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n" -eq 0 ]]; then
  echo "ERROR: no run_*.json under metrics/"
  exit 1
fi
rm -f "$OUT_METRICS"
(
  cd "$METRICS"
  zip -q "$OUT_METRICS" run_*.json
)
ls -lh "$OUT_METRICS"
echo "Packed $n metric JSON files → $(basename "$OUT_METRICS")"

echo ""
echo "=== Upload to Google Drive → My Drive/trustfed/ ==="
echo "  1) $(ls -lh TrustFed-Agent-colab.zip | awk '{print $5}')  TrustFed-Agent-colab.zip"
echo "  2) $(ls -lh iomt_data_bundle.zip | awk '{print $5}')  iomt_data_bundle.zip"
echo "  3) $(ls -lh iomt_metrics_latest.zip | awk '{print $5}')  iomt_metrics_latest.zip"
echo "  4) colab/Run_TrustFed_RL_IoMT_Matrix.ipynb  (File → Upload notebook in Colab)"
echo ""
echo "In Colab (while local B5 still runs — parallel safe):"
echo "  !python scripts/colab_run_experiments.py --task trustfed_rl_matrix --phases B5star,B4star --download"
echo ""
echo "After local B5 is 12/12, optionally:"
echo "  !python scripts/colab_run_experiments.py --task trustfed_rl_matrix --phases B5,B5star,B4star --download"
echo ""
echo "Phase ids: B5 = full stack, B5star = B5†, B4star = B4†"
echo "See colab/COLAB_INSTRUCTIONS.md"
