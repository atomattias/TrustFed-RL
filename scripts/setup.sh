#!/usr/bin/env bash
# Create venv and install TrustFed-Agent dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "=== TrustFed-Agent setup ==="
echo "Root: $ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel
pip install -r requirements.txt

echo ""
echo "=== Verifying installation ==="
python scripts/verify_install.py

echo ""
echo "=== Checking data paths ==="
python scripts/check_data.py || true

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $ROOT"
