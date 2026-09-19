#!/usr/bin/env bash
# WP5 smoke: B3 + B5, 2 rounds, seed 42; archive under results/trustfed_agent/smoke_wp5/
# Does not leave 2-round metrics in the paper metrics/ tree (restores prior B5 if present).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv312/bin/python}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK=TRUE

METRICS="$ROOT/results/trustfed_agent/metrics"
SMOKE="$ROOT/results/trustfed_agent/smoke_wp5"
mkdir -p "$SMOKE" "$METRICS"

B5_PAPER="$METRICS/run_trustfed_agent_iomt_seed_42.json"
if [[ -f "$B5_PAPER" ]]; then
  cp "$B5_PAPER" "$SMOKE/backup_run_trustfed_agent_iomt_seed_42.json"
fi

echo "=== B3 (trustfed_governance) 2 rounds seed 42 ==="
"$PY" trustfed_agent_runner.py --approach trustfed_governance --dataset iomt \
  --data-dir data/CSVs/iomt_clients --num-rounds 2 --random-state 42

echo "=== B5 (trustfed_agent) 2 rounds seed 42 ==="
"$PY" trustfed_agent_runner.py --approach trustfed_agent --dataset iomt \
  --data-dir data/CSVs/iomt_clients --num-rounds 2 --random-state 42

for f in run_trustfed_governance_iomt_seed_42.json run_trustfed_agent_iomt_seed_42.json; do
  cp "$METRICS/$f" "$SMOKE/$f"
done

if [[ -f "$SMOKE/backup_run_trustfed_agent_iomt_seed_42.json" ]]; then
  cp "$SMOKE/backup_run_trustfed_agent_iomt_seed_42.json" "$B5_PAPER"
  echo "Restored prior paper B5 metrics at $B5_PAPER"
fi
# Keep B3 smoke metrics in metrics/ (usually no prior paper B3) and smoke/

"$PY" - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
from xai.explainer import fetch_governance_decision

smoke = Path("results/trustfed_agent/smoke_wp5")
for name in (
    "run_trustfed_governance_iomt_seed_42.json",
    "run_trustfed_agent_iomt_seed_42.json",
):
    m = json.loads((smoke / name).read_text())
    gov = m.get("governance") or {}
    xai = m.get("xai") or {}
    assert not xai.get("error"), xai.get("error")
    assert xai.get("decision_id") is not None, f"{name}: missing xai.decision_id"
    assert (gov.get("n_decisions") or 0) > 0, f"{name}: missing governance.n_decisions"
    g = fetch_governance_decision(Path(gov["policy_db"]), int(xai["decision_id"]))
    assert g and g.get("governance_outcome"), f"{name}: cannot load decision"
    print(
        name,
        "rounds=",
        m.get("num_rounds"),
        "n_decisions=",
        gov.get("n_decisions"),
        "decision_id=",
        xai.get("decision_id"),
        "outcome=",
        g.get("governance_outcome"),
    )
print("WP5_SMOKE_OK")
PY
