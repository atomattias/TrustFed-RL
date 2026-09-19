"""WUSTL DO4 governance scenarios (same cyber, vary κ)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_wustl_do4_scenarios import run_scenarios  # noqa: E402
import tempfile


def test_wustl_do4_same_cyber_isolate_vs_escalate():
    with tempfile.TemporaryDirectory() as td:
        out = run_scenarios(Path(td) / "do4.sqlite")
    assert out["summary"]["all_passed"], out["summary"]
    by_id = {c["case_id"]: c for c in out["cases"]}
    assert by_id["do4_isolate_admin_ws"]["decision"]["final_action"].upper() == "ISOLATE"
    assert by_id["do4_escalate_ventilator"]["decision"]["final_action"].upper() != "ISOLATE"
    # Core DO4 contrast: same isolate cyber → different authorized action
    assert (
        by_id["do4_isolate_admin_ws"]["decision"]["final_action"]
        != by_id["do4_escalate_ventilator"]["decision"]["final_action"]
    )
