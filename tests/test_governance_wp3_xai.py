"""WP3: forensic XAI pack reads canonical governance DB + condition logs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from governance.policy_repository import PolicyRepository  # noqa: E402
from governance.seed_import import ensure_canonical_db  # noqa: E402
from xai.explainer import (  # noqa: E402
    build_forensic_record,
    export_forensic_json,
    fetch_governance_decision,
    find_decision_ids,
    summarize_governance_db,
)


class TestGovernanceWP3XAI(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = Path(self._td.name) / "gov.sqlite"
        ensure_canonical_db(self.db)
        self.repo = PolicyRepository(
            self.db, experiment_run_id="wp3_run", experiment_seed=42
        )
        d = self.repo.evaluate(
            {
                "rl_action": "isolate",
                "trust": 0.4,
                "attack_severity": 0.95,
                "participant_id": "hospital_09_compromised_iomt",
                "device_type": "gateway",
            }
        )
        self.decision_id = self.repo.record_decision(
            d,
            {
                "rl_action": "isolate",
                "trust": 0.4,
                "attack_severity": 0.95,
                "participant_id": "hospital_09_compromised_iomt",
                "device_type": "gateway",
            },
            round_num=1,
            agent_id="hospital_09_compromised_iomt",
        )

    def tearDown(self) -> None:
        self.repo.close()
        self._td.cleanup()

    def test_fetch_canonical_fields_and_condition_log(self) -> None:
        gov = fetch_governance_decision(self.db, self.decision_id)
        self.assertIsNotNone(gov)
        assert gov is not None
        self.assertEqual(gov["schema"], "colleague_v1")
        self.assertEqual(gov["participant_id"], "hospital_09_compromised_iomt")
        self.assertEqual(gov["experiment_round"], 1)
        self.assertEqual(gov["experiment_run_id"], "wp3_run")
        self.assertIn(gov["governance_outcome"], {"APPROVE", "MODIFY", "REJECT", "ESCALATE", "POSTPONE"})
        self.assertIn(
            gov["final_action"],
            {"MONITOR", "ALERT", "THROTTLE", "BLOCK", "ISOLATE", "NO_ACTION"},
        )
        # legacy aliases
        self.assertEqual(gov["matched_outcome"], gov["governance_outcome"])
        self.assertEqual(gov["agent_id"], gov["participant_id"])
        self.assertTrue(isinstance(gov["human_review_required"], bool))
        self.assertGreaterEqual(len(gov["matched_rules"]), 0)
        self.assertGreater(len(gov["condition_evaluation_log"]), 0)
        self.assertIn("permitted_automatic_actions", gov)
        self.assertIsNotNone(gov.get("device_identifier"))
        self.assertIn("condition_description", gov["condition_evaluation_log"][0])
        if gov["decisive_rule"]:
            self.assertIn("rule_name", gov["decisive_rule"])

    def test_find_and_summarize(self) -> None:
        ids = find_decision_ids(
            self.db,
            participant_id="hospital_09_compromised_iomt",
            experiment_round=1,
            experiment_run_id="wp3_run",
        )
        self.assertEqual(ids, [self.decision_id])
        summary = summarize_governance_db(self.db, experiment_run_id="wp3_run")
        self.assertEqual(summary["n_decisions"], 1)
        self.assertEqual(summary["schema"], "colleague_v1")
        self.assertGreater(summary["n_condition_logs"], 0)
        self.assertEqual(summary["human_review_rate"], self.repo.human_review_rate("wp3_run"))

    def test_forensic_pack_export(self) -> None:
        # Minimal events with decision_id so explain_response picks DB row
        events = [
            {
                "event_type": "trust_update",
                "agent_id": "hospital_09_compromised_iomt",
                "round": 1,
                "trust": 0.55,
                "signals": {"V": 0.5, "S": 0.5, "D": 0.5, "U": 0.5, "C": 0.5, "R": 0.5},
            },
            {
                "event_type": "response_proposed",
                "agent_id": "hospital_09_compromised_iomt",
                "round": 1,
                "action": "isolate",
                "confidence": 0.9,
                "device_type": "gateway",
                "status": "escalated",
                "decision_id": self.decision_id,
                "human_review": True,
            },
        ]
        weights = {k: 1 / 6 for k in ("V", "S", "D", "U", "C", "R")}
        record = build_forensic_record(
            agent_id="hospital_09_compromised_iomt",
            round_num=1,
            events=events,
            weights=weights,
            policy_db=self.db,
            include_condition_log=True,
            experiment_run_id="wp3_run",
        )
        self.assertEqual(record["schema"], "colleague_v1")
        self.assertIn("governance_outcome", record["governance"])
        self.assertGreater(len(record["governance"]["condition_evaluation_log"]), 0)
        self.assertEqual(record["round_decision_ids"], [self.decision_id])
        self.assertEqual(record["governance"]["agent_id"], record["governance"]["participant_id"])
        self.assertEqual(
            record["governance"]["matched_outcome"],
            record["governance"]["governance_outcome"],
        )
        self.assertIn("Decisive rule", record["narrative"])
        out = Path(self._td.name) / "forensic.json"
        export_forensic_json(record, out)
        loaded = json.loads(out.read_text())
        self.assertEqual(loaded["governance"]["decision_id"], self.decision_id)

    def test_fallback_prefers_latest_decision(self) -> None:
        d2 = self.repo.evaluate(
            {
                "rl_action": "monitor",
                "trust": 0.9,
                "participant_id": "hospital_09_compromised_iomt",
                "device_type": "gateway",
            }
        )
        did2 = self.repo.record_decision(
            d2,
            {
                "rl_action": "monitor",
                "trust": 0.9,
                "participant_id": "hospital_09_compromised_iomt",
                "device_type": "gateway",
            },
            round_num=1,
            agent_id="hospital_09_compromised_iomt",
        )
        events = [
            {
                "event_type": "trust_update",
                "agent_id": "hospital_09_compromised_iomt",
                "round": 1,
                "trust": 0.5,
                "signals": {"V": 0.5, "S": 0.5, "D": 0.5, "U": 0.5, "C": 0.5, "R": 0.5},
            },
            {
                "event_type": "response_proposed",
                "agent_id": "hospital_09_compromised_iomt",
                "round": 1,
                "action": "monitor",
                "status": "approved",
            },
        ]
        weights = {k: 1 / 6 for k in ("V", "S", "D", "U", "C", "R")}
        record = build_forensic_record(
            agent_id="hospital_09_compromised_iomt",
            round_num=1,
            events=events,
            weights=weights,
            policy_db=self.db,
            experiment_run_id="wp3_run",
        )
        self.assertEqual(record["governance"]["decision_id"], did2)
        self.assertEqual(record["round_decision_ids"], [self.decision_id, did2])


if __name__ == "__main__":
    unittest.main()
