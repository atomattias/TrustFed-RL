"""WP2: context builder, null-safe evaluate, permit clamp, record, rate-limit."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.autonomous_agent import ResponseAction  # noqa: E402
from governance.context_builder import bin_attack_severity, build_governance_context  # noqa: E402
from governance.policy_engine import PolicyEngine  # noqa: E402
from governance.policy_repository import (  # noqa: E402
    PolicyRepository,
    clamp_to_permitted,
    cmp_condition,
    resolve_final_action,
)
from governance.security_validator import SecurityValidator, db_action_to_effector  # noqa: E402
from governance.seed_import import ensure_canonical_db  # noqa: E402


class TestGovernanceWP2(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = Path(self._td.name) / "gov.sqlite"
        ensure_canonical_db(self.db)
        self.repo = PolicyRepository(
            self.db,
            experiment_run_id="test_run",
            experiment_seed=42,
        )

    def tearDown(self) -> None:
        self.repo.close()
        self._td.cleanup()

    def test_ops_and_null_semantics(self) -> None:
        self.assertTrue(cmp_condition("=", "ISOLATE", "ISOLATE"))
        self.assertTrue(cmp_condition("IN", "BLOCK", "BLOCK,ISOLATE"))
        self.assertTrue(cmp_condition("IN", "block", " BLOCK , ISOLATE "))
        self.assertTrue(cmp_condition(">=", 0.8, "0.70"))
        self.assertTrue(cmp_condition("!=", "HIGH", "CRITICAL"))
        self.assertFalse(cmp_condition("!=", None, "CRITICAL"))  # missing ⇒ False
        self.assertFalse(cmp_condition("=", None, "TRUE"))
        self.assertTrue(cmp_condition("NOT IN", "MONITOR", "BLOCK,ISOLATE"))
        self.assertFalse(cmp_condition("??", 1, "1"))
        # Critical: boolean equality must respect threshold
        self.assertTrue(cmp_condition("=", True, "TRUE"))
        self.assertFalse(cmp_condition("=", True, "FALSE"))
        self.assertTrue(cmp_condition("=", False, "FALSE"))
        self.assertFalse(cmp_condition("=", False, "TRUE"))

    def test_human_approval_default_not_pending(self) -> None:
        ctx = build_governance_context({"rl_action": "block", "trust": 0.5}, None)
        self.assertEqual(ctx["human_approval_status"], "NOT_REQUIRED")
        ctx2 = build_governance_context(
            {"rl_action": "block", "escalation_required": True}, None
        )
        self.assertEqual(ctx2["human_approval_status"], "PENDING")

    def test_condition_log_written(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "isolate",
                "trust": 0.5,
                "attack_severity": 0.95,
                "participant_id": "hospital_01_high_quality_iomt",
                "device_type": "infusion_pump",
            }
        )
        did = self.repo.record_decision(
            d,
            {
                "rl_action": "isolate",
                "trust": 0.5,
                "participant_id": "hospital_01_high_quality_iomt",
                "device_type": "infusion_pump",
            },
            round_num=1,
            agent_id="hospital_01_high_quality_iomt",
        )
        n = self.repo._conn.execute(
            "SELECT COUNT(*) FROM condition_evaluation_log"
        ).fetchone()[0]
        self.assertGreater(n, 0)
        self.assertIsNotNone(did)

    def test_unconditional_rules_skipped(self) -> None:
        self.assertEqual(self.repo.executable_rule_count(), 12)
        self.assertEqual(len(self.repo._load_active_rules()), 12)

    def test_resolve_final_and_permit(self) -> None:
        self.assertEqual(resolve_final_action("ESCALATE", "ISOLATE", None), "MONITOR")
        self.assertEqual(resolve_final_action("REJECT", "BLOCK", None), "NO_ACTION")
        self.assertEqual(resolve_final_action("MODIFY", "BLOCK", None), "ALERT")
        act, over = clamp_to_permitted(
            "ISOLATE", "MONITOR,ALERT,THROTTLE", life_support=True
        )
        self.assertEqual(act, "MONITOR")
        self.assertEqual(over, "ESCALATE")

    def test_life_support_isolate_rule9(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "isolate",
                "trust": 0.5,
                "attack_severity": 0.95,
                "contextual_risk": 0.5,
                "participant_id": "hospital_01_high_quality_iomt",  # VENTILATOR
                "device_type": "infusion_pump",  # roster wins → ventilator
            }
        )
        # Roster binds hospital_01 → MD-ICU-001 ventilator life-support
        self.assertIn(d.outcome, {"MODIFY", "ESCALATE", "REJECT"})
        self.assertNotEqual(d.final_action, "ISOLATE")
        self.assertNotEqual(d.final_action.lower(), "escalate")
        self.assertIn(9, d.matched_rule_ids)  # life-support isolate rule

    def test_record_canonical_and_run_scoped_rate(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "monitor",
                "trust": 0.9,
                "attack_severity": 0.2,
                "participant_id": "hospital_08_low_quality_iomt",
                "device_type": "printer",
            }
        )
        did = self.repo.record_decision(
            d,
            {
                "rl_action": "monitor",
                "trust": 0.9,
                "participant_id": "hospital_08_low_quality_iomt",
                "device_type": "printer",
            },
            round_num=3,
            agent_id="hospital_08_low_quality_iomt",
            experiment_run_id="test_run",
        )
        self.assertIsNotNone(did)
        row = self.repo._conn.execute(
            "SELECT * FROM governance_decisions WHERE decision_id=?", (did,)
        ).fetchone()
        self.assertEqual(row["experiment_round"], 3)
        self.assertEqual(row["experiment_run_id"], "test_run")
        self.assertEqual(row["governance_outcome"], d.outcome)
        self.assertIn(
            row["final_action"],
            {"MONITOR", "ALERT", "THROTTLE", "BLOCK", "ISOLATE", "NO_ACTION"},
        )
        self.assertNotEqual(row["final_action"], "escalate")
        n_match = self.repo._conn.execute(
            "SELECT COUNT(*) FROM decision_rule_matches WHERE decision_id=?", (did,)
        ).fetchone()[0]
        self.assertEqual(n_match, len(d.matched_rule_ids))
        self.assertGreaterEqual(self.repo.human_review_rate("test_run"), 0.0)
        self.assertEqual(self.repo.human_review_rate("other_run"), 0.0)

    def test_rate_limit_final_action_legal(self) -> None:
        engine = PolicyEngine({})
        val = SecurityValidator(
            {
                "response": {
                    "allowed_actions": [
                        "monitor",
                        "alert",
                        "throttle",
                        "block",
                        "isolate",
                        "escalate",
                    ],
                    "rate_limit_per_round": 0,
                }
            },
            engine,
            policy_repository=self.repo,
        )
        action = ResponseAction(
            action_type="block",
            severity=0.7,
            confidence=0.95,
            agent_id="hospital_12_medium_quality_iomt",
            round_num=1,
            target="printer",
        )
        # First force APPROVE path then rate-limit: with rate_limit 0, aggressive APPROVE of block gets escalated
        res = val.validate(
            action,
            trust_score=0.9,
            round_num=1,
            agent_id="hospital_12_medium_quality_iomt",
            device_type="printer",
            attack_severity=0.2,
            contextual_risk=0.1,
            experiment_run_id="test_run",
        )
        row = self.repo._conn.execute(
            "SELECT governance_outcome, final_action FROM governance_decisions ORDER BY decision_id DESC LIMIT 1"
        ).fetchone()
        # Either rules modified it, or rate-limit set ESCALATE/MONITOR
        self.assertIn(row["final_action"], {"MONITOR", "ALERT", "THROTTLE", "BLOCK", "ISOLATE", "NO_ACTION"})
        self.assertNotEqual(row["final_action"], "escalate")
        effector, sev = db_action_to_effector("NO_ACTION")
        self.assertEqual(effector, "monitor")
        self.assertEqual(sev, 0.0)

    def test_severity_binning_and_context(self) -> None:
        self.assertEqual(bin_attack_severity(0.9), "CRITICAL")
        self.assertEqual(bin_attack_severity(0.75), "HIGH")
        ctx = build_governance_context(
            {"rl_action": "isolate", "trust": 0.5, "attack_severity": 0.91},
            {
                "device_id": 1,
                "device_type": "VENTILATOR",
                "criticality": "CRITICAL",
                "life_support_status": 1,
                "patient_data_access": 0,
                "regulated_medical_device": 1,
                "permitted_automatic_actions": "MONITOR,ALERT,THROTTLE",
            },
        )
        self.assertEqual(ctx["attack_severity"], "CRITICAL")
        self.assertTrue(ctx["life_support_status"])
        self.assertEqual(ctx["device_criticality"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
