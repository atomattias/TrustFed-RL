"""WP5: §10 edge cases, synthetic goldens (rules 3/8/9/10), transaction rollback."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from governance.canonical_schema import EXPECTED_SEED_COUNTS  # noqa: E402
from governance.context_builder import build_governance_context  # noqa: E402
from governance.device_binding import alias_detection_type  # noqa: E402
from governance.policy_repository import (  # noqa: E402
    GovernanceDecision,
    PolicyRepository,
    cmp_condition,
    resolve_final_action,
)
from governance.seed_import import ensure_canonical_db  # noqa: E402


class TestGovernanceWP5Edge(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = Path(self._td.name) / "gov.sqlite"
        ensure_canonical_db(self.db)
        self.repo = PolicyRepository(
            self.db, experiment_run_id="wp5_run", experiment_seed=42
        )

    def tearDown(self) -> None:
        self.repo.close()
        self._td.cleanup()

    def test_acceptance_seed_and_tables(self) -> None:
        self.assertEqual(self.repo.seed_counts(), EXPECTED_SEED_COUNTS)
        self.assertEqual(self.repo.executable_rule_count(), 12)
        tables = {
            r[0]
            for r in self.repo._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for t in (
            "policy_sources",
            "governance_rules",
            "rule_conditions",
            "medical_device_profiles",
            "governance_decisions",
            "decision_rule_matches",
            "condition_evaluation_log",
        ):
            self.assertIn(t, tables)

    def test_ops_whitespace_in_and_null_ne(self) -> None:
        self.assertTrue(cmp_condition("IN", "ISOLATE", " BLOCK , ISOLATE , ALERT "))
        self.assertTrue(cmp_condition("NOT IN", "MONITOR", "BLOCK, ISOLATE"))
        self.assertFalse(cmp_condition("!=", None, "CRITICAL"))
        self.assertFalse(cmp_condition("!=", None, "TRUE"))

    def test_null_substitute_legal_final(self) -> None:
        self.assertEqual(resolve_final_action("ESCALATE", "ISOLATE", None), "MONITOR")
        self.assertEqual(resolve_final_action("REJECT", "BLOCK", None), "NO_ACTION")
        self.assertEqual(resolve_final_action("POSTPONE", "ALERT", None), "MONITOR")
        self.assertEqual(resolve_final_action("MODIFY", "BLOCK", None), "ALERT")
        for oc, sub in (
            ("ESCALATE", None),
            ("REJECT", None),
            ("MODIFY", None),
        ):
            act = resolve_final_action(oc, "ISOLATE", sub)
            self.assertIn(
                act, {"MONITOR", "ALERT", "THROTTLE", "BLOCK", "ISOLATE", "NO_ACTION"}
            )
            self.assertNotEqual(act.lower(), "escalate")

    def test_precedence_reject_over_modify(self) -> None:
        # Synthetic rules on a private marker (via context_overrides — not remapped).
        conn = self.repo._conn
        conn.execute(
            """
            INSERT INTO governance_rules(
                rule_id, policy_source_id, rule_name, domain, priority,
                recommended_outcome, substitute_action, human_review_required,
                version, rule_status
            ) VALUES
            (9001, 1, 'syn_modify', 'CYBERSECURITY', 1, 'MODIFY', 'ALERT', 0, '1.0', 'ACTIVE'),
            (9002, 1, 'syn_reject', 'CYBERSECURITY', 5, 'REJECT', 'MONITOR', 1, '1.0', 'ACTIVE')
            """
        )
        conn.execute(
            """
            INSERT INTO rule_conditions(
                condition_id, rule_id, context_field, operator, threshold_value,
                condition_order, status
            ) VALUES
            (9001, 9001, 'wp5_marker', '=', 'SYN', 1, 'ACTIVE'),
            (9002, 9002, 'wp5_marker', '=', 'SYN', 1, 'ACTIVE')
            """
        )
        conn.commit()
        d = self.repo.evaluate(
            {
                "rl_action": "block",
                "trust": 0.5,
                "participant_id": "hospital_10_compromised_iomt",  # WS-ADMIN, no patient data
                "device_type": "printer",
                "context_overrides": {"wp5_marker": "SYN"},
            }
        )
        self.assertEqual(d.outcome, "REJECT")
        self.assertEqual(d.decisive_rule_id, 9002)
        self.assertIn(9001, d.matched_rule_ids)
        self.assertIn(9002, d.matched_rule_ids)

    def test_tie_lower_priority_then_lower_rule_id(self) -> None:
        conn = self.repo._conn
        conn.execute(
            """
            INSERT INTO governance_rules(
                rule_id, policy_source_id, rule_name, domain, priority,
                recommended_outcome, substitute_action, human_review_required,
                version, rule_status
            ) VALUES
            (9101, 1, 'tie_hi_pri', 'CYBERSECURITY', 2, 'APPROVE', NULL, 0, '1.0', 'ACTIVE'),
            (9102, 1, 'tie_lo_pri', 'CYBERSECURITY', 1, 'APPROVE', NULL, 0, '1.0', 'ACTIVE'),
            (9103, 1, 'tie_same_pri_hi_id', 'CYBERSECURITY', 1, 'APPROVE', NULL, 0, '1.0', 'ACTIVE')
            """
        )
        for cid, rid in ((9101, 9101), (9102, 9102), (9103, 9103)):
            conn.execute(
                """
                INSERT INTO rule_conditions(
                    condition_id, rule_id, context_field, operator, threshold_value,
                    condition_order, status
                ) VALUES (?, ?, 'wp5_marker', '=', 'TIE', 1, 'ACTIVE')
                """,
                (cid, rid),
            )
        conn.commit()
        d = self.repo.evaluate(
            {
                "rl_action": "monitor",
                "trust": 0.9,
                "participant_id": "hospital_10_compromised_iomt",
                "device_type": "printer",
                "context_overrides": {"wp5_marker": "TIE"},
            }
        )
        # Same outcome APPROVE: lower priority wins (1 over 2); among priority=1, lower rule_id (9102)
        self.assertEqual(d.decisive_rule_id, 9102)

    def test_infusion_pump_alias_binds_device_id(self) -> None:
        self.assertEqual(alias_detection_type("infusion_pump"), "INFUSION_PUMP")
        # Participant without roster pump → hash/type bind to an INFUSION_PUMP profile
        d = self.repo.evaluate(
            {
                "rl_action": "monitor",
                "trust": 0.9,
                "attack_severity": 0.1,
                "participant_id": "hospital_unmapped_alias_test",
                "device_type": "infusion_pump",
            }
        )
        self.assertIsNotNone(d.device_id)
        row = self.repo._conn.execute(
            "SELECT device_type FROM medical_device_profiles WHERE device_id=?",
            (d.device_id,),
        ).fetchone()
        self.assertEqual(row["device_type"], "INFUSION_PUMP")

    def test_match_insert_failure_rolls_back_decision(self) -> None:
        d = GovernanceDecision(
            outcome="APPROVE",
            final_action="MONITOR",
            human_review=False,
            explanation="wp5_rollback",
            matched_rule_ids=[1],
            matched_rule_versions=["1.0"],
            decisive_rule_id=1,
            matched_rules=[
                {
                    "rule_id": 1,
                    "version": "1.0",
                    "recommended_outcome": "APPROVE",
                    "substitute_action": None,
                    "priority": 1,
                    "matched_condition_count": 1,
                    "failed_condition_count": 0,
                    "conditions": [],
                }
            ],
        )
        before = self.repo._conn.execute(
            "SELECT COUNT(*) FROM governance_decisions"
        ).fetchone()[0]
        # Force FK failure on match insert by using a nonexistent rule_id after decision insert
        d.matched_rules[0]["rule_id"] = 999999
        d.matched_rule_ids = [999999]
        d.decisive_rule_id = 999999
        with self.assertRaises(sqlite3.IntegrityError):
            self.repo.record_decision(
                d,
                {
                    "rl_action": "monitor",
                    "trust": 0.5,
                    "participant_id": "hospital_12_medium_quality_iomt",
                    "device_type": "gateway",
                },
                round_num=1,
                agent_id="hospital_12_medium_quality_iomt",
            )
        after = self.repo._conn.execute(
            "SELECT COUNT(*) FROM governance_decisions"
        ).fetchone()[0]
        self.assertEqual(after, before)
        n_match = self.repo._conn.execute(
            "SELECT COUNT(*) FROM decision_rule_matches"
        ).fetchone()[0]
        self.assertEqual(n_match, 0)

    def test_golden_rule3_clinical_harm_modify(self) -> None:
        # CRITICAL + clinical_harm_risk>=0.70 + BLOCK/ISOLATE → rule 3 MODIFY→THROTTLE
        # (may also match stronger outcomes; assert rule 3 fires and isolate/block not final)
        d = self.repo.evaluate(
            {
                "rl_action": "block",
                "trust": 0.5,
                "attack_severity": 0.95,
                "clinical_harm_risk": 0.85,
                "participant_id": "hospital_01_high_quality_iomt",  # VENTILATOR CRITICAL
                "device_type": "infusion_pump",
            }
        )
        self.assertIn(3, d.matched_rule_ids)
        self.assertIn(d.outcome, {"MODIFY", "ESCALATE", "REJECT"})
        self.assertNotIn(d.final_action, {"BLOCK", "ISOLATE"})

    def test_golden_rule8_block_life_support(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "block",
                "trust": 0.5,
                "attack_severity": 0.5,
                "participant_id": "hospital_01_high_quality_iomt",
                "device_type": "infusion_pump",
            }
        )
        self.assertIn(8, d.matched_rule_ids)
        self.assertNotEqual(d.final_action, "BLOCK")
        # Rule 8 alone → MODIFY/THROTTLE; permit clamp may escalate
        self.assertIn(d.outcome, {"MODIFY", "ESCALATE", "REJECT"})

    def test_golden_rule9_isolate_life_support(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "isolate",
                "trust": 0.5,
                "attack_severity": 0.5,
                "participant_id": "hospital_01_high_quality_iomt",
                "device_type": "infusion_pump",
            }
        )
        self.assertIn(9, d.matched_rule_ids)
        self.assertNotEqual(d.final_action, "ISOLATE")
        self.assertNotEqual(d.final_action.lower(), "escalate")

    def test_golden_rule10_privacy_reject(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "alert",
                "trust": 0.5,
                "attack_severity": 0.2,
                "participant_id": "hospital_09_compromised_iomt",  # WS with patient_data
                "device_type": "gateway",
                "cybersecurity_purpose_authorised": False,
            }
        )
        self.assertIn(10, d.matched_rule_ids)
        self.assertEqual(d.outcome, "REJECT")
        self.assertEqual(d.final_action, "MONITOR")
        self.assertTrue(d.human_review)

    def test_context_covers_seed_fields(self) -> None:
        ctx = build_governance_context(
            {
                "rl_action": "isolate",
                "trust": 0.4,
                "attack_severity": 0.91,
                "contextual_risk": 0.3,
            },
            {
                "device_id": 1,
                "device_type": "VENTILATOR",
                "criticality": "CRITICAL",
                "life_support_status": 1,
                "patient_data_access": 1,
                "regulated_medical_device": 1,
                "permitted_automatic_actions": "MONITOR,ALERT,THROTTLE",
            },
        )
        required = {
            "rl_action",
            "device_criticality",
            "patient_data_access",
            "life_support_status",
            "regulated_medical_device",
            "cybersecurity_purpose_authorised",
            "attack_severity",
            "risk_score",
            "patient_safety_risk",
            "clinical_harm_risk",
            "clinical_impact",
            "clinical_risk_assessed",
            "human_override_requested",
            "operator_authorised",
            "human_approval_status",
            "escalation_required",
            "contains_personal_identifiers",
            "identifiers_required_for_response",
            "security_alert_generated",
            "contains_patient_information",
            "minimum_necessary_data",
        }
        missing = required - set(ctx)
        self.assertFalse(missing, f"missing context fields: {missing}")
        self.assertEqual(ctx["attack_severity"], "CRITICAL")

    def test_human_review_rate_scoped(self) -> None:
        d = self.repo.evaluate(
            {
                "rl_action": "isolate",
                "trust": 0.4,
                "attack_severity": 0.9,
                "participant_id": "hospital_01_high_quality_iomt",
                "device_type": "infusion_pump",
            }
        )
        self.repo.record_decision(
            d,
            {
                "rl_action": "isolate",
                "trust": 0.4,
                "participant_id": "hospital_01_high_quality_iomt",
                "device_type": "infusion_pump",
            },
            round_num=1,
            agent_id="hospital_01_high_quality_iomt",
            experiment_run_id="wp5_run",
        )
        self.assertGreater(self.repo.human_review_rate("wp5_run"), 0.0)
        self.assertEqual(self.repo.human_review_rate("never_ran"), 0.0)


if __name__ == "__main__":
    unittest.main()
