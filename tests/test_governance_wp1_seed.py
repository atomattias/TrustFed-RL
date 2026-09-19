"""WP1 acceptance: canonical schema + colleague seed counts/IDs."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from governance.canonical_schema import EXPECTED_SEED_COUNTS, SCHEMA_VERSION  # noqa: E402
from governance.device_binding import (  # noqa: E402
    alias_detection_type,
    load_device_aliases,
    load_device_roster,
)
from governance.policy_repository import PolicyRepository  # noqa: E402
from governance.seed_import import ensure_canonical_db, spot_check_seed  # noqa: E402


class TestGovernanceWP1(unittest.TestCase):
    def test_ensure_canonical_counts_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "gov.sqlite"
            summary = ensure_canonical_db(db)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(summary["counts"], EXPECTED_SEED_COUNTS)

            conn = sqlite3.connect(str(db))
            conn.execute("PRAGMA foreign_keys = ON")
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            spot = spot_check_seed(conn)
            self.assertEqual(spot["rule_9_outcome"], "MODIFY")
            self.assertEqual(spot["rule_9_substitute"], "ALERT")
            self.assertEqual(len(spot["rule_9_conditions"]), 2)
            self.assertEqual(spot["device_1_type"], "VENTILATOR")
            self.assertEqual(spot["decisions"], 0)
            # FK enforced: orphan condition should fail
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO rule_conditions(
                        condition_id, rule_id, context_field, operator,
                        threshold_value, condition_order, status
                    ) VALUES (9999, 9999, 'x', '=', '1', 1, 'ACTIVE')
                    """
                )
            conn.close()

            repo = PolicyRepository(db)
            self.assertEqual(repo.rule_count(), 45)
            self.assertEqual(repo.executable_rule_count(), 12)
            self.assertTrue(repo.runtime_api_ready)
            self.assertEqual(len(repo._load_active_rules()), 12)
            repo.close()

    def test_idempotent_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "gov.sqlite"
            ensure_canonical_db(db)
            ensure_canonical_db(db)  # second open must not duplicate
            repo = PolicyRepository(db)
            self.assertEqual(repo.seed_counts(), EXPECTED_SEED_COUNTS)
            repo.close()

    def test_aliases_and_roster_configs(self) -> None:
        aliases = load_device_aliases()
        self.assertEqual(aliases["infusion_pump"], "INFUSION_PUMP")
        self.assertEqual(alias_detection_type("vitals_monitor"), "PATIENT_MONITOR")
        roster = load_device_roster()
        mapping = roster["participant_to_device_identifier"]
        self.assertIn("hospital_09_compromised_iomt", mapping)
        self.assertEqual(mapping["hospital_09_compromised_iomt"], "WS-DOCTOR-011")
        self.assertEqual(len(mapping), 12)


if __name__ == "__main__":
    unittest.main()
