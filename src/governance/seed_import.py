"""Import colleague Table VI seed data into a canonical SQLite policy DB (WP1)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Union

from governance.canonical_schema import (
    CANONICAL_DDL,
    EXPECTED_SEED_COUNTS,
    SCHEMA_VERSION,
)

PathLike = Union[str, Path]

_DEFAULT_SEED = (
    Path(__file__).resolve().parents[2] / "data" / "governance" / "colleague_seed_v1.json"
)


def default_seed_path() -> Path:
    return _DEFAULT_SEED


def _as_int_bool(v: Any) -> int:
    if v is True or v == 1 or v == "1" or v == "t" or v == "true":
        return 1
    return 0


def apply_canonical_ddl(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(CANONICAL_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def seed_is_loaded(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='seed_loaded'"
    ).fetchone()
    return bool(row and row[0] == "1")


def schema_version(conn: sqlite3.Connection) -> Optional[str]:
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        return str(row[0]) if row else None
    except sqlite3.Error:
        return None


def count_table(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row[0] if not isinstance(row, sqlite3.Row) else row["n"])


def verify_seed_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    got = {t: count_table(conn, t) for t in EXPECTED_SEED_COUNTS}
    bad = {t: (got[t], exp) for t, exp in EXPECTED_SEED_COUNTS.items() if got[t] != exp}
    if bad:
        raise RuntimeError(f"seed count mismatch: {bad}")
    return got


def _seed_table_nonempty(conn: sqlite3.Connection) -> bool:
    return any(count_table(conn, t) > 0 for t in EXPECTED_SEED_COUNTS)


def load_seed_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"seed schema_version {data.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    return data


def spot_check_seed(conn: sqlite3.Connection) -> Dict[str, Any]:
    """WP1 cutover checks: rule 9 life-support conditions + device 1 + no sample decisions."""
    rule = conn.execute(
        """
        SELECT rule_id, recommended_outcome, substitute_action, rule_name
        FROM governance_rules WHERE rule_id=9
        """
    ).fetchone()
    if rule is None:
        raise RuntimeError("spot-check failed: rule_id=9 missing")

    def _row(r: Any) -> Dict[str, Any]:
        if r is None:
            return {}
        if isinstance(r, sqlite3.Row):
            return {k: r[k] for k in r.keys()}
        if isinstance(r, dict):
            return r
        # plain tuple from default row factory — match SELECT column order
        return {}

    cond_rows = conn.execute(
        """
        SELECT condition_order, context_field, operator, threshold_value
        FROM rule_conditions WHERE rule_id=9
        ORDER BY condition_order, condition_id
        """
    ).fetchall()
    if len(cond_rows) != 2:
        raise RuntimeError(
            f"spot-check failed: rule 9 expected 2 conditions, got {len(cond_rows)}"
        )

    if isinstance(cond_rows[0], sqlite3.Row):
        c0 = _row(cond_rows[0])
        c1 = _row(cond_rows[1])
    else:
        c0 = {
            "condition_order": cond_rows[0][0],
            "context_field": cond_rows[0][1],
            "operator": cond_rows[0][2],
            "threshold_value": cond_rows[0][3],
        }
        c1 = {
            "condition_order": cond_rows[1][0],
            "context_field": cond_rows[1][1],
            "operator": cond_rows[1][2],
            "threshold_value": cond_rows[1][3],
        }

    if c0["context_field"] != "life_support_status" or c0["threshold_value"] != "TRUE":
        raise RuntimeError(f"spot-check failed: rule 9 cond1 unexpected: {c0}")
    if c1["context_field"] != "rl_action" or c1["threshold_value"] != "ISOLATE":
        raise RuntimeError(f"spot-check failed: rule 9 cond2 unexpected: {c1}")

    device = conn.execute(
        """
        SELECT device_id, device_identifier, device_type, criticality, life_support_status
        FROM medical_device_profiles WHERE device_id=1
        """
    ).fetchone()
    if device is None:
        raise RuntimeError("spot-check failed: device_id=1 missing")
    if isinstance(device, sqlite3.Row):
        device_type = device["device_type"]
        device_identifier = device["device_identifier"]
        rule_outcome = rule["recommended_outcome"]
        rule_sub = rule["substitute_action"]
    else:
        device_type = device[2]
        device_identifier = device[1]
        rule_outcome = rule[1]
        rule_sub = rule[2]
    if device_type != "VENTILATOR":
        raise RuntimeError(f"spot-check failed: device_id=1 type={device_type}")

    return {
        "rule_9_outcome": rule_outcome,
        "rule_9_substitute": rule_sub,
        "rule_9_conditions": [c0, c1],
        "device_1_type": device_type,
        "device_1_identifier": device_identifier,
        "decisions": count_table(conn, "governance_decisions"),
        "matches": count_table(conn, "decision_rule_matches"),
    }


def _clear_seed_tables(conn: sqlite3.Connection) -> None:
    for t in (
        "condition_evaluation_log",
        "decision_rule_matches",
        "governance_decisions",
        "rule_conditions",
        "governance_rules",
        "medical_device_profiles",
        "policy_sources",
    ):
        conn.execute(f"DELETE FROM {t}")
    # Keep AUTOINCREMENT counters clean for runtime tables
    conn.execute("DELETE FROM sqlite_sequence")


def import_colleague_seed(
    conn: sqlite3.Connection,
    seed_path: Optional[PathLike] = None,
    *,
    replace: bool = False,
) -> Dict[str, int]:
    """
    Insert the four seed tables. Does not import sample decisions/matches/logs.

    If replace=True, clears seed (+ decision) tables first.
    """
    seed = load_seed_json(seed_path or default_seed_path())
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()  # end any implicit transaction from DDL

    if not replace and seed_is_loaded(conn) and count_table(conn, "governance_rules") > 0:
        counts = verify_seed_counts(conn)
        spot_check_seed(conn)
        return counts

    # Partial import without seed_loaded marker → force replace to avoid PK clashes
    if not replace and not seed_is_loaded(conn) and _seed_table_nonempty(conn):
        replace = True

    try:
        if replace:
            _clear_seed_tables(conn)

        for row in seed["policy_sources"]:
            conn.execute(
                """
                INSERT INTO policy_sources(
                    policy_source_id, source_name, jurisdiction, article_section,
                    publication_year, document_reference, description, status
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    int(row["policy_source_id"]),
                    row["source_name"],
                    row.get("jurisdiction"),
                    row.get("article_section"),
                    row.get("publication_year"),
                    row.get("document_reference"),
                    row.get("description"),
                    row.get("status") or "ACTIVE",
                ),
            )

        for row in seed["governance_rules"]:
            conn.execute(
                """
                INSERT INTO governance_rules(
                    rule_id, policy_source_id, rule_name, domain, priority,
                    recommended_outcome, substitute_action, human_review_required,
                    version, rule_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(row["rule_id"]),
                    int(row["policy_source_id"]),
                    row["rule_name"],
                    row["domain"],
                    int(row["priority"]),
                    row["recommended_outcome"],
                    row.get("substitute_action"),
                    _as_int_bool(row.get("human_review_required")),
                    str(row.get("version") or "1.0"),
                    row.get("rule_status") or "ACTIVE",
                ),
            )

        for row in seed["rule_conditions"]:
            conn.execute(
                """
                INSERT INTO rule_conditions(
                    condition_id, rule_id, context_field, operator, threshold_value,
                    logical_connector, condition_order, condition_description, status
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(row["condition_id"]),
                    int(row["rule_id"]),
                    row["context_field"],
                    row["operator"],
                    row["threshold_value"],
                    row.get("logical_connector"),
                    int(row.get("condition_order") or 1),
                    row.get("condition_description"),
                    row.get("status") or "ACTIVE",
                ),
            )

        for row in seed["medical_device_profiles"]:
            conn.execute(
                """
                INSERT INTO medical_device_profiles(
                    device_id, device_identifier, device_name, device_type,
                    clinical_function, criticality, life_support_status,
                    regulated_medical_device, active_clinical_use, patient_data_access,
                    permitted_automatic_actions, location, network_segment, device_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(row["device_id"]),
                    row["device_identifier"],
                    row["device_name"],
                    row["device_type"],
                    row.get("clinical_function"),
                    row["criticality"],
                    _as_int_bool(row.get("life_support_status")),
                    _as_int_bool(row.get("regulated_medical_device", True)),
                    _as_int_bool(row.get("active_clinical_use")),
                    _as_int_bool(row.get("patient_data_access")),
                    row.get("permitted_automatic_actions"),
                    row.get("location"),
                    row.get("network_segment"),
                    row.get("device_status") or "ACTIVE",
                ),
            )

        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('seed_loaded', '1')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('seed_path', ?)",
            (str(Path(seed_path or default_seed_path()).resolve()),),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    counts = verify_seed_counts(conn)
    spot_check_seed(conn)
    return counts


def ensure_canonical_db(
    db_path: PathLike,
    seed_path: Optional[PathLike] = None,
    *,
    replace_seed: bool = False,
) -> Dict[str, Any]:
    """Create DB file, apply DDL, import seed if needed. Returns summary dict."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        apply_canonical_ddl(conn)
        counts = import_colleague_seed(conn, seed_path, replace=replace_seed)
        spot = spot_check_seed(conn)
        return {
            "db_path": str(path.resolve()),
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "spot_check": {
                "rule_9_outcome": spot["rule_9_outcome"],
                "device_1_type": spot["device_1_type"],
                "decisions": spot["decisions"],
            },
        }
    finally:
        conn.close()
