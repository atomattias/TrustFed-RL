"""
Table VI Governance Policy Repository (SQLite).

WP1: colleague-compatible canonical schema + seed import.
WP2: context builder, null-safe evaluate, canonical record_decision.

Fail-safe precedence (master Eq. 3):
  REJECT ≻ ESCALATE ≻ POSTPONE ≻ MODIFY ≻ APPROVE
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from governance.canonical_schema import SCHEMA_VERSION
from governance.context_builder import (
    build_governance_context,
    clamp01,
    normalize_db_action,
)
from governance.device_binding import (
    load_device_aliases,
    load_device_roster,
    resolve_device_profile,
)
from governance.seed_import import (
    apply_canonical_ddl,
    import_colleague_seed,
    schema_version,
    spot_check_seed,
    verify_seed_counts,
)

OUTCOME_PRECEDENCE = {
    "REJECT": 5,
    "ESCALATE": 4,
    "POSTPONE": 3,
    "MODIFY": 2,
    "APPROVE": 1,
}

GOVERNANCE_DOMAINS = (
    "patient_safety",
    "gdpr_privacy",
    "mdr",
    "hospital_policy",
    "organisational_oversight",
    "audit",
)

ACTION_SEVERITY_ORDER = ["MONITOR", "ALERT", "THROTTLE", "BLOCK", "ISOLATE"]

_OP_ALIASES = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "in": "IN",
    "not_in": "NOT IN",
    "notin": "NOT IN",
}


@dataclass
class GovernanceDecision:
    outcome: str  # APPROVE / MODIFY / POSTPONE / ESCALATE / REJECT
    final_action: str  # UPPER DB action enum
    human_review: bool = False
    explanation: str = ""
    matched_rule_ids: List[int] = field(default_factory=list)
    matched_rule_versions: List[str] = field(default_factory=list)
    decision_id: Optional[int] = None
    substitute_action: Optional[str] = None
    review_status: str = "NOT_REQUIRED"
    execution_status: str = "EXECUTED"
    device_id: Optional[int] = None
    decisive_rule_id: Optional[int] = None
    matched_rules: List[Dict[str, Any]] = field(default_factory=list)


def _parse_threshold(op: str, raw: Any) -> Any:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if op in ("IN", "NOT IN"):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts
    # boolean
    up = s.upper()
    if up in {"TRUE", "T", "YES"}:
        return True
    if up in {"FALSE", "F", "NO"}:
        return False
    # numeric
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _normalize_left(left: Any, right: Any) -> Any:
    if left is None:
        return None
    if isinstance(right, bool):
        if isinstance(left, bool):
            return left
        if isinstance(left, (int, float)) and not isinstance(left, bool):
            return bool(left)
        return str(left).strip().upper() in {"TRUE", "T", "1", "YES"}
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return float(left)
        except (TypeError, ValueError):
            return left
    if isinstance(right, list):
        # IN list of enums — upper-case left for compare
        return str(left).strip().upper() if not isinstance(left, (int, float, bool)) else left
    if isinstance(right, str):
        # enum-ish compare
        if right.upper() in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
            "MONITOR",
            "ALERT",
            "THROTTLE",
            "BLOCK",
            "ISOLATE",
            "NO_ACTION",
            "PENDING",
            "TRUE",
            "FALSE",
        } or (isinstance(left, str) and left.upper() == left):
            return str(left).strip().upper()
        return left
    return left


def cmp_condition(op: str, left: Any, threshold_raw: Any) -> bool:
    """Null-safe colleague operator compare. Missing left ⇒ False."""
    op_n = _OP_ALIASES.get(str(op).strip().lower(), str(op).strip().upper())
    if op_n not in {"=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN"}:
        logging.getLogger(__name__).warning("unknown governance operator %r → False", op)
        return False

    if left is None:
        return False

    right = _parse_threshold(op_n, threshold_raw)
    if right is None and op_n in ("IN", "NOT IN", "=", "!=", ">", ">=", "<", "<="):
        if op_n in ("IN", "NOT IN") or threshold_raw is None or str(threshold_raw).strip() == "":
            logging.getLogger(__name__).warning(
                "empty governance threshold for op %s → False", op_n
            )
            return False

    if op_n in ("IN", "NOT IN"):
        if not right:
            return False
        vals = []
        for v in right:
            pv = _parse_threshold("=", v)
            if isinstance(pv, bool):
                vals.append(pv)
            else:
                vals.append(str(v).strip().upper())
        sample = vals[0] if vals else ""
        left_n = _normalize_left(left, sample)
        if isinstance(sample, bool):
            left_cmp: Any = bool(left_n)
        else:
            if isinstance(left_n, bool):
                left_cmp = "TRUE" if left_n else "FALSE"
            elif isinstance(left_n, (int, float)):
                left_cmp = str(left_n)
            else:
                left_cmp = str(left_n).strip().upper()
            vals = [
                ("TRUE" if v else "FALSE") if isinstance(v, bool) else str(v).strip().upper()
                for v in vals
            ]
        hit = left_cmp in vals
        return (not hit) if op_n == "NOT IN" else hit

    left_n = _normalize_left(left, right)

    # Boolean equality must respect both sides
    if isinstance(right, bool) or isinstance(left_n, bool):
        lb = bool(left_n) if not isinstance(left_n, bool) else left_n
        if isinstance(right, bool):
            rb = right
        else:
            rb = _parse_threshold("=", right)
            if not isinstance(rb, bool):
                rb = str(right).strip().upper() in {"TRUE", "T", "1", "YES"}
        if op_n == "=":
            return lb == rb
        if op_n == "!=":
            return lb != rb
        return False

    try:
        if op_n == "=":
            if isinstance(right, str) and isinstance(left_n, str):
                return left_n.upper() == right.upper()
            return left_n == right
        if op_n == "!=":
            if isinstance(right, str) and isinstance(left_n, str):
                return left_n.upper() != right.upper()
            return left_n != right
        lf = float(left_n)
        rf = float(right)
        if op_n == ">":
            return lf > rf
        if op_n == ">=":
            return lf >= rf
        if op_n == "<":
            return lf < rf
        if op_n == "<=":
            return lf <= rf
    except (TypeError, ValueError):
        return False
    return False


def resolve_final_action(outcome: str, rl_action: str, substitute: Optional[str]) -> str:
    oc = str(outcome).upper()
    rl = normalize_db_action(rl_action)
    sub = normalize_db_action(substitute) if substitute else None
    if oc == "APPROVE":
        return rl
    if oc == "MODIFY":
        return sub or "ALERT"
    if oc == "REJECT":
        return sub or "NO_ACTION"
    if oc in ("ESCALATE", "POSTPONE"):
        return sub or "MONITOR"
    return rl


def clamp_to_permitted(
    action: str,
    permitted_csv: Optional[str],
    *,
    life_support: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    If action not permitted, escalate to MONITOR on life-support; else downgrade
    to max permitted by severity order. Returns (action, outcome_override_or_None).
    """
    act = normalize_db_action(action)
    if act == "NO_ACTION":
        return act, None
    if not permitted_csv:
        return act, None
    permitted = {
        normalize_db_action(p) for p in str(permitted_csv).split(",") if p.strip()
    }
    permitted.discard("NO_ACTION")
    if act in permitted:
        return act, None
    if life_support or act in {"BLOCK", "ISOLATE"}:
        return "MONITOR", "ESCALATE"
    # Downgrade to strongest permitted ≤ requested
    if act not in ACTION_SEVERITY_ORDER:
        return "MONITOR", None
    want = ACTION_SEVERITY_ORDER.index(act)
    for cand in reversed(ACTION_SEVERITY_ORDER[: want + 1]):
        if cand in permitted:
            return cand, None
    # nothing weaker permitted
    for cand in ACTION_SEVERITY_ORDER:
        if cand in permitted:
            return cand, None
    return "MONITOR", "ESCALATE"


class PolicyRepository:
    """SQLite policy store — colleague Table VI schema (schema_version=colleague_v1)."""

    runtime_api_ready: bool = True

    def __init__(
        self,
        db_path: str | Path,
        *,
        seed_path: Optional[str | Path] = None,
        device_aliases_path: Optional[str | Path] = None,
        device_roster_path: Optional[str | Path] = None,
        recreate_if_legacy: bool = True,
        experiment_run_id: Optional[str] = None,
        experiment_seed: Optional[int] = None,
    ):
        self.db_path = Path(db_path)
        self.seed_path = Path(seed_path) if seed_path else None
        self.experiment_run_id = experiment_run_id
        self.experiment_seed = experiment_seed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if recreate_if_legacy and self.db_path.exists() and self._is_legacy_db_file():
            bak = self.db_path.with_suffix(self.db_path.suffix + ".pre_canonical.bak")
            if not bak.exists():
                self.db_path.replace(bak)
            else:
                self.db_path.unlink()
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self.device_aliases = load_device_aliases(device_aliases_path)
        self.device_roster = load_device_roster(device_roster_path)
        self.log_condition_evaluations = True  # write condition_evaluation_log for matched rules
        self.initialize_schema()
        self._profile_cache_built = False
        self._profiles_by_id: Dict[str, Dict[str, Any]] = {}
        self._profiles_by_type: Dict[str, List[Dict[str, Any]]] = {}
        self._all_profiles: List[Dict[str, Any]] = []

    def _is_legacy_db_file(self) -> bool:
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                ver = schema_version(conn)
                if ver == SCHEMA_VERSION:
                    return False
                cols = {
                    r[1]
                    for r in conn.execute("PRAGMA table_info(governance_rules)").fetchall()
                }
                if "recommended_outcome" in cols:
                    return False
                if "outcome" in cols or not cols:
                    return True
                return ver != SCHEMA_VERSION
            finally:
                conn.close()
        except sqlite3.Error:
            return True

    def close(self) -> None:
        self._conn.close()

    def initialize_schema(self) -> None:
        apply_canonical_ddl(self._conn)
        import_colleague_seed(self._conn, self.seed_path)
        verify_seed_counts(self._conn)
        spot_check_seed(self._conn)

    def _ensure_profile_index(self) -> None:
        if self._profile_cache_built:
            return
        rows = [
            dict(r)
            for r in self._conn.execute("SELECT * FROM medical_device_profiles")
        ]
        self._all_profiles = rows
        self._profiles_by_id = {str(r["device_identifier"]): r for r in rows}
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_type.setdefault(str(r["device_type"]).upper(), []).append(r)
        self._profiles_by_type = by_type
        self._profile_cache_built = True

    @property
    def schema_version(self) -> str:
        return schema_version(self._conn) or SCHEMA_VERSION

    def seed_counts(self) -> Dict[str, int]:
        return verify_seed_counts(self._conn)

    def rule_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM governance_rules WHERE rule_status='ACTIVE'"
        ).fetchone()
        return int(row["n"]) if row else 0

    def executable_rule_count(self) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(DISTINCT r.rule_id) AS n
            FROM governance_rules r
            JOIN rule_conditions c ON c.rule_id = r.rule_id
            WHERE r.rule_status='ACTIVE' AND c.status='ACTIVE'
            """
        ).fetchone()
        return int(row["n"]) if row else 0

    def seed_default_iomt_policies(self) -> None:
        raise RuntimeError(
            "seed_default_iomt_policies() is disabled; "
            "canonical colleague seed is imported via seed_import.import_colleague_seed"
        )

    def get_device_profile(self, device_type: Optional[str]) -> Optional[Dict[str, Any]]:
        if not device_type:
            return None
        row = self._conn.execute(
            """
            SELECT * FROM medical_device_profiles
            WHERE upper(device_type)=upper(?) AND device_status='ACTIVE'
            ORDER BY device_id ASC LIMIT 1
            """,
            (device_type,),
        ).fetchone()
        return dict(row) if row else None

    def list_devices_by_type(self, device_type: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM medical_device_profiles
            WHERE upper(device_type)=upper(?)
            ORDER BY device_id ASC
            """,
            (device_type,),
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_profile_for_context(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._ensure_profile_index()
        return resolve_device_profile(
            profiles_by_identifier=self._profiles_by_id,
            profiles_by_type=self._profiles_by_type,
            all_active_profiles=self._all_profiles,
            participant_id=str(raw.get("participant_id") or raw.get("agent_id") or ""),
            detection_device_type=str(raw.get("device_type") or ""),
            aliases=self.device_aliases,
            roster=self.device_roster,
        )

    def _load_active_rules(self) -> List[Dict[str, Any]]:
        rules = []
        for row in self._conn.execute(
            """
            SELECT * FROM governance_rules
            WHERE rule_status='ACTIVE'
            ORDER BY priority ASC, rule_id ASC
            """
        ):
            rule = dict(row)
            conds = [
                dict(c)
                for c in self._conn.execute(
                    """
                    SELECT * FROM rule_conditions
                    WHERE rule_id=? AND status='ACTIVE'
                    ORDER BY condition_order ASC, condition_id ASC
                    """,
                    (rule["rule_id"],),
                )
            ]
            if not conds:
                continue
            rule["outcome"] = rule["recommended_outcome"]
            rule["human_review"] = bool(rule["human_review_required"])
            rule["description"] = rule.get("rule_name") or ""
            rule["conditions"] = conds
            rules.append(rule)
        return rules

    def _rule_matches(self, rule: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[bool, int, int]:
        conds = rule.get("conditions") or []
        if not conds:
            return False, 0, 0
        results: List[bool] = []
        connectors: List[str] = []
        for c in conds:
            field_name = c["context_field"]
            op = c["operator"]
            thr = c.get("threshold_value", c.get("threshold"))
            left = ctx.get(field_name)  # missing → None → False
            results.append(cmp_condition(op, left, thr))
            connectors.append(str(c.get("logical_connector") or "AND").upper())
        ok = results[0]
        for i in range(1, len(results)):
            if connectors[i] == "OR":
                ok = ok or results[i]
            else:
                ok = ok and results[i]
        matched_n = sum(1 for r in results if r)
        failed_n = len(results) - matched_n
        return bool(ok), matched_n, failed_n

    def evaluate(self, context: Dict[str, Any]) -> GovernanceDecision:
        raw = dict(context)
        profile = self.resolve_profile_for_context(raw)
        ctx = build_governance_context(raw, profile)

        matched: List[Dict[str, Any]] = []
        match_meta: Dict[int, Tuple[int, int]] = {}
        for rule in self._load_active_rules():
            ok, matched_n, failed_n = self._rule_matches(rule, ctx)
            if ok:
                matched.append(rule)
                match_meta[int(rule["rule_id"])] = (matched_n, failed_n)

        if not matched:
            final = normalize_db_action(ctx.get("rl_action", "MONITOR"))
            final, oc_over = clamp_to_permitted(
                final,
                ctx.get("permitted_automatic_actions"),
                life_support=bool(ctx.get("life_support_status")),
            )
            outcome = oc_over or "APPROVE"
            human = outcome in ("ESCALATE", "POSTPONE")
            return GovernanceDecision(
                outcome=outcome,
                final_action=final,
                human_review=human,
                explanation="no_matching_rule" if not oc_over else "permit_clamp_escalate",
                review_status="PENDING" if human else "NOT_REQUIRED",
                execution_status="EXECUTED",
                device_id=int(profile["device_id"]) if profile else None,
            )

        best = max(
            matched,
            key=lambda r: (
                OUTCOME_PRECEDENCE.get(str(r["recommended_outcome"]).upper(), 0),
                -int(r["priority"]),
                -int(r["rule_id"]),
            ),
        )
        outcome = str(best["recommended_outcome"]).upper()
        sub = best.get("substitute_action")
        final = resolve_final_action(outcome, str(ctx.get("rl_action", "MONITOR")), sub)
        final, oc_over = clamp_to_permitted(
            final,
            ctx.get("permitted_automatic_actions"),
            life_support=bool(ctx.get("life_support_status")),
        )
        if oc_over and OUTCOME_PRECEDENCE[oc_over] > OUTCOME_PRECEDENCE.get(outcome, 0):
            outcome = oc_over

        human = bool(best.get("human_review_required")) or outcome in ("ESCALATE", "POSTPONE")
        if outcome == "POSTPONE":
            exec_status = "WITHHELD"
        else:
            exec_status = "EXECUTED"

        explanation = best.get("rule_name") or f"matched_rule_{best['rule_id']}"
        return GovernanceDecision(
            outcome=outcome,
            final_action=final,
            human_review=human,
            explanation=explanation,
            matched_rule_ids=[int(r["rule_id"]) for r in matched],
            matched_rule_versions=[str(r.get("version", "1.0")) for r in matched],
            substitute_action=normalize_db_action(sub) if sub else None,
            review_status="PENDING" if human else "NOT_REQUIRED",
            execution_status=exec_status,
            device_id=int(profile["device_id"]) if profile else None,
            decisive_rule_id=int(best["rule_id"]),
            matched_rules=[
                {
                    **r,
                    "matched_condition_count": match_meta[int(r["rule_id"])][0],
                    "failed_condition_count": match_meta[int(r["rule_id"])][1],
                }
                for r in matched
            ],
        )

    def record_decision(
        self,
        decision: GovernanceDecision,
        context: Dict[str, Any],
        round_num: int = 0,
        agent_id: str = "",
        *,
        experiment_run_id: Optional[str] = None,
        experiment_seed: Optional[int] = None,
    ) -> int:
        profile = self.resolve_profile_for_context(
            {**context, "participant_id": agent_id or context.get("participant_id")}
        )
        ctx = build_governance_context(
            {**context, "participant_id": agent_id or context.get("participant_id")},
            profile,
        )
        device_id = decision.device_id or (int(profile["device_id"]) if profile else None)
        if device_id is None:
            # FK requires device — bind fallback profile
            self._ensure_profile_index()
            if self._all_profiles:
                device_id = int(self._all_profiles[0]["device_id"])

        final = normalize_db_action(decision.final_action)
        outcome = str(decision.outcome).upper()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        run_id = experiment_run_id or self.experiment_run_id
        seed = experiment_seed if experiment_seed is not None else self.experiment_seed

        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO governance_decisions(
                    device_id, participant_id, rl_recommendation,
                    trust_score, risk_score, attack_confidence, patient_safety_risk,
                    attack_type, attack_severity, governance_outcome, final_action,
                    explanation, human_review_required, review_status, execution_status,
                    decision_timestamp, experiment_round, experiment_seed, experiment_run_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    device_id,
                    agent_id or ctx.get("participant_id") or "",
                    str(ctx.get("rl_action") or context.get("rl_action") or "MONITOR")
                    .strip()
                    .upper(),
                    clamp01(ctx.get("trust_score", 0.0)),
                    clamp01(ctx.get("risk_score", 0.0)),
                    clamp01(ctx.get("attack_confidence", 0.0)),
                    clamp01(ctx.get("patient_safety_risk", 0.0)),
                    str(ctx.get("attack_type") or "ANOMALOUS_TRAFFIC"),
                    str(ctx.get("attack_severity") or "LOW"),
                    outcome,
                    final,
                    decision.explanation,
                    1 if decision.human_review else 0,
                    decision.review_status or ("PENDING" if decision.human_review else "NOT_REQUIRED"),
                    decision.execution_status or "EXECUTED",
                    ts,
                    int(round_num),
                    seed,
                    run_id,
                ),
            )
            decision_id = int(cur.lastrowid)
            decisive = decision.decisive_rule_id
            rules = decision.matched_rules or []
            if not rules and decision.matched_rule_ids:
                for rid, ver in zip(decision.matched_rule_ids, decision.matched_rule_versions or []):
                    rules.append(
                        {
                            "rule_id": rid,
                            "version": ver,
                            "recommended_outcome": outcome,
                            "substitute_action": decision.substitute_action,
                            "priority": None,
                            "matched_condition_count": 0,
                            "failed_condition_count": 0,
                        }
                    )
            for r in rules:
                rid = int(r["rule_id"])
                cur.execute(
                    """
                    INSERT INTO decision_rule_matches(
                        decision_id, rule_id, rule_version, evaluation_result,
                        match_priority, contributed_outcome, contributed_action,
                        evaluation_details, failed_condition_count, matched_condition_count,
                        decisive_rule, evaluated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        decision_id,
                        rid,
                        str(r.get("version", "1.0")),
                        1,
                        r.get("priority"),
                        str(r.get("recommended_outcome") or outcome).upper(),
                        normalize_db_action(r.get("substitute_action"))
                        if r.get("substitute_action")
                        else None,
                        f"rule_{rid}",
                        int(r.get("failed_condition_count") or 0),
                        int(r.get("matched_condition_count") or 0),
                        1 if decisive is not None and rid == decisive else 0,
                        ts,
                    ),
                )
                match_id = int(cur.lastrowid)
                if self.log_condition_evaluations:
                    conds = r.get("conditions") or []
                    if not conds:
                        conds = [
                            dict(c)
                            for c in self._conn.execute(
                                """
                                SELECT * FROM rule_conditions
                                WHERE rule_id=? AND status='ACTIVE'
                                ORDER BY condition_order, condition_id
                                """,
                                (rid,),
                            )
                        ]
                    for c in conds:
                        field_name = c["context_field"]
                        op = c["operator"]
                        thr = c.get("threshold_value", c.get("threshold"))
                        observed = ctx.get(field_name)
                        passed = cmp_condition(op, observed, thr)
                        if observed is None:
                            value_type = "STRING"
                        elif isinstance(observed, bool):
                            value_type = "BOOLEAN"
                        elif isinstance(observed, int) and not isinstance(observed, bool):
                            value_type = "INTEGER"
                        elif isinstance(observed, float):
                            value_type = "DECIMAL"
                        else:
                            value_type = "STRING"
                        cur.execute(
                            """
                            INSERT INTO condition_evaluation_log(
                                match_id, condition_id, context_field, operator,
                                expected_value, observed_value, value_type,
                                evaluation_result, data_origin, experiment_run_id,
                                scenario_label, evaluation_details, evaluated_at
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                match_id,
                                int(c["condition_id"]),
                                field_name,
                                op,
                                str(thr) if thr is not None else None,
                                None if observed is None else str(observed),
                                value_type,
                                1 if passed else 0,
                                "SIMULATION",
                                run_id,
                                f"round_{round_num}",
                                None,
                                ts,
                            ),
                        )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        decision.decision_id = decision_id
        decision.device_id = device_id
        return decision_id

    def human_review_rate(self, experiment_run_id: Optional[str] = None) -> float:
        run_id = experiment_run_id or self.experiment_run_id
        if run_id:
            row = self._conn.execute(
                """
                SELECT AVG(human_review_required) AS r FROM governance_decisions
                WHERE experiment_run_id=?
                """,
                (run_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT AVG(human_review_required) AS r FROM governance_decisions"
            ).fetchone()
        if not row or row["r"] is None:
            return 0.0
        return float(row["r"])
