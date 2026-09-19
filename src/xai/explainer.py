"""Explainable AI helpers for TrustFed-RL (master Contrib. 5 / §H).

WP3: forensic exports read colleague-compatible Table VI columns and optional
condition_evaluation_log rows.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SIGNAL_ORDER = ("V", "S", "D", "U", "C", "R")


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def explain_trust(
    signals: Dict[str, Any],
    weights: Dict[str, float],
    *,
    fused_trust: Optional[float] = None,
    rl_delta: float = 0.0,
) -> Dict[str, Any]:
    """Decompose T = Σ w_k x_k (+ optional RL calibration delta)."""
    xs: Dict[str, float] = {}
    for k in SIGNAL_ORDER:
        xs[k] = _as_float(signals.get(k, signals.get(
            {"V": "accuracy", "S": "stability", "D": "drift", "U": "uncertainty",
             "C": "communication", "R": "contextual_safety"}.get(k, k),
            0.0,
        )))
    ws = {k: _as_float(weights.get(k, 0.0)) for k in SIGNAL_ORDER}
    contributions = {k: ws[k] * xs[k] for k in SIGNAL_ORDER}
    predicted = float(sum(contributions.values()))
    return {
        "formula": "T = sum_k w_k * x_k (+ rl_delta)",
        "signals": xs,
        "weights": ws,
        "contributions": contributions,
        "predicted_T": predicted,
        "rl_delta": float(rl_delta),
        "reported_T": _as_float(fused_trust, predicted) if fused_trust is not None else predicted,
        "dominant_signal": max(contributions, key=contributions.get) if contributions else None,
        "weakest_signal": min(xs, key=xs.get) if xs else None,
    }


def load_audit_events(audit_path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    path = Path(audit_path)
    if not path.exists():
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def filter_agent_events(
    events: Iterable[Dict[str, Any]],
    agent_id: str,
    *,
    round_num: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out = []
    for e in events:
        if e.get("agent_id") != agent_id:
            continue
        if round_num is not None and e.get("round") != round_num:
            continue
        out.append(e)
    return out


def _row_keys(row: sqlite3.Row) -> set:
    return set(row.keys())


def _fetch_condition_logs(
    conn: sqlite3.Connection,
    decision_id: int,
) -> List[Dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT cel.evaluation_id, cel.match_id, cel.condition_id, cel.context_field,
                   cel.operator, cel.expected_value, cel.observed_value, cel.value_type,
                   cel.evaluation_result, cel.data_origin, cel.experiment_run_id,
                   cel.scenario_label, cel.evaluated_at,
                   drm.rule_id, drm.decisive_rule
            FROM condition_evaluation_log cel
            JOIN decision_rule_matches drm ON drm.match_id = cel.match_id
            WHERE drm.decision_id = ?
            ORDER BY cel.evaluation_id ASC
            """,
            (int(decision_id),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        out.append(
            {
                "evaluation_id": int(r["evaluation_id"]),
                "match_id": int(r["match_id"]),
                "condition_id": int(r["condition_id"]),
                "rule_id": int(r["rule_id"]),
                "decisive_rule": bool(r["decisive_rule"]),
                "context_field": r["context_field"],
                "operator": r["operator"],
                "expected_value": r["expected_value"],
                "observed_value": r["observed_value"],
                "value_type": r["value_type"],
                "passed": bool(r["evaluation_result"]),
                "data_origin": r["data_origin"],
                "experiment_run_id": r["experiment_run_id"],
                "scenario_label": r["scenario_label"],
                "evaluated_at": r["evaluated_at"],
            }
        )
    return out


def _enrich_matches(
    conn: sqlite3.Connection,
    matches: Sequence[sqlite3.Row],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in matches:
        mk = _row_keys(m)
        rid = int(m["rule_id"])
        rule_name = domain = None
        try:
            rrow = conn.execute(
                """
                SELECT rule_name, domain, recommended_outcome, substitute_action, priority
                FROM governance_rules WHERE rule_id=?
                """,
                (rid,),
            ).fetchone()
            if rrow:
                rule_name = rrow["rule_name"]
                domain = rrow["domain"]
        except sqlite3.OperationalError:
            pass
        item = {
            "rule_id": rid,
            "rule_version": m["rule_version"],
            "evaluation_result": m["evaluation_result"],
            "decisive_rule": bool(m["decisive_rule"]) if "decisive_rule" in mk else None,
            "rule_name": rule_name,
            "domain": domain,
        }
        if "contributed_outcome" in mk:
            item["contributed_outcome"] = m["contributed_outcome"]
        if "contributed_action" in mk:
            item["contributed_action"] = m["contributed_action"]
        if "matched_condition_count" in mk:
            item["matched_condition_count"] = m["matched_condition_count"]
        if "failed_condition_count" in mk:
            item["failed_condition_count"] = m["failed_condition_count"]
        out.append(item)
    return out


def fetch_governance_decision(
    db_path: Path,
    decision_id: int,
    *,
    include_condition_log: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load one governance decision in canonical + legacy-compatible form."""
    path = Path(db_path)
    if not path.exists() or decision_id is None:
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM governance_decisions WHERE decision_id = ?",
            (int(decision_id),),
        ).fetchone()
        if row is None:
            return None
        try:
            matches = conn.execute(
                """
                SELECT * FROM decision_rule_matches WHERE decision_id = ?
                ORDER BY match_id ASC
                """,
                (int(decision_id),),
            ).fetchall()
        except sqlite3.OperationalError:
            matches = []

        keys = _row_keys(row)
        round_num = row["experiment_round"] if "experiment_round" in keys else row["round_num"]
        participant_id = (
            row["participant_id"] if "participant_id" in keys else row["agent_id"]
        )
        governance_outcome = (
            row["governance_outcome"] if "governance_outcome" in keys else row["matched_outcome"]
        )
        human_review_required = (
            row["human_review_required"] if "human_review_required" in keys else row["human_review"]
        )
        trust = row["trust_score"] if "trust_score" in keys else row["trust"]
        if "risk_score" in keys:
            risk = row["risk_score"]
        elif "contextual_risk" in keys:
            risk = row["contextual_risk"]
        else:
            risk = 0.0
        ts = row["decision_timestamp"] if "decision_timestamp" in keys else row["timestamp"]
        device_type = row["device_type"] if "device_type" in keys else None
        device_name = device_identifier = criticality = None
        life_support_status = None
        permitted_automatic_actions = None
        regulated_medical_device = None
        if "device_id" in keys and row["device_id"] is not None:
            try:
                drow = conn.execute(
                    """
                    SELECT device_type, device_name, device_identifier, criticality,
                           life_support_status, permitted_automatic_actions,
                           regulated_medical_device, patient_data_access, device_status
                    FROM medical_device_profiles WHERE device_id=?
                    """,
                    (row["device_id"],),
                ).fetchone()
                if drow:
                    device_type = device_type or drow["device_type"]
                    device_name = drow["device_name"]
                    device_identifier = drow["device_identifier"]
                    criticality = drow["criticality"]
                    life_support_status = bool(drow["life_support_status"])
                    permitted_automatic_actions = drow["permitted_automatic_actions"]
                    regulated_medical_device = bool(drow["regulated_medical_device"])
            except sqlite3.OperationalError:
                pass
        if criticality is None and "device_criticality" in keys:
            criticality = row["device_criticality"]

        matched = _enrich_matches(conn, matches)
        decisive = next((m for m in matched if m.get("decisive_rule")), None)
        condition_log = (
            _fetch_condition_logs(conn, int(decision_id)) if include_condition_log else []
        )
        # Attach condition_description when available
        if condition_log:
            try:
                for entry in condition_log:
                    crow = conn.execute(
                        """
                        SELECT condition_description FROM rule_conditions
                        WHERE condition_id=?
                        """,
                        (entry["condition_id"],),
                    ).fetchone()
                    if crow:
                        entry["condition_description"] = crow["condition_description"]
            except sqlite3.OperationalError:
                pass

        payload = {
            # Canonical Table VI / colleague_v1
            "schema": "colleague_v1",
            "decision_id": int(row["decision_id"]),
            "device_id": int(row["device_id"])
            if "device_id" in keys and row["device_id"] is not None
            else None,
            "device_identifier": device_identifier,
            "device_name": device_name,
            "device_type": device_type,
            "device_criticality": criticality,
            "life_support_status": life_support_status,
            "regulated_medical_device": regulated_medical_device,
            "permitted_automatic_actions": permitted_automatic_actions,
            "participant_id": participant_id,
            "rl_recommendation": row["rl_recommendation"],
            "trust_score": float(trust or 0.0),
            "risk_score": float(risk or 0.0),
            "attack_confidence": float(row["attack_confidence"] or 0.0)
            if "attack_confidence" in keys
            else None,
            "patient_safety_risk": float(row["patient_safety_risk"] or 0.0)
            if "patient_safety_risk" in keys
            else None,
            "attack_type": row["attack_type"] if "attack_type" in keys else None,
            "attack_severity": row["attack_severity"],
            "governance_outcome": governance_outcome,
            "final_action": row["final_action"],
            "explanation": row["explanation"],
            "human_review_required": bool(human_review_required),
            "review_status": row["review_status"],
            "execution_status": row["execution_status"] if "execution_status" in keys else None,
            "decision_timestamp": ts,
            "experiment_round": int(round_num or 0),
            "experiment_seed": row["experiment_seed"] if "experiment_seed" in keys else None,
            "experiment_run_id": row["experiment_run_id"] if "experiment_run_id" in keys else None,
            "matched_rules": matched,
            "decisive_rule": decisive,
            "condition_evaluation_log": condition_log,
            # Legacy aliases (pre-colleague_v1 consumers / narrative helpers)
            "round_num": int(round_num or 0),
            "agent_id": participant_id,
            "matched_outcome": governance_outcome,
            "human_review": bool(human_review_required),
            "trust": float(trust or 0.0),
            "contextual_risk": float(risk or 0.0),
            "timestamp": ts,
        }
        return payload
    finally:
        conn.close()


def find_decision_ids(
    db_path: Path,
    *,
    participant_id: Optional[str] = None,
    experiment_round: Optional[int] = None,
    experiment_run_id: Optional[str] = None,
    limit: int = 50,
) -> List[int]:
    """List decision_ids matching participant/round/run filters."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    try:
        clauses = []
        params: List[Any] = []
        # detect schema
        cols = {r[1] for r in conn.execute("PRAGMA table_info(governance_decisions)")}
        if participant_id:
            if "participant_id" in cols:
                clauses.append("participant_id=?")
            else:
                clauses.append("agent_id=?")
            params.append(participant_id)
        if experiment_round is not None:
            if "experiment_round" in cols:
                clauses.append("experiment_round=?")
            else:
                clauses.append("round_num=?")
            params.append(int(experiment_round))
        if experiment_run_id and "experiment_run_id" in cols:
            clauses.append("experiment_run_id=?")
            params.append(experiment_run_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT decision_id FROM governance_decisions{where} ORDER BY decision_id ASC LIMIT ?"
        params.append(int(limit))
        return [int(r[0]) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def summarize_governance_db(
    db_path: Path,
    *,
    experiment_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate governance metrics from the policy DB for experiment reports."""
    path = Path(db_path)
    empty = {
        "schema": "colleague_v1",
        "n_decisions": 0,
        "human_review_rate": 0.0,
        "outcome_counts": {},
        "final_action_counts": {},
        "n_matches": 0,
        "n_condition_logs": 0,
        "experiment_run_id": experiment_run_id,
    }
    if not path.exists():
        return empty
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(governance_decisions)")}
        if "governance_outcome" not in cols and "matched_outcome" not in cols:
            return empty
        where = ""
        params: List[Any] = []
        if experiment_run_id and "experiment_run_id" in cols:
            where = " WHERE experiment_run_id=?"
            params = [experiment_run_id]
        outcome_col = "governance_outcome" if "governance_outcome" in cols else "matched_outcome"
        hr_col = (
            "human_review_required" if "human_review_required" in cols else "human_review"
        )
        n = conn.execute(
            f"SELECT COUNT(*) AS n FROM governance_decisions{where}", params
        ).fetchone()["n"]
        hr = conn.execute(
            f"SELECT AVG({hr_col}) AS r FROM governance_decisions{where}", params
        ).fetchone()["r"]
        outcomes = {
            str(r[0]): int(r[1])
            for r in conn.execute(
                f"SELECT {outcome_col}, COUNT(*) FROM governance_decisions{where} GROUP BY 1",
                params,
            )
        }
        actions = {
            str(r[0]): int(r[1])
            for r in conn.execute(
                f"SELECT final_action, COUNT(*) FROM governance_decisions{where} GROUP BY 1",
                params,
            )
        }
        try:
            if where:
                n_matches = conn.execute(
                    """
                    SELECT COUNT(*) FROM decision_rule_matches m
                    JOIN governance_decisions d ON d.decision_id=m.decision_id
                    WHERE d.experiment_run_id=?
                    """,
                    params,
                ).fetchone()[0]
                n_logs = conn.execute(
                    """
                    SELECT COUNT(*) FROM condition_evaluation_log c
                    JOIN decision_rule_matches m ON m.match_id=c.match_id
                    JOIN governance_decisions d ON d.decision_id=m.decision_id
                    WHERE d.experiment_run_id=?
                    """,
                    params,
                ).fetchone()[0]
            else:
                n_matches = conn.execute(
                    "SELECT COUNT(*) FROM decision_rule_matches"
                ).fetchone()[0]
                n_logs = conn.execute(
                    "SELECT COUNT(*) FROM condition_evaluation_log"
                ).fetchone()[0]
        except sqlite3.OperationalError:
            n_matches, n_logs = 0, 0
        return {
            "schema": "colleague_v1" if "governance_outcome" in cols else "legacy",
            "n_decisions": int(n or 0),
            "human_review_rate": float(hr or 0.0),
            "outcome_counts": outcomes,
            "final_action_counts": actions,
            "n_matches": int(n_matches or 0),
            "n_condition_logs": int(n_logs or 0),
            "experiment_run_id": experiment_run_id,
            "policy_db": str(path.resolve()),
        }
    finally:
        conn.close()


def explain_response_from_events(
    events: Sequence[Dict[str, Any]],
    *,
    policy_db: Optional[Path] = None,
    include_condition_log: bool = True,
) -> Dict[str, Any]:
    """Build RL + governance explanation from audit events for one agent/round."""
    proposed = next((e for e in events if e.get("event_type") == "response_proposed"), None)
    executed = next(
        (
            e for e in events
            if e.get("event_type") in ("response_executed", "policy_violation")
        ),
        None,
    )
    trust_ev = next((e for e in events if e.get("event_type") == "trust_update"), None)

    rl_block = {
        "recommended": (proposed or executed or {}).get("action"),
        "confidence": (proposed or {}).get("confidence"),
        "device_type": (proposed or executed or {}).get("device_type"),
        "status": (proposed or executed or {}).get("status"),
        "reason": (proposed or executed or {}).get("reason"),
        "human_review": (proposed or executed or {}).get("human_review", False),
        "decision_id": (proposed or executed or {}).get("decision_id"),
        "matched_rule_ids": (proposed or executed or {}).get("matched_rule_ids"),
    }

    gov = None
    did = rl_block.get("decision_id")
    if policy_db is not None and did is not None:
        gov = fetch_governance_decision(
            policy_db, int(did), include_condition_log=include_condition_log
        )

    return {
        "trust_event": trust_ev,
        "rl": rl_block,
        "governance": gov
        or {
            "governance_outcome": rl_block.get("status"),
            "matched_outcome": rl_block.get("status"),
            "final_action": (executed or proposed or {}).get("action"),
            "explanation": rl_block.get("reason"),
            "human_review_required": rl_block.get("human_review"),
            "human_review": rl_block.get("human_review"),
            "matched_rule_ids": rl_block.get("matched_rule_ids"),
            "decision_id": did,
            "condition_evaluation_log": [],
        },
        "executed": executed,
    }


def build_forensic_record(
    *,
    agent_id: str,
    round_num: int,
    events: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    policy_db: Optional[Path] = None,
    audit_path: Optional[Path] = None,
    include_condition_log: bool = True,
    experiment_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-round forensic package: trust decomposition + RL + governance."""
    scoped = filter_agent_events(events, agent_id, round_num=round_num)
    bundle = explain_response_from_events(
        scoped, policy_db=policy_db, include_condition_log=include_condition_log
    )
    trust_ev = bundle.get("trust_event") or {}
    signals = trust_ev.get("signals") or {}
    trust_explain = explain_trust(
        signals,
        weights,
        fused_trust=trust_ev.get("trust"),
        rl_delta=_as_float(trust_ev.get("rl_delta", 0.0)),
    )
    gov = bundle["governance"] or {}
    round_decision_ids: List[int] = []
    if policy_db and Path(policy_db).exists():
        round_decision_ids = find_decision_ids(
            Path(policy_db),
            participant_id=agent_id,
            experiment_round=round_num,
            experiment_run_id=experiment_run_id,
            limit=100,
        )
    # If audit lacked decision_id, prefer latest decision in the round (not first)
    if policy_db and not gov.get("decision_id") and round_decision_ids:
        gov = fetch_governance_decision(
            Path(policy_db),
            round_decision_ids[-1],
            include_condition_log=include_condition_log,
        ) or gov
        bundle["governance"] = gov

    db_summary = None
    if policy_db and Path(policy_db).exists():
        db_summary = summarize_governance_db(
            Path(policy_db), experiment_run_id=experiment_run_id
        )

    return {
        "schema": "colleague_v1",
        "agent_id": agent_id,
        "participant_id": agent_id,
        "round": round_num,
        "experiment_run_id": experiment_run_id or gov.get("experiment_run_id"),
        "trust": trust_explain,
        "rl": bundle["rl"],
        "governance": gov,
        "round_decision_ids": round_decision_ids,
        "governance_db_summary": db_summary,
        "narrative": _narrative(agent_id, round_num, trust_explain, bundle),
        "sources": {
            "audit": str(audit_path) if audit_path else None,
            "policy_db": str(policy_db) if policy_db else None,
        },
    }


def _narrative(
    agent_id: str,
    round_num: int,
    trust: Dict[str, Any],
    bundle: Dict[str, Any],
) -> str:
    weak = trust.get("weakest_signal")
    dom = trust.get("dominant_signal")
    rl = bundle.get("rl") or {}
    gov = bundle.get("governance") or {}
    final = gov.get("final_action") or rl.get("recommended")
    outcome = (
        gov.get("governance_outcome")
        or gov.get("matched_outcome")
        or gov.get("outcome")
        or rl.get("status")
    )
    decisive = gov.get("decisive_rule") or {}
    rule_bit = ""
    if decisive.get("rule_id"):
        name = decisive.get("rule_name") or f"rule_{decisive['rule_id']}"
        rule_bit = f" Decisive rule: {name}."
    n_conds = len(gov.get("condition_evaluation_log") or [])
    cond_bit = f" Condition traces: {n_conds}." if n_conds else ""
    return (
        f"{agent_id} round {round_num}: fused trust T≈{trust.get('reported_T', 0):.3f} "
        f"(weakest={weak}, largest contribution={dom}). "
        f"RL recommended '{rl.get('recommended')}' → governance {outcome} "
        f"→ final action '{final}'.{rule_bit}{cond_bit}"
    )


def export_forensic_json(
    record: Dict[str, Any],
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path


def default_signal_weights(trust_config_path: Optional[Path] = None) -> Dict[str, float]:
    """Load master signal weights from trust_config.json (multi_signal.signal_weights)."""
    if not trust_config_path or not Path(trust_config_path).exists():
        raise FileNotFoundError(
            "trust_config.json required for XAI weights (master Table V); "
            f"missing: {trust_config_path}"
        )
    cfg = json.loads(Path(trust_config_path).read_text())
    w = (
        cfg.get("signal_weights")
        or (cfg.get("multi_signal") or {}).get("signal_weights")
        or {}
    )
    if not w:
        raise ValueError("trust_config.json has no multi_signal.signal_weights")
    return {k: float(w.get(k, 0.0)) for k in SIGNAL_ORDER}
