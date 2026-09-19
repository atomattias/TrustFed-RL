"""
Colleague-compatible Table VI schema for SQLite (schema_version=colleague_v1).

Source of truth: TrustFed_RL Postgres dump; see docs/GOVERNANCE_SCHEMA_ALIGNMENT_PLAN.md.
"""

from __future__ import annotations

SCHEMA_VERSION = "colleague_v1"

# Expected seed row counts after import (WP1 acceptance)
EXPECTED_SEED_COUNTS = {
    "policy_sources": 34,
    "governance_rules": 45,
    "rule_conditions": 31,
    "medical_device_profiles": 14,
}

CANONICAL_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_sources (
    policy_source_id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,
    jurisdiction TEXT,
    article_section TEXT,
    publication_year INTEGER,
    document_reference TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS governance_rules (
    rule_id INTEGER PRIMARY KEY,
    policy_source_id INTEGER NOT NULL,
    rule_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    priority INTEGER NOT NULL,
    recommended_outcome TEXT NOT NULL
        CHECK (recommended_outcome IN ('APPROVE','MODIFY','REJECT','ESCALATE','POSTPONE')),
    substitute_action TEXT
        CHECK (
            substitute_action IS NULL OR substitute_action IN
            ('MONITOR','ALERT','THROTTLE','BLOCK','ISOLATE','NO_ACTION')
        ),
    human_review_required INTEGER NOT NULL DEFAULT 0,
    version TEXT NOT NULL DEFAULT '1.0',
    rule_status TEXT NOT NULL DEFAULT 'ACTIVE',
    FOREIGN KEY (policy_source_id) REFERENCES policy_sources(policy_source_id)
);

CREATE TABLE IF NOT EXISTS rule_conditions (
    condition_id INTEGER PRIMARY KEY,
    rule_id INTEGER NOT NULL,
    context_field TEXT NOT NULL,
    operator TEXT NOT NULL
        CHECK (operator IN ('=','!=','>','>=','<','<=','IN','NOT IN')),
    threshold_value TEXT NOT NULL,
    logical_connector TEXT
        CHECK (logical_connector IS NULL OR logical_connector IN ('AND','OR')),
    condition_order INTEGER NOT NULL DEFAULT 1,
    condition_description TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    FOREIGN KEY (rule_id) REFERENCES governance_rules(rule_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS medical_device_profiles (
    device_id INTEGER PRIMARY KEY,
    device_identifier TEXT NOT NULL UNIQUE,
    device_name TEXT NOT NULL,
    device_type TEXT NOT NULL,
    clinical_function TEXT,
    criticality TEXT NOT NULL
        CHECK (criticality IN ('LOW','MEDIUM','HIGH','CRITICAL')),
    life_support_status INTEGER NOT NULL DEFAULT 0,
    regulated_medical_device INTEGER NOT NULL DEFAULT 1,
    active_clinical_use INTEGER NOT NULL DEFAULT 0,
    patient_data_access INTEGER NOT NULL DEFAULT 0,
    permitted_automatic_actions TEXT,
    location TEXT,
    network_segment TEXT,
    device_status TEXT NOT NULL DEFAULT 'ACTIVE'
        CHECK (device_status IN ('ACTIVE','INACTIVE','MAINTENANCE','QUARANTINED'))
);

CREATE TABLE IF NOT EXISTS governance_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER,
    participant_id TEXT,
    rl_recommendation TEXT,
    trust_score REAL CHECK (trust_score IS NULL OR (trust_score >= 0 AND trust_score <= 1)),
    risk_score REAL CHECK (risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 1)),
    attack_confidence REAL CHECK (attack_confidence IS NULL OR (attack_confidence >= 0 AND attack_confidence <= 1)),
    patient_safety_risk REAL CHECK (patient_safety_risk IS NULL OR (patient_safety_risk >= 0 AND patient_safety_risk <= 1)),
    attack_type TEXT,
    attack_severity TEXT,
    governance_outcome TEXT NOT NULL
        CHECK (governance_outcome IN ('APPROVE','MODIFY','REJECT','ESCALATE','POSTPONE')),
    final_action TEXT NOT NULL
        CHECK (final_action IN ('MONITOR','ALERT','THROTTLE','BLOCK','ISOLATE','NO_ACTION')),
    explanation TEXT,
    human_review_required INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'NOT_REQUIRED'
        CHECK (review_status IN ('NOT_REQUIRED','PENDING','APPROVED','REJECTED','OVERRIDDEN')),
    execution_status TEXT NOT NULL DEFAULT 'EXECUTED'
        CHECK (execution_status IN ('PENDING','EXECUTED','WITHHELD','FAILED')),
    decision_timestamp TEXT NOT NULL,
    experiment_round INTEGER,
    experiment_seed INTEGER,
    experiment_run_id TEXT,
    FOREIGN KEY (device_id) REFERENCES medical_device_profiles(device_id)
);

CREATE TABLE IF NOT EXISTS decision_rule_matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL,
    rule_id INTEGER NOT NULL,
    rule_version TEXT,
    evaluation_result INTEGER NOT NULL DEFAULT 1,
    match_priority INTEGER,
    contributed_outcome TEXT
        CHECK (
            contributed_outcome IS NULL OR contributed_outcome IN
            ('APPROVE','MODIFY','REJECT','ESCALATE','POSTPONE')
        ),
    contributed_action TEXT
        CHECK (
            contributed_action IS NULL OR contributed_action IN
            ('MONITOR','ALERT','THROTTLE','BLOCK','ISOLATE','NO_ACTION')
        ),
    evaluation_details TEXT,
    failed_condition_count INTEGER DEFAULT 0,
    matched_condition_count INTEGER DEFAULT 0,
    decisive_rule INTEGER NOT NULL DEFAULT 0,
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES governance_decisions(decision_id) ON DELETE CASCADE,
    FOREIGN KEY (rule_id) REFERENCES governance_rules(rule_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS condition_evaluation_log (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    condition_id INTEGER NOT NULL,
    context_field TEXT NOT NULL,
    operator TEXT NOT NULL,
    expected_value TEXT,
    observed_value TEXT,
    value_type TEXT
        CHECK (
            value_type IS NULL OR value_type IN
            ('STRING','INTEGER','DECIMAL','BOOLEAN')
        ),
    evaluation_result INTEGER NOT NULL,
    data_origin TEXT
        CHECK (
            data_origin IS NULL OR data_origin IN
            ('SAMPLE','SYNTHETIC_TEST','SIMULATION','ACTUAL_TEST','PILOT_DEPLOYMENT')
        ),
    experiment_run_id TEXT,
    scenario_label TEXT,
    evaluation_details TEXT,
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES decision_rule_matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (condition_id) REFERENCES rule_conditions(condition_id) ON DELETE CASCADE
);
"""
