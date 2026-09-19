"""
Governance decision context builder (WP2).

Builds colleague-compatible context fields from TrustFed-RL detections + device profiles.
Defaults follow docs/GOVERNANCE_SCHEMA_ALIGNMENT_PLAN.md §5.1 (provisional WP0).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

CRITICALITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_LABELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
DB_ACTIONS = frozenset({"MONITOR", "ALERT", "THROTTLE", "BLOCK", "ISOLATE", "NO_ACTION"})
RL_ACTIONS = frozenset({"monitor", "alert", "throttle", "block", "isolate", "escalate"})


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return max(0.0, min(1.0, v))


def bin_attack_severity(score: Any) -> str:
    """Map float confidence/severity ∈ [0,1] → colleague categorical label."""
    v = clamp01(score, 0.0)
    if v >= 0.85:
        return "CRITICAL"
    if v >= 0.70:
        return "HIGH"
    if v >= 0.40:
        return "MEDIUM"
    return "LOW"


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().upper()
    return s in {"1", "T", "TRUE", "YES", "Y"}


def normalize_db_action(action: Any) -> str:
    a = str(action or "MONITOR").strip().upper()
    if a == "ESCALATE":
        # escalate is RL-only; never a final_action
        return "MONITOR"
    if a in DB_ACTIONS:
        return a
    return "MONITOR"


def normalize_rl_action(action: Any) -> str:
    a = str(action or "monitor").strip().lower()
    if a == "no_action":
        return "monitor"
    return a


def map_attack_type(attack_class: Optional[str]) -> str:
    if not attack_class:
        return "ANOMALOUS_TRAFFIC"
    key = str(attack_class).strip().upper().replace(" ", "_").replace("-", "_")
    known = {
        "MALWARE",
        "RANSOMWARE",
        "COMMAND_AND_CONTROL",
        "ANOMALOUS_TRAFFIC",
        "DATA_EXFILTRATION",
        "MODEL_POISONING",
        "CREDENTIAL_COMPROMISE",
        "BOTNET_ACTIVITY",
    }
    aliases = {
        "C2": "COMMAND_AND_CONTROL",
        "CN_C": "COMMAND_AND_CONTROL",
        "EXFIL": "DATA_EXFILTRATION",
        "POISONING": "MODEL_POISONING",
        "POISON": "MODEL_POISONING",
    }
    if key in known:
        return key
    if key in aliases:
        return aliases[key]
    for k in known:
        if k in key or key in k:
            return k
    return "ANOMALOUS_TRAFFIC"


def _criticality_from_profile(profile: Optional[Mapping[str, Any]], fallback_float: float) -> str:
    if profile and profile.get("criticality"):
        c = str(profile["criticality"]).upper()
        if c in CRITICALITY_RANK:
            return c
    # float MDP C_i → categorical fallback
    v = clamp01(fallback_float, 0.5)
    if v >= 0.9:
        return "CRITICAL"
    if v >= 0.7:
        return "HIGH"
    if v >= 0.4:
        return "MEDIUM"
    return "LOW"


def derive_patient_safety_risk(criticality: str, severity_label: str, life_support: bool) -> float:
    base = 0.15 + 0.2 * CRITICALITY_RANK.get(criticality, 1) + 0.15 * CRITICALITY_RANK.get(
        severity_label, 0
    )
    if life_support:
        base += 0.25
    return clamp01(base)


def derive_clinical_harm_risk(criticality: str, severity_label: str, life_support: bool) -> float:
    return derive_patient_safety_risk(criticality, severity_label, life_support)


def build_governance_context(
    raw: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Expand a validator/runner payload into colleague condition fields.

    `raw` typical keys: rl_action, trust/trust_score, attack_severity (float),
    contextual_risk, device_type (detection), participant_id, attack_class,
    attack_confidence, escalation_required, overrides for any context field.
    """
    rl = normalize_rl_action(raw.get("rl_action", "monitor")).upper()
    # RL escalate stays as recommendation token for conditions that use rl_action IN (...)
    # Colleague conditions use MONITOR..ISOLATE; escalate recommendations won't match IN lists
    # that omit it — which is correct. Keep UPPER of raw for condition compare:
    rl_for_ctx = str(raw.get("rl_action", "monitor")).strip().upper()
    if rl_for_ctx == "NO_ACTION":
        rl_for_ctx = "MONITOR"

    trust = clamp01(raw.get("trust_score", raw.get("trust", 0.0)))
    risk = clamp01(raw.get("risk_score", raw.get("contextual_risk", 0.0)))
    conf = clamp01(
        raw.get(
            "attack_confidence",
            raw.get("attack_severity", raw.get("confidence", 0.0)),
        )
    )
    # Prefer explicit categorical if caller already set it
    sev_raw = raw.get("attack_severity_label", raw.get("attack_severity"))
    if isinstance(sev_raw, str) and sev_raw.strip().upper() in SEVERITY_LABELS:
        severity_label = sev_raw.strip().upper()
    else:
        severity_label = bin_attack_severity(sev_raw if sev_raw is not None else conf)

    crit = _criticality_from_profile(profile, float(raw.get("device_criticality", 0.5) or 0.5))
    life = to_bool(profile.get("life_support_status")) if profile else False
    regulated = to_bool(profile.get("regulated_medical_device", True)) if profile else True
    patient_data = to_bool(profile.get("patient_data_access")) if profile else False
    active_clinical = to_bool(profile.get("active_clinical_use")) if profile else False

    patient_safety = derive_patient_safety_risk(crit, severity_label, life)
    clinical_harm = derive_clinical_harm_risk(crit, severity_label, life)

    aggressive = rl_for_ctx in {"THROTTLE", "BLOCK", "ISOLATE", "ALERT"}

    ctx: Dict[str, Any] = {
        "rl_action": rl_for_ctx,
        "trust_score": trust,
        "trust": trust,  # legacy alias
        "risk_score": risk,
        "contextual_risk": risk,
        "attack_confidence": conf,
        "attack_severity": severity_label,
        "attack_type": map_attack_type(raw.get("attack_type") or raw.get("attack_class")),
        "device_criticality": crit,
        "clinical_impact": crit if crit in CRITICALITY_RANK else "MEDIUM",
        "life_support_status": life,
        "life_support": int(life),  # legacy
        "regulated_medical_device": regulated,
        "patient_data_access": patient_data,
        "active_clinical_use": active_clinical,
        "patient_safety_risk": patient_safety,
        "clinical_harm_risk": clinical_harm,
        "cybersecurity_purpose_authorised": to_bool(
            raw.get("cybersecurity_purpose_authorised", True)
        ),
        "clinical_risk_assessed": to_bool(raw.get("clinical_risk_assessed", False)),
        "human_override_requested": to_bool(raw.get("human_override_requested", False)),
        "operator_authorised": to_bool(raw.get("operator_authorised", False)),
        "human_approval_status": str(
            raw.get(
                "human_approval_status",
                "PENDING"
                if to_bool(raw.get("escalation_required", False))
                or to_bool(raw.get("human_override_requested", False))
                else "NOT_REQUIRED",
            )
        ).upper(),
        "escalation_required": to_bool(raw.get("escalation_required", False)),
        "contains_personal_identifiers": to_bool(
            raw.get("contains_personal_identifiers", patient_data)
        ),
        "identifiers_required_for_response": to_bool(
            raw.get("identifiers_required_for_response", False)
        ),
        "security_alert_generated": to_bool(raw.get("security_alert_generated", aggressive)),
        "contains_patient_information": to_bool(
            raw.get("contains_patient_information", patient_data)
        ),
        "minimum_necessary_data": to_bool(raw.get("minimum_necessary_data", True)),
        "device_type": (profile or {}).get("device_type") or raw.get("device_type"),
        "device_id": (profile or {}).get("device_id") or raw.get("device_id"),
        "participant_id": raw.get("participant_id") or raw.get("agent_id") or "",
        "permitted_automatic_actions": (profile or {}).get("permitted_automatic_actions"),
    }

    # Explicit overrides win (tests / future human-in-the-loop)
    for k, v in raw.items():
        if k in ctx and k not in {
            "trust",
            "attack_severity",
            "device_criticality",
            "rl_action",
        }:
            # only override known clinical flags when explicitly provided
            if k.endswith("_authorised") or k.endswith("_requested") or k.endswith("_assessed"):
                ctx[k] = to_bool(v) if not isinstance(v, str) or v.upper() in {
                    "TRUE", "FALSE", "T", "F", "1", "0", "PENDING", "NOT_REQUIRED"
                } else v
    # Allow full override dict
    overrides = raw.get("context_overrides") or {}
    if isinstance(overrides, Mapping):
        ctx.update(dict(overrides))

    # Re-assert rl_action UPPER after overrides unless override set it
    if "rl_action" not in overrides:
        ctx["rl_action"] = rl_for_ctx
    else:
        ctx["rl_action"] = str(ctx["rl_action"]).strip().upper()

    return ctx
