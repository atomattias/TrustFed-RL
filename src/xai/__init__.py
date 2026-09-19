"""TrustFed-RL XAI package (master Contribution 5)."""

from xai.explainer import (
    build_forensic_record,
    default_signal_weights,
    explain_trust,
    export_forensic_json,
    fetch_governance_decision,
    find_decision_ids,
    load_audit_events,
    summarize_governance_db,
)
from xai.visualize import plot_hospital09_forensic

__all__ = [
    "build_forensic_record",
    "default_signal_weights",
    "explain_trust",
    "export_forensic_json",
    "fetch_governance_decision",
    "find_decision_ids",
    "load_audit_events",
    "plot_hospital09_forensic",
    "summarize_governance_db",
]
