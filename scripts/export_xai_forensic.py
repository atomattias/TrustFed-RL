#!/usr/bin/env python3
"""Export forensic XAI JSON + hospital_09 figure from latest IoMT audit run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai.explainer import (  # noqa: E402
    build_forensic_record,
    default_signal_weights,
    export_forensic_json,
    load_audit_events,
)
from xai.visualize import plot_hospital09_forensic  # noqa: E402


def latest_audit(audit_dir: Path) -> Path:
    files = sorted(audit_dir.glob("audit_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No audit_*.jsonl in {audit_dir}")
    return files[0]


def main() -> int:
    p = argparse.ArgumentParser(description="TrustFed-RL XAI forensic export")
    p.add_argument("--agent-id", default="hospital_09_compromised_iomt")
    p.add_argument("--peer-id", default="hospital_01_high_quality_iomt")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--run-id", default=None, help="experiment_run_id filter for DB lookup")
    p.add_argument(
        "--no-condition-log",
        action="store_true",
        help="Omit condition_evaluation_log from forensic JSON",
    )
    p.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Audit JSONL (default: latest under results/trustfed_agent/audit/iomt)",
    )
    p.add_argument(
        "--policy-db",
        type=Path,
        default=ROOT / "results/trustfed_agent/governance/policy_repo_iomt.sqlite",
    )
    p.add_argument(
        "--trust-config",
        type=Path,
        default=ROOT / "config/trust_config.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/trustfed_agent/xai",
    )
    args = p.parse_args()

    audit = args.audit or latest_audit(ROOT / "results/trustfed_agent/audit/iomt")
    weights = default_signal_weights(args.trust_config)
    events = load_audit_events(audit)
    record = build_forensic_record(
        agent_id=args.agent_id,
        round_num=args.round,
        events=events,
        weights=weights,
        policy_db=args.policy_db if args.policy_db.exists() else None,
        audit_path=audit,
        include_condition_log=not args.no_condition_log,
        experiment_run_id=args.run_id,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_forensic_json(
        record, args.out_dir / f"forensic_{args.agent_id}_r{args.round}.json"
    )
    fig_path = plot_hospital09_forensic(
        audit,
        agent_id=args.agent_id,
        peer_id=args.peer_id,
        round_num=args.round,
        weights=weights,
        out_path=args.out_dir / "hospital_09_forensic.png",
        policy_db=args.policy_db if args.policy_db.exists() else None,
        experiment_run_id=args.run_id,
        decision_id=(record.get("governance") or {}).get("decision_id"),
    )
    print(
        json.dumps(
            {
                "forensic_json": str(json_path),
                "figure": str(fig_path),
                "narrative": record["narrative"],
                "schema": record.get("schema"),
                "n_condition_logs": len(
                    (record.get("governance") or {}).get("condition_evaluation_log") or []
                ),
                "round_decision_ids": record.get("round_decision_ids"),
                "governance_db_summary": record.get("governance_db_summary"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
