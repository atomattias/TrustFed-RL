"""Tamper-evident audit logging for governance compliance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ComplianceMonitor:
    def __init__(self, audit_log_dir: str, run_id: Optional[str] = None):
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = self.audit_log_dir / f"audit_{self.run_id}.jsonl"
        self._events: List[Dict[str, Any]] = []

    def log_event(self, event: Dict[str, Any]) -> None:
        event = {
            **event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
        }
        self._events.append(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_response_history(self, agent_id: str, window: int = 10) -> List[Dict[str, Any]]:
        events = [
            e for e in self._events
            if e.get("agent_id") == agent_id
            and e.get("event_type") in ("response_executed", "response_proposed", "policy_violation")
        ]
        return events[-window:]

    @property
    def audit_completeness(self) -> float:
        rounds = {e.get("round") for e in self._events if e.get("round") is not None}
        if not rounds:
            return 0.0
        complete = 0
        for r in rounds:
            types = {e.get("event_type") for e in self._events if e.get("round") == r}
            if "trust_update" in types and ("response_executed" in types or "response_proposed" in types):
                complete += 1
        return complete / len(rounds)
