"""XAI figures for TrustFed-RL (master Contrib. 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from xai.explainer import SIGNAL_ORDER, explain_trust, filter_agent_events, load_audit_events


def plot_hospital09_forensic(
    audit_path: Path,
    *,
    agent_id: str = "hospital_09_compromised_iomt",
    peer_id: str = "hospital_01_high_quality_iomt",
    round_num: int = 1,
    weights: Optional[Dict[str, float]] = None,
    out_path: Optional[Path] = None,
    policy_db: Optional[Path] = None,
    experiment_run_id: Optional[str] = None,
    decision_id: Optional[int] = None,
) -> Path:
    """Two-panel figure: signal contributions vs peer + trust trajectory.

    Optional policy_db annotates the title with governance outcome/final_action.
    """
    from xai.explainer import fetch_governance_decision, find_decision_ids

    events = load_audit_events(Path(audit_path))
    if not weights:
        raise ValueError("weights required (pass trust_config multi_signal.signal_weights)")

    def trust_at(aid: str, r: int) -> Dict[str, Any]:
        scoped = filter_agent_events(events, aid, round_num=r)
        te = next((e for e in scoped if e.get("event_type") == "trust_update"), None)
        if not te:
            return explain_trust({}, weights)
        return explain_trust(
            te.get("signals") or {},
            weights,
            fused_trust=te.get("trust"),
            rl_delta=float(te.get("rl_delta") or 0.0),
        )

    target = trust_at(agent_id, round_num)
    peer = trust_at(peer_id, round_num)

    gov_note = ""
    if policy_db and Path(policy_db).exists():
        did = decision_id
        if did is None:
            # Prefer decision_id from audit events for this round
            scoped = filter_agent_events(events, agent_id, round_num=round_num)
            for e in scoped:
                if e.get("decision_id") is not None:
                    did = int(e["decision_id"])
                    break
        if did is None:
            ids = find_decision_ids(
                Path(policy_db),
                participant_id=agent_id,
                experiment_round=round_num,
                experiment_run_id=experiment_run_id,
                limit=100,
            )
            did = ids[-1] if ids else None
        if did is not None:
            gov = fetch_governance_decision(
                Path(policy_db), did, include_condition_log=False
            )
            if gov:
                gov_note = (
                    f" | gov {gov.get('governance_outcome')}→{gov.get('final_action')}"
                )

    # Trust over rounds for target
    rounds: List[int] = []
    trusts: List[float] = []
    for e in filter_agent_events(events, agent_id):
        if e.get("event_type") != "trust_update":
            continue
        if e.get("round") is None:
            continue
        rounds.append(int(e["round"]))
        trusts.append(float(e.get("trust") or 0.0))
    # de-duplicate by round (keep last)
    by_r = {}
    for r, t in zip(rounds, trusts):
        by_r[r] = t
    r_sorted = sorted(by_r)
    t_sorted = [by_r[r] for r in r_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(SIGNAL_ORDER))
    width = 0.36
    t_contrib = [target["contributions"].get(k, 0.0) for k in SIGNAL_ORDER]
    p_contrib = [peer["contributions"].get(k, 0.0) for k in SIGNAL_ORDER]
    axes[0].bar(x - width / 2, t_contrib, width, label=agent_id.replace("_iomt", ""), color="#c0392b")
    axes[0].bar(x + width / 2, p_contrib, width, label=peer_id.replace("_iomt", ""), color="#2980b9")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(SIGNAL_ORDER))
    axes[0].set_ylabel("Contribution w·x")
    axes[0].set_title(f"Trust signal contributions (round {round_num})")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_ylim(0, max(max(t_contrib + p_contrib + [0.01]) * 1.25, 0.05))

    axes[1].plot(r_sorted, t_sorted, marker="o", color="#c0392b", label=agent_id.replace("_iomt", ""))
    axes[1].axhline(target.get("reported_T", 0), color="#c0392b", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Fused trust T")
    axes[1].set_title("Trust trajectory (compromised client)")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.suptitle(
        f"XAI forensic view — {agent_id}\n"
        f"T≈{target.get('reported_T', 0):.3f}; weakest={target.get('weakest_signal')}"
        f"{gov_note}",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(out_path) if out_path else Path("results/trustfed_agent/xai/hospital_09_forensic.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_signal_contributions(
    trust_explain: Dict[str, Any],
    *,
    title: str = "Trust signal contributions",
    out_path: Optional[Path] = None,
) -> Path:
    contrib = [trust_explain.get("contributions", {}).get(k, 0.0) for k in SIGNAL_ORDER]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(list(SIGNAL_ORDER), contrib, color="#34495e")
    ax.set_ylabel("w·x")
    ax.set_title(title)
    ax.set_ylim(0, max(contrib + [0.01]) * 1.2)
    out = Path(out_path) if out_path else Path("results/trustfed_agent/xai/signal_contributions.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
