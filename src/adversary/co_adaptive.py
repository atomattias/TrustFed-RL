"""
Co-adaptive adversary for TrustFed-RL (master threat model).

Algorithm CoAdaptiveAttack (per federated round t):
  Input: compromised clients A, defender telemetry s_t, start round t0
  Output: poisoned updates / inflated trust signals / forged detections

  1. If t < t0: return (no attack)
  2. Choose mode m_t ∈ {
       poisoning,           # corrupt local updates / labels
       on_off,              # alternate silent ↔ attack participation
       trust_inflation,     # inflate V,S,U,C,R evidence
       rl_exploitation,     # flood high-confidence false positives
       governance_gaming    # probe near policy thresholds
     } from schedule(s_t)
  3. For each i ∈ A:
       a. If m_t = on_off and silent phase: mark missed round (C↓); skip upload
       b. Else allow local train
       c. If m_t ∈ {poisoning, on_off(attack)}: poison model update Δ_i
       d. If m_t ∈ {trust_inflation, governance_gaming, on_off(attack)}:
            inflate multi-signal trust evidence
       e. If m_t ∈ {rl_exploitation, governance_gaming}:
            inject forged attack detections on benign traffic
  4. Return modified artefacts to the experiment loop

This formalizes the master threats: network/insider compromise, model poisoning,
adaptive on–off behaviour, and unsafe autonomous-response probing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agents.autonomous_agent import DetectionResult


ATTACK_MODES = (
    "poisoning",
    "on_off",
    "trust_inflation",
    "rl_exploitation",
    "governance_gaming",
)


class CoAdaptiveAdversary:
    """Adaptive adversary covering the master IoMT FL threat model."""

    def __init__(
        self,
        start_round: int = 4,
        rng: Optional[np.random.Generator] = None,
        target_client_ids: Optional[list] = None,
        flood_fraction: float = 0.25,
        poison_scale: float = 3.0,
        on_off_period: int = 2,
    ):
        self.start_round = int(start_round)
        self.rng = rng or np.random.default_rng(42)
        self.target_client_ids: List[str] = list(target_client_ids or [])
        self.flood_fraction = float(flood_fraction)
        self.poison_scale = float(poison_scale)
        self.on_off_period = max(2, int(on_off_period))
        self.active = False
        self.mode = "trust_inflation"
        self._round_num = 0
        self._history: List[Dict[str, Any]] = []

    def is_target(self, client_id: str) -> bool:
        return client_id in self.target_client_ids

    def on_round_start(self, round_num: int, defender_state: Optional[Dict[str, Any]] = None) -> None:
        """Select attack mode from defender telemetry (co-adaptation)."""
        self._round_num = int(round_num)
        self.active = self._round_num >= self.start_round
        if not self.active:
            self.mode = "inactive"
            return

        state = defender_state or {}
        viol_rate = float(state.get("violation_rate", 0.0) or 0.0)
        resp_prec = float(state.get("response_precision", 1.0) or 1.0)
        # Phase schedule mixes threat-model attacks with co-adaptation to defender state
        phase = (self._round_num - self.start_round) % 5
        if phase == 0:
            self.mode = "poisoning"
        elif phase == 1:
            self.mode = "on_off"
        elif viol_rate < 0.05 and resp_prec > 0.5:
            self.mode = "rl_exploitation"
        elif viol_rate < 0.15:
            self.mode = "governance_gaming"
        else:
            self.mode = "trust_inflation"

        self._history.append({"round": self._round_num, "mode": self.mode, "state": dict(state)})

    def on_off_silent(self, client_id: str) -> bool:
        """True when on–off adversary withholds participation this round."""
        if not self.active or self.mode != "on_off" or not self.is_target(client_id):
            return False
        # Silent on odd attack-phase ticks
        return ((self._round_num - self.start_round) // 1) % self.on_off_period == 1

    def should_skip_participation(self, client_id: str) -> bool:
        return self.on_off_silent(client_id)

    def perturb_trust_signals(self, client_id: str, signals: Dict[str, float]) -> Dict[str, float]:
        if not self.active or not self.is_target(client_id):
            return signals
        if self.mode not in ("trust_inflation", "governance_gaming", "on_off", "poisoning"):
            return signals
        # During silent on–off, do not inflate (absence is the signal)
        if self.mode == "on_off" and self.on_off_silent(client_id):
            return signals

        out = dict(signals)
        bump_v, bump_s = 0.15, 0.2
        if self.mode == "poisoning":
            bump_v, bump_s = 0.08, 0.1  # milder camouflage while poisoning
        for key, bump in (
            ("V", bump_v),
            ("accuracy", bump_v),
            ("S", bump_s),
            ("stability", bump_s),
            ("C", bump_s),
            ("communication", bump_s),
            ("R", bump_s),
            ("contextual_safety", bump_s),
        ):
            if key in out:
                out[key] = min(1.0, float(out[key]) + bump)
        out["drift"] = max(0.0, float(out.get("drift", out.get("D", 0.5))) - 0.1)
        out["D"] = out["drift"]
        out["uncertainty"] = min(1.0, float(out.get("uncertainty", out.get("U", 0.5))) + 0.1)
        out["U"] = out["uncertainty"]
        if "contextual_risk" in out:
            out["contextual_risk"] = max(0.0, float(out["contextual_risk"]) - 0.2)
            out["R"] = 1.0 - float(out["contextual_risk"])
            out["contextual_safety"] = out["R"]
        return out

    def poison_model_update(self, client_id: str, update: Dict[str, Any]) -> Dict[str, Any]:
        """Poison local model parameters and/or training labels (model/data poisoning)."""
        if not self.active or not self.is_target(client_id):
            return update
        if self.mode not in ("poisoning", "on_off"):
            return update
        if self.mode == "on_off" and self.on_off_silent(client_id):
            return update

        out = dict(update)
        params = out.get("parameters")
        if isinstance(params, dict):
            poisoned = {}
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    noise = self.rng.normal(0.0, self.poison_scale, size=v.shape)
                    poisoned[k] = v + noise
                elif isinstance(v, (int, float)):
                    poisoned[k] = float(v) + float(self.rng.normal(0.0, self.poison_scale))
                else:
                    poisoned[k] = v
            out["parameters"] = poisoned

        # Label flipping on uploaded local data (retraining aggregators)
        y = out.get("y_train")
        if y is not None:
            try:
                import pandas as pd

                y_arr = np.asarray(y).copy()
                n = len(y_arr)
                if n > 0:
                    n_flip = max(1, int(0.3 * n))
                    idx = self.rng.choice(n, size=min(n_flip, n), replace=False)
                    y_arr[idx] = 1 - y_arr[idx]
                    # Keep Series so FedAvg/trust aggregators can pd.concat labels
                    if isinstance(y, pd.Series):
                        out["y_train"] = pd.Series(y_arr, index=y.index)
                    else:
                        out["y_train"] = pd.Series(y_arr)
            except Exception:
                pass

        out["poisoned"] = True
        out["poison_mode"] = self.mode
        return out

    def should_flood_benign(self) -> bool:
        return self.active and self.mode in ("rl_exploitation", "governance_gaming")

    def craft_benign_confidence(self) -> float:
        if self.mode == "governance_gaming":
            # Sit near block threshold to probe governance (master unsafe-action threat)
            return float(self.rng.uniform(0.76, 0.84))
        return float(self.rng.uniform(0.80, 0.92))

    def inject_rl_exploitation(
        self,
        detections: List[DetectionResult],
    ) -> List[DetectionResult]:
        """Forge high-confidence attack predictions on benign traffic (RL / gov probing)."""
        if not self.should_flood_benign():
            return detections

        benign_idx = [
            i for i, d in enumerate(detections)
            if d.ground_truth == 0 and d.agent_id in self.target_client_ids
        ]
        if not benign_idx:
            benign_idx = [i for i, d in enumerate(detections) if d.ground_truth == 0]
        if not benign_idx:
            return detections

        n_flood = max(1, int(len(benign_idx) * self.flood_fraction))
        chosen = self.rng.choice(benign_idx, size=min(n_flood, len(benign_idx)), replace=False)
        out = list(detections)
        for i in chosen:
            d = out[i]
            conf = self.craft_benign_confidence()
            # Prefer life-critical device types when gaming governance
            device = d.device_type
            if self.mode == "governance_gaming":
                device = str(self.rng.choice(
                    ["infusion_pump", "vitals_monitor", "ecg", "patient_monitor"]
                ))
            out[i] = DetectionResult(
                predicted_label=1,
                confidence=conf,
                attack_class="attack",
                agent_id=d.agent_id,
                round_num=d.round_num,
                ground_truth=d.ground_truth,
                device_type=device,
                attack_severity=conf,
                device_criticality=1.0 if device in {
                    "infusion_pump", "vitals_monitor", "ecg", "patient_monitor"
                } else getattr(d, "device_criticality", 0.5),
                contextual_risk=getattr(d, "contextual_risk", 0.0),
            )
        return out

    def summarize(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "mode": self.mode,
            "round": self._round_num,
            "targets": list(self.target_client_ids),
            "history": list(self._history[-20:]),
        }
