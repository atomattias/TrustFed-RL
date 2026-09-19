"""
Full TrustFed-Agent experiment: federated learning + governance + response loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from agents.autonomous_agent import DetectionResult

import numpy as np
import pandas as pd

from agents.autonomous_agent import AutonomousAgent
from adversary.co_adaptive import CoAdaptiveAdversary
from adversary.late_compromise import LateCompromiseConfig, LateCompromiseController, fraction_tag
from config_loader import (
    apply_trust_config,
    load_agent_config,
    load_governance_config,
    load_rl_config,
    load_trust_config,
    resolve_dataset_configs,
    resolve_performance_signal,
    resolve_signal_weights,
    resolve_trust_config_path,
)
from evaluation import (
    estimate_update_payload_bytes,
    evaluate_model_on_test,
    resolve_metrics_run_id,
    save_trustfed_agent_metrics,
    summarize_resource_metrics,
    summarize_robustness_metrics,
    summarize_trust_metrics,
    compute_natural_trust_discrimination,
    compute_trust_recovery,
)
from privacy.quantize import unpack_lr_parameters
from privacy.weighted_secagg import (
    WeightedSecAggRound,
    assert_no_plaintext,
    install_logistic_from_avg,
    privacy_update_from_masked,
    weighted_payload,
)
from federated_client import FederatedClient
from federated_server import TrustAwareAggregator, TrustManager
from governance.compliance_monitor import ComplianceMonitor
from governance.policy_engine import PolicyEngine
from governance.policy_repository import PolicyRepository
from governance.security_validator import SecurityValidator, default_policy_db_path
from preprocessing import load_client_data, prepare_labels, prepare_features, split_data
from rl.environment import ResponseSimulator
from rl.observation import ACTION_TYPES
from rl.response_strategy import ResponseStrategy
from rl.trust_calibrator import TrustCalibrator


@dataclass
class AgentExperimentConfig:
    data_dir: str
    approach: str = "trustfed_agent"
    num_rounds: int = 30
    random_state: int = 42
    model_type: str = "logistic_regression"
    enable_lambda5: bool = True
    disable_cr_signals: bool = False
    enable_governance: bool = True
    enable_rl: bool = True
    adversary_mode: str = "static"
    dataset: str = "iomt"
    test_csv: Optional[str] = None
    val_ref_csv: Optional[str] = None
    trust_config_path: Optional[str] = None
    governance_config_path: Optional[str] = None
    rl_config_path: Optional[str] = None
    agent_config_path: Optional[str] = None
    max_response_incidents_per_round: int = 50
    enable_weighted_secagg: bool = False
    share_floor: float = 0.0
    uniform_contextual_prior: bool = False
    include_R_in_T: Optional[bool] = None  # None → trust_config; False for natural default
    late_compromise: Optional[LateCompromiseConfig] = None
    late_compromise_config_path: Optional[str] = None
    metrics_suffix: Optional[str] = None


class AgentExperimentRunner:
    def __init__(self, cfg: AgentExperimentConfig, root: Path):
        self.cfg = cfg
        self.root = root
        gov_path, rl_path = resolve_dataset_configs(
            cfg.dataset,
            cfg.governance_config_path,
            cfg.rl_config_path,
            root,
        )
        trust_path = resolve_trust_config_path(cfg.dataset, cfg.trust_config_path, root)
        self.trust_config = load_trust_config(trust_path)
        self.gov_config = load_governance_config(gov_path)
        self.rl_config = load_rl_config(rl_path)
        self.agent_config = load_agent_config(cfg.agent_config_path)
        self.perf_signal = resolve_performance_signal(cfg.data_dir, self.trust_config)

        disable = ["C", "R"] if cfg.disable_cr_signals else None
        include_r = cfg.include_R_in_T
        if include_r is None and cfg.dataset == "iomt_natural":
            include_r = False
        signal_weights = resolve_signal_weights(
            self.trust_config,
            disable_signals=disable,
            include_R_in_T=include_r,
        )
        self.include_R_in_T = bool(signal_weights.get("R", 0.0) > 1e-12)
        # Closed-loop response fidelity maps into contextual risk R (RL/gov state).
        # When R∉T, fidelity still affects RL observation via contextual_risk, not aggregation trust.
        self.include_response_in_R = bool(
            self.trust_config.get("multi_signal", {}).get("include_response_fidelity_in_R", True)
        ) and bool(cfg.enable_lambda5)

        self.trust_manager = TrustManager(
            use_multi_signal=True,
            signal_weights=signal_weights,
        )
        apply_trust_config(self.trust_manager, self.trust_config)

        self.late_compromise: Optional[LateCompromiseController] = None
        self._late_cfg = self._load_late_compromise_config()

        self.policy_engine = PolicyEngine(self.gov_config)
        if cfg.share_floor:
            self.policy_engine.share_floor = float(cfg.share_floor)
        audit_dir = self.gov_config.get("compliance", {}).get(
            "audit_log_dir", "results/trustfed_agent/audit"
        )
        audit_path = Path(audit_dir)
        if not audit_path.is_absolute():
            audit_path = root / audit_path
        self.compliance = ComplianceMonitor(str(audit_path))
        repo_cfg = self.gov_config.get("policy_repository", {})
        db_path = repo_cfg.get("db_path")
        if db_path:
            db_path = Path(db_path)
            if not db_path.is_absolute():
                db_path = root / db_path
        else:
            db_path = default_policy_db_path(root, cfg.dataset)
        seed_path = repo_cfg.get("seed_path")
        if seed_path:
            seed_path = Path(seed_path)
            if not seed_path.is_absolute():
                seed_path = root / seed_path
        aliases_path = repo_cfg.get("device_aliases_path")
        if aliases_path:
            aliases_path = Path(aliases_path)
            if not aliases_path.is_absolute():
                aliases_path = root / aliases_path
        roster_path = repo_cfg.get("device_roster_path")
        if roster_path:
            roster_path = Path(roster_path)
            if not roster_path.is_absolute():
                roster_path = root / roster_path
        self.policy_repository = PolicyRepository(
            db_path,
            seed_path=seed_path,
            device_aliases_path=aliases_path,
            device_roster_path=roster_path,
            experiment_run_id=f"{cfg.dataset}_{cfg.approach}_seed_{cfg.random_state}",
            experiment_seed=int(cfg.random_state),
        )
        repo_for_validator = (
            self.policy_repository if self.policy_repository.runtime_api_ready else None
        )
        self.validator = SecurityValidator(
            self.gov_config, self.policy_engine, policy_repository=repo_for_validator
        )
        self.simulator = ResponseSimulator(
            self.rl_config.get("reward_weights", {}),
            self.rl_config.get("action_costs", {}),
        )
        self.trust_calibrator = TrustCalibrator(self.rl_config, self.gov_config)
        rl_ckpt = self.rl_config.get("training", {}).get(
            "checkpoint_dir", "results/trustfed_agent/rl_checkpoints"
        )
        self.response_strategy = ResponseStrategy(
            self.rl_config,
            use_rl=cfg.enable_rl,
            checkpoint_dir=str(root / rl_ckpt),
            max_rounds=cfg.num_rounds,
        )
        self._incident_buffer: List[Tuple[DetectionResult, float]] = []
        self._gov_intervention_count = 0
        self._gov_decision_count = 0
        # β from governance: paper α_i ∝ T_i ⇒ trust_exponent_beta=1 (iomt_natural).
        _agg_beta = float(
            (self.gov_config.get("aggregation") or {}).get("trust_exponent_beta", 1.0)
        )
        self.aggregator = TrustAwareAggregator(
            model_type=cfg.model_type, trust_manager=self.trust_manager, beta=_agg_beta
        )
        self.adversary: Optional[CoAdaptiveAdversary] = None
        if cfg.adversary_mode == "co_adaptive":
            self.adversary = CoAdaptiveAdversary(
                rng=np.random.default_rng(cfg.random_state),
                target_client_ids=[],
            )

        self.clients: List[FederatedClient] = []
        self.agents: List[AutonomousAgent] = []
        self.global_model = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

    def setup_clients(self) -> None:
        data_dir = Path(self.cfg.data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {data_dir}")

        val_ref_path = self._resolve_val_ref_csv_path(data_dir)
        use_shared = val_ref_path is not None and (
            self.cfg.dataset == "iomt_natural" or "iomt_natural" in str(data_dir)
        )
        if self.cfg.dataset == "iomt_natural" and val_ref_path is None:
            raise FileNotFoundError(
                f"iomt_natural requires iomt_val_ref.csv next to clients dir ({data_dir.parent})"
            )

        val_checksums = []
        for i, csv in enumerate(csv_files):
            client = FederatedClient(
                client_id=csv.stem,
                data_path=str(csv),
                model_type=self.cfg.model_type,
                random_state=self.cfg.random_state + i,
                client_quality="unknown" if use_shared else "unknown",
                clean_validation_source=str(val_ref_path) if use_shared else None,
                use_shared_val_ref=use_shared,
                performance_signal=self.perf_signal,
                uniform_contextual_prior=self.cfg.uniform_contextual_prior or use_shared,
            )
            client.load_data()
            if use_shared and client.y_val is not None:
                val_checksums.append(
                    (
                        len(client.y_val),
                        float(client.y_val.mean()),
                        tuple(client.X_val.columns.tolist()),
                    )
                )
            self.trust_manager.initialize_client(client.client_id)
            self.clients.append(client)
            self.agents.append(
                AutonomousAgent(client, agent_config=self.agent_config)
            )
            if self.adversary and "compromised" in csv.stem.lower():
                self.adversary.target_client_ids.append(client.client_id)

        if use_shared and val_checksums:
            if len(set(val_checksums)) != 1:
                raise RuntimeError(
                    f"Shared val_ref mismatch across clients: {set(val_checksums)}"
                )
            print(
                f"  ✅ All {len(self.clients)} clients share identical full-file "
                f"D_val (n={val_checksums[0][0]}, attack_rate={val_checksums[0][1]:.4f})"
            )

        if self._late_cfg is not None and self._late_cfg.enabled:
            self.late_compromise = LateCompromiseController(
                self._late_cfg,
                [c.client_id for c in self.clients],
            )
            print(f"  Late compromise: {self.late_compromise.summarize()}")

        # Hold-out test set (IoMT: iomt_test_set.csv)
        test_path = self._resolve_test_csv_path(data_dir)
        df = load_client_data(str(test_path))
        df = prepare_labels(df)
        X = prepare_features(df)
        y = df["label"]
        if test_path.name in (
            "heterogeneous_test_set.csv",
            "ctu13_test_set.csv",
            "iomt_test_set.csv",
            "wustl_ehms_test_set.csv",
        ) or self.cfg.test_csv or self.cfg.dataset == "iomt_natural":
            self.X_test = X
            self.y_test = y
        else:
            _, X_test, _, y_test = split_data(
                X, y, test_size=0.1, random_state=self.cfg.random_state
            )
            self.X_test = X_test
            self.y_test = y_test

    def _resolve_val_ref_csv_path(self, data_dir: Path) -> Optional[Path]:
        if self.cfg.val_ref_csv:
            p = Path(self.cfg.val_ref_csv)
            if p.exists():
                return p
        if self.cfg.dataset == "iomt_natural" or "iomt_natural" in str(data_dir):
            candidates = [
                data_dir.parent / "iomt_val_ref.csv",
                self.root / "data" / "CSVs" / "iomt_natural" / "iomt_val_ref.csv",
            ]
            for path in candidates:
                if path.exists():
                    return path
        return None

    def _resolve_test_csv_path(self, data_dir: Path) -> Path:
        if self.cfg.test_csv:
            p = Path(self.cfg.test_csv)
            if p.exists():
                return p

        if self.cfg.dataset == "ctu13":
            candidates = [
                data_dir.parent / "ctu13_test_set.csv",
                self.root.parent / "data" / "CSVs" / "ctu13_test_set.csv",
            ]
        elif self.cfg.dataset == "iomt_natural":
            candidates = [
                data_dir.parent / "iomt_test_set.csv",
                self.root / "data" / "CSVs" / "iomt_natural" / "iomt_test_set.csv",
            ]
        elif self.cfg.dataset == "iomt":
            candidates = [
                data_dir.parent / "iomt_test_set.csv",
                self.root / "data" / "CSVs" / "iomt_test_set.csv",
                self.root.parent / "data" / "CSVs" / "iomt_test_set.csv",
            ]
        elif self.cfg.dataset == "wustl_ehms":
            candidates = [
                data_dir.parent / "wustl_ehms_test_set.csv",
                self.root / "data" / "CSVs" / "wustl_ehms_test_set.csv",
                self.root.parent / "data" / "CSVs" / "wustl_ehms_test_set.csv",
            ]
        else:
            candidates = [
                data_dir.parent / "heterogeneous_test_set.csv",
                self.root.parent / "data" / "CSVs" / "heterogeneous_test_set.csv",
                self.root / "data" / "CSVs" / "heterogeneous_test_set.csv",
            ]
        for path in candidates:
            if path.exists():
                return path
        csv_files = sorted(data_dir.glob("*.csv"))
        return csv_files[0]

    def _load_late_compromise_config(self) -> Optional[LateCompromiseConfig]:
        if self.cfg.late_compromise is not None:
            cfg = self.cfg.late_compromise
            cfg.seed = int(self.cfg.random_state)
            return cfg
        # Explicit path or iomt_natural default
        path = self.cfg.late_compromise_config_path
        if path is None and self.cfg.dataset == "iomt_natural":
            cand = self.root / "config" / "iomt_natural_adversary.json"
            path = str(cand) if cand.exists() else None
        if path is None:
            return None
        import json
        raw = json.loads(Path(path).read_text())
        cfg = LateCompromiseConfig.from_dict(raw)
        cfg.seed = int(self.cfg.random_state)
        return cfg

    def _apply_governed_trust_for_aggregation(self, updates: List[Dict]) -> None:
        """Apply governance aggregation constraints without corrupting trust histories.

        Writes normalized `aggregation_weight` onto each update for the aggregator.
        Trust scores in TrustManager remain behavioural trust in [0, 1].
        """
        if not self.cfg.enable_governance:
            ts = np.array(
                [float(u.get("trust", self.trust_manager.get_trust(u["client_id"])))
                 for u in updates],
                dtype=float,
            )
            total = float(ts.sum())
            weights = (
                np.ones(len(ts)) / max(len(ts), 1)
                if total <= 0
                else ts / total
            )
            for i, u in enumerate(updates):
                u["aggregation_weight"] = float(weights[i])
            return
        ids = [u["client_id"] for u in updates]
        trusts = np.array([self.trust_manager.get_trust(cid) for cid in ids], dtype=float)
        floor = float(self.policy_engine.min_trust_threshold)
        trusts = np.where(trusts >= floor, trusts, 0.0)
        weights = self.policy_engine.constrain_aggregation_weights(trusts, ids)
        for i, u in enumerate(updates):
            u["trust"] = float(trusts[i])
            u["aggregation_weight"] = float(weights[i])

    def _aggregate_weighted_secagg(
        self, client_updates: List[Dict[str, Any]], round_num: int
    ) -> Any:
        """Mask v_i=α_i θ_i, sum, install LR. Mutates updates into privacy payloads."""
        if not client_updates:
            raise ValueError("No client updates for weighted SecAgg")
        ids = [str(u["client_id"]) for u in client_updates]
        packed_vs = []
        n_features = None
        classes = None
        for u in client_updates:
            params = u.get("parameters") or {}
            alpha = float(u.get("aggregation_weight", u.get("trust", 0.0)))
            vec, dim = weighted_payload(params, alpha)
            if n_features is None:
                n_features = int(np.asarray(params["coef"]).shape[-1])
                classes = np.asarray(params.get("classes", [0, 1]))
            packed_vs.append(vec)
        dim = int(packed_vs[0].size)
        rnd = WeightedSecAggRound(ids, dim)
        rnd.setup_keys()
        masked_map = {}
        share_b = rnd.share_blob_bytes()
        for u, vec in zip(client_updates, packed_vs):
            cid = str(u["client_id"])
            mv = rnd.mask_vector(cid, vec)
            masked_map[cid] = mv
        qsum = rnd.unmask_sum(masked_map)
        rec = rnd.dequantize_sum(qsum)
        coef, intercept = unpack_lr_parameters(rec, int(n_features))
        model = install_logistic_from_avg(coef, intercept, classes)
        self.aggregator.global_model = model
        self.aggregator.training_feature_columns = getattr(
            self, "X_test", None
        )
        if self.X_test is not None:
            cols = list(self.X_test.columns)
            self.aggregator.training_feature_columns = cols
            model.feature_names_in_ = np.array(cols, dtype=object)

        privacy_updates = []
        for u in client_updates:
            cid = str(u["client_id"])
            alpha = float(u.get("aggregation_weight", 0.0))
            pu = privacy_update_from_masked(
                client_id=cid,
                trust=float(u.get("trust", 0.0)),
                aggregation_weight=alpha,
                masked=masked_map[cid],
                share_blob_bytes=share_b,
                n_features=int(n_features),
                classes=classes,
                collaboration_share=alpha > 0,
                round_num=round_num,
            )
            assert_no_plaintext(pu)
            privacy_updates.append(pu)
        client_updates[:] = privacy_updates
        return model

    def run_round(self, round_num: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if self.adversary:
            self.adversary.on_round_start(round_num, self.simulator.summarize())

        poison_report: Dict[str, Any] = {}
        if self.late_compromise is not None:
            poison_report = self.late_compromise.sync_round(round_num, self.clients)
            if (
                poison_report.get("attack_active")
                and round_num == self.late_compromise.t_a
            ):
                print(
                    f"[LateCompromise] t_a={self.late_compromise.t_a} "
                    f"mode={self.late_compromise.cfg.poison_mode}: {poison_report.get('flipped')}"
                )
            self.compliance.log_event({
                "event_type": "late_compromise_sync",
                "round": round_num,
                "active": self.late_compromise.is_compromised_round(round_num),
                "attack_active": poison_report.get("attack_active"),
                "recovery": poison_report.get("recovery"),
                "phase": poison_report.get("phase"),
                "flipped": poison_report.get("flipped"),
                "schedule": self.late_compromise.summarize(),
            })

        client_updates = []
        client_round_rewards: Dict[str, List[float]] = {}
        client_contextual_risk: Dict[str, float] = {}
        for client, agent in zip(self.clients, self.agents):
            # On–off / late-compromise: compromised client withholds participation (C↓)
            skip = False
            if self.adversary and self.adversary.should_skip_participation(client.client_id):
                skip = True
            if (
                self.late_compromise
                and self.late_compromise.should_skip_participation(
                    client.client_id, round_num
                )
            ):
                skip = True
            if skip:
                client.record_communication_round(participated=False, late=False, incomplete=False)
                # Decay trust via missed communication evidence only
                signals = client.compute_multi_signal_trust_signals(
                    operational_penalty=0.0,
                    include_response_fidelity_in_R=False,
                )
                # Option 3: same Δτ channel as participate; participated=False so
                # skippers cannot evade a saturated peer offset (ranking inversion).
                if self.cfg.enable_rl:
                    signals = self.trust_calibrator.apply_to_signals(
                        client.client_id, signals, participated=False
                    )
                client_contextual_risk[client.client_id] = float(signals.get("contextual_risk", 0.0))
                self.trust_manager.update_trust(
                    client.client_id,
                    round_num,
                    computed_trust=signals.get("V", 0.5),
                    multi_signal_signals=signals,
                )
                client.trust_score = self.trust_manager.get_trust(client.client_id)
                self.compliance.log_event({
                    "event_type": "client_skip",
                    "round": round_num,
                    "agent_id": client.client_id,
                    "skip_reason": (
                        self.late_compromise.participation_skip_reason(
                            client.client_id, round_num
                        )
                        if self.late_compromise
                        else None
                    ),
                    "adversary_mode": (
                        self.late_compromise.cfg.poison_mode
                        if self.late_compromise
                        else getattr(self.adversary, "mode", None)
                    ),
                    "trust": client.trust_score,
                    "signals": {"C": signals.get("C"), "V": signals.get("V")},
                    "rl_delta": signals.get("rl_trust_calibration_delta", 0.0),
                })
                client_round_rewards[client.client_id] = []
                continue

            client.train()
            client.record_communication_round(participated=True, late=False, incomplete=False)
            operational_penalty = (
                agent.get_response_fidelity_penalty() if self.include_response_in_R else 0.0
            )
            signals = client.compute_multi_signal_trust_signals(
                operational_penalty=operational_penalty,
                include_response_fidelity_in_R=self.include_response_in_R,
            )
            # Ablation B5−S: force neutral C/R evidence (weights already zeroed).
            if self.cfg.disable_cr_signals:
                signals["C"] = signals["communication"] = 1.0
                signals["R"] = signals["contextual_safety"] = 1.0
                signals["contextual_risk"] = 0.0
            if self.adversary:
                signals = self.adversary.perturb_trust_signals(client.client_id, signals)
            if self.cfg.enable_rl:
                signals = self.trust_calibrator.apply_to_signals(
                    client.client_id, signals, participated=True
                )

            client_contextual_risk[client.client_id] = float(signals.get("contextual_risk", 0.0))
            self.trust_manager.update_trust(
                client.client_id,
                round_num,
                computed_trust=signals.get("V", signals.get("accuracy", 0.5)),
                multi_signal_signals=signals,
            )
            # Keep client.trust_score in sync with manager (aggregation / reporting)
            client.trust_score = self.trust_manager.get_trust(client.client_id)
            self.compliance.log_event({
                "event_type": "trust_update",
                "round": round_num,
                "agent_id": client.client_id,
                "trust": self.trust_manager.get_trust(client.client_id),
                "signals": {
                    "V": signals.get("V"),
                    "S": signals.get("S"),
                    "D": signals.get("D"),
                    "U": signals.get("U"),
                    "C": signals.get("C"),
                    "R": signals.get("R"),
                    "contextual_risk": signals.get("contextual_risk"),
                },
                "response_fidelity_penalty": operational_penalty,
                "rl_delta": signals.get("rl_trust_calibration_delta", 0.0),
            })
            # Record V/S/D history AFTER fusion so this round's signals use prior history,
            # then get_model_update will no-op if the round was already recorded.
            client.record_round_performance(round_num)
            include_data = not bool(self.cfg.enable_weighted_secagg)
            update = client.get_model_update(round_num=round_num, include_data=include_data)
            if self.late_compromise is not None:
                update = self.late_compromise.maybe_sign_flip_update(
                    client.client_id, update, round_num
                )
            if self.adversary:
                update = self.adversary.poison_model_update(client.client_id, update)
            client_updates.append(update)
            client_round_rewards[client.client_id] = []

        self._apply_governed_trust_for_aggregation(client_updates)
        if self.cfg.enable_weighted_secagg:
            if not self.cfg.enable_governance:
                ts = np.array(
                    [float(u.get("trust", 0.0)) for u in client_updates], dtype=float
                )
                total = float(ts.sum())
                if total <= 0:
                    weights = np.ones(len(ts)) / max(len(ts), 1)
                else:
                    weights = ts / total
                for u, w in zip(client_updates, weights):
                    u["aggregation_weight"] = float(w)
            self.global_model = self._aggregate_weighted_secagg(client_updates, round_num)
        else:
            self.global_model = self.aggregator.aggregate(client_updates, use_retraining=True)

        ppo_used_this_round = self.response_strategy.ppo_active
        round_rewards: List[float] = []
        round_incidents: List[Tuple[DetectionResult, float]] = []
        for agent in self.agents:
            agent.set_global_model(self.global_model)
            X, y = agent.observe()
            detections = agent.detect(X, y, round_num=round_num)
            risk = client_contextual_risk.get(agent.agent_id, 0.0)
            for det in detections:
                det.contextual_risk = risk
            if self.adversary:
                detections = self.adversary.inject_rl_exploitation(detections)
            trust_score = self.trust_manager.get_trust(agent.agent_id)

            for det in detections[: self.cfg.max_response_incidents_per_round]:
                round_incidents.append((det, trust_score))
                allowed = (
                    self.policy_engine.get_allowed_actions(trust_score)
                    if self.cfg.enable_governance else None
                )
                action = self.response_strategy.select_action(det, trust_score, allowed)
                if not action.target:
                    action.target = getattr(det, "device_type", "") or ""

                if self.cfg.enable_governance:
                    result = self.validator.validate(
                        action,
                        trust_score,
                        round_num,
                        agent_id=agent.agent_id,
                        device_type=getattr(det, "device_type", None),
                        attack_severity=getattr(det, "attack_severity", det.confidence),
                        contextual_risk=getattr(det, "contextual_risk", 0.0),
                        device_criticality=getattr(det, "device_criticality", 0.5),
                        attack_class=getattr(det, "attack_class", None),
                        experiment_run_id=self.policy_repository.experiment_run_id,
                    )
                    status, action, reason = result.status, result.action, result.reason
                    human_review = bool(result.human_review)
                    matched_rule_ids = result.matched_rule_ids
                    decision_id = result.decision_id
                else:
                    status, reason, human_review = "approved", "governance_disabled", False
                    matched_rule_ids, decision_id = None, None

                self.compliance.log_event({
                    "event_type": "response_proposed",
                    "round": round_num,
                    "agent_id": det.agent_id,
                    "action": action.action_type,
                    "confidence": action.confidence,
                    "device_type": getattr(det, "device_type", ""),
                    "status": status,
                    "reason": reason,
                    "human_review": human_review,
                    "matched_rule_ids": matched_rule_ids,
                    "decision_id": decision_id,
                })
                if self.cfg.enable_governance:
                    st = str(status or "").lower()
                    if st and st not in ("approved", "approve"):
                        self._gov_intervention_count += 1
                    self._gov_decision_count += 1

                outcome, reward = self.simulator.step(det, action, status, trust_score)
                event_type = (
                    "policy_violation" if status == "rejected" else "response_executed"
                )
                self.compliance.log_event({
                    "event_type": event_type,
                    "round": round_num,
                    "agent_id": det.agent_id,
                    "action": action.action_type,
                    "status": status,
                    "reason": reason,
                    "human_review": human_review if self.cfg.enable_governance else False,
                    "fp_response": outcome.get("fp_response_flag", False),
                    "reward": reward,
                })
                client_round_rewards[det.agent_id].append(reward)
                agent.record_response_outcome({
                    "fp_response": outcome.get("fp_response_flag", False),
                    "policy_violation": outcome.get("policy_violation", 0) > 0,
                    "inappropriate_severity": outcome.get("inappropriate_severity", False),
                    "human_review": human_review,
                    "severity": action.severity,
                })
                round_rewards.append(reward)

        rl_train_status: Dict[str, Any] = {"status": "skipped"}
        if self.cfg.enable_rl and round_incidents:
            buf_size = int(self.rl_config.get("training", {}).get("replay_buffer_size", 2000))
            self._incident_buffer.extend(round_incidents)
            self._incident_buffer = self._incident_buffer[-buf_size:]

            def allowed_fn(trust: float) -> List[str]:
                if self.cfg.enable_governance:
                    return self.policy_engine.get_allowed_actions(trust)
                return list(ACTION_TYPES)

            rl_train_status = self.response_strategy.train_on_incidents(
                self._incident_buffer, allowed_actions_fn=allowed_fn
            )
        trust_cal_updates = self.trust_calibrator.update_from_round(
            client_round_rewards, enable_rl=self.cfg.enable_rl
        )

        det_metrics = {}
        if self.global_model is not None and self.X_test is not None and self.y_test is not None:
            X_eval = self.X_test
            cols = getattr(self.aggregator, "training_feature_columns", None)
            if cols:
                X_eval = self.X_test.reindex(columns=cols, fill_value=0.0)
            det_metrics = evaluate_model_on_test(self.global_model, X_eval, self.y_test)

        # Resource metrics (Table IV Resource Aware): wall latency + model-update bytes.
        uplink_bytes = int(sum(estimate_update_payload_bytes(u) for u in client_updates))
        # One global-model broadcast per participating client (symmetric LR payload).
        per_model = (
            int(uplink_bytes / len(client_updates)) if client_updates else 0
        )
        downlink_bytes = int(per_model * len(client_updates))
        n_uplinks = len(client_updates)
        resources = {
            "round_latency_sec": float(time.perf_counter() - t0),
            "uplink_bytes": uplink_bytes,
            "downlink_bytes": downlink_bytes,
            "messages": n_uplinks + (n_uplinks if downlink_bytes else 0),
            "participating_clients": n_uplinks,
        }

        # Per-client trust / V / α for natural-partition analysis (WP5)
        trust_scores = self.trust_manager.get_all_trust_scores()
        alpha_by_id = {
            str(u.get("client_id")): float(u.get("aggregation_weight", u.get("trust", 0.0)))
            for u in client_updates
        }
        # If no updates (all skipped), fall back to renormalized trust
        if not alpha_by_id and trust_scores:
            total_t = sum(trust_scores.values()) or 1.0
            alpha_by_id = {k: float(v) / total_t for k, v in trust_scores.items()}
        attackers = set(self.late_compromise.attacker_ids) if self.late_compromise else set()
        client_trust = {}
        for client in self.clients:
            cid = client.client_id
            try:
                v_i = float(client.get_validation_performance_metric())
            except Exception:
                v_i = None
            client_trust[cid] = {
                "T": float(trust_scores.get(cid, client.trust_score or 0.0)),
                "V": v_i,
                "alpha": float(alpha_by_id.get(cid, 0.0)),
                "is_attacker": cid in attackers,
            }

        return {
            "round": round_num,
            "mean_reward": float(np.mean(round_rewards)) if round_rewards else 0.0,
            "ppo_used_for_responses": ppo_used_this_round,
            "ppo_trained_after_round": self.response_strategy.ppo_active,
            "rl_train": rl_train_status,
            "trust_calibration": self.trust_calibrator.summarize(),
            "trust_cal_updates": trust_cal_updates,
            "detection": det_metrics,
            "response_summary": self.simulator.summarize(),
            "resources": resources,
            "client_trust": client_trust,
            "compromise_active": bool(
                self.late_compromise and self.late_compromise.is_compromised_round(round_num)
            ),
            "attack_active": bool(
                poison_report.get("attack_active")
                if poison_report
                else (
                    self.late_compromise.is_attack_active(round_num)
                    if self.late_compromise
                    else False
                )
            ),
            "recovery_phase": bool(poison_report.get("recovery")) if poison_report else False,
            "adversary_mode": self.adversary.mode if self.adversary and self.adversary.active else "none",
            "adversary": self.adversary.summarize() if self.adversary else None,
        }

    def run(self) -> Dict[str, Any]:
        self.setup_clients()
        round_logs = []
        for r in range(1, self.cfg.num_rounds + 1):
            log = self.run_round(r)
            round_logs.append(log)
            ppo_tag = "PPO" if log.get("ppo_used_for_responses") else "static"
            print(
                f"Round {r} [{ppo_tag}]: F1={log['detection'].get('f1_score', 0):.3f} "
                f"reward={log['mean_reward']:.3f} "
                f"resp_prec={log['response_summary']['response_precision']:.3f}"
            )

        resp = self.simulator.summarize()
        last_det = round_logs[-1]["detection"] if round_logs else {}
        late_tag = None
        if self.cfg.dataset == "iomt_natural":
            if self.late_compromise is None or not self.late_compromise.cfg.enabled:
                late_tag = "benign"
            else:
                parts = [
                    fraction_tag(
                        self.late_compromise.cfg.adversary_fraction,
                        len(self.late_compromise.attacker_ids),
                        len(self.clients) or 12,
                    ),
                ]
                mode = self.late_compromise.cfg.poison_mode
                if mode and mode != "label_flip":
                    parts.append(mode)
                late_tag = "_".join(parts)
        run_id = resolve_metrics_run_id(
            self.cfg.approach,
            enable_lambda5=self.cfg.enable_lambda5,
            adversary_mode=self.cfg.adversary_mode,
            dataset=self.cfg.dataset,
            disable_cr_signals=self.cfg.disable_cr_signals,
            uniform_contextual_prior=self.cfg.uniform_contextual_prior,
            late_compromise_tag=late_tag,
            include_R_in_T=bool(self.include_R_in_T),
        )
        # DO3 bake-off: unmasked PPO must not overwrite Maskable B5 metrics.
        if (
            self.cfg.enable_rl
            and self.cfg.approach == "trustfed_agent"
            and not bool(self.rl_config.get("use_action_masking", True))
        ):
            run_id = f"{run_id}_nomask"
        if self.cfg.metrics_suffix:
            run_id = f"{run_id}_{self.cfg.metrics_suffix}"
        adv_summary = self.adversary.summarize() if self.adversary else None
        metrics = {
            "approach": self.cfg.approach,
            "run_id": run_id,
            "dataset": self.cfg.dataset,
            "seed": self.cfg.random_state,
            "num_rounds": self.cfg.num_rounds,
            "adversary": self.cfg.adversary_mode,
            "enable_lambda5": self.cfg.enable_lambda5,
            "disable_cr_signals": self.cfg.disable_cr_signals,
            "uniform_contextual_prior": self.cfg.uniform_contextual_prior,
            "include_R_in_T": self.include_R_in_T,
            "late_compromise": (
                self.late_compromise.summarize() if self.late_compromise else None
            ),
            "signal_weights": dict(self.trust_manager.signal_weights),
            "enable_governance": self.cfg.enable_governance,
            "enable_rl": self.cfg.enable_rl,
            "enable_weighted_secagg": self.cfg.enable_weighted_secagg,
            "adversary_summary": adv_summary,
            "detection": {
                "f1": last_det.get("f1_score"),
                "fnr": last_det.get("false_negative_rate"),
                "fpr": last_det.get("false_positive_rate"),
                "precision": last_det.get("precision"),
                "recall": last_det.get("recall"),
                "accuracy": last_det.get("accuracy"),
            },
            "trust": summarize_trust_metrics(self.trust_manager),
            "trust_discrimination": (
                compute_natural_trust_discrimination(
                    round_logs,
                    compromise_round=self.late_compromise.t_a,
                    attacker_ids=sorted(self.late_compromise.attacker_ids),
                )
                if self.late_compromise and self.late_compromise.cfg.enabled
                else {"available": False}
            ),
            "trust_recovery": (
                compute_trust_recovery(
                    round_logs,
                    compromise_round=self.late_compromise.t_a,
                    attacker_ids=sorted(self.late_compromise.attacker_ids),
                )
                if self.late_compromise
                and self.late_compromise.cfg.enabled
                and self.late_compromise.uses_on_off()
                else {"available": False, "reason": "not_on_off"}
            ),
            "robustness": summarize_robustness_metrics(
                round_logs, adversary_summary=adv_summary
            ),
            "resources": summarize_resource_metrics(round_logs),
            "response": {
                "precision": resp["response_precision"],
                "fp_response_rate": resp["fp_response_rate"],
                "cumulative_defense_utility": resp["cumulative_defense_utility"],
                "mean_reward": (
                    float(resp["cumulative_defense_utility"]) / max(len(self.simulator.history), 1)
                    if self.simulator.history
                    else 0.0
                ),
                **self.response_strategy.proposal_stats(),
            },
            "governance": {
                "violation_rate": resp["violation_rate"],
                "compliance_rate": 1.0 - resp["violation_rate"],
                "audit_completeness": self.compliance.audit_completeness,
                "human_review_rate": self.policy_repository.human_review_rate(
                    self.policy_repository.experiment_run_id
                )
                if self.cfg.enable_governance else 0.0,
                "policy_db": str(self.policy_repository.db_path),
                "schema_version": getattr(
                    self.policy_repository, "schema_version", None
                ),
                "experiment_run_id": self.policy_repository.experiment_run_id,
                "intervention_rate": (
                    float(self._gov_intervention_count) / float(self._gov_decision_count)
                    if self._gov_decision_count
                    else 0.0
                ),
                "n_governance_decisions": int(self._gov_decision_count),
                "n_governance_interventions": int(self._gov_intervention_count),
            },
            "rl": {
                "ppo_active_at_end": self.response_strategy.ppo_active,
                "final_train_status": self.response_strategy.train_status,
                "use_action_masking": self.response_strategy.use_action_masking,
                "rounds_with_ppo": sum(
                    1 for log in round_logs if log.get("ppo_used_for_responses")
                ),
                "trust_calibration": self.trust_calibrator.summarize(),
                **self.response_strategy.proposal_stats(),
            },
            "round_logs": round_logs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out_dir = self.root / "results" / "trustfed_agent" / "metrics"

        # Optional XAI forensic export (master Contrib. 5) for compromised hospital_09
        # Must run before save so metrics JSON includes xai + DB summary.
        try:
            from xai.explainer import (
                build_forensic_record,
                export_forensic_json,
                load_audit_events,
                summarize_governance_db,
            )
            from xai.visualize import plot_hospital09_forensic

            audit_path = Path(self.compliance.log_path)
            xai_dir = self.root / "results" / "trustfed_agent" / "xai"
            events = load_audit_events(audit_path)
            weights = dict(self.trust_manager.signal_weights)
            agent_id = "hospital_09_compromised_iomt"
            gov_run_id = (
                self.policy_repository.experiment_run_id
                if self.cfg.enable_governance
                else None
            )
            record = build_forensic_record(
                agent_id=agent_id,
                round_num=1,
                events=events,
                weights=weights,
                policy_db=Path(self.policy_repository.db_path)
                if self.cfg.enable_governance else None,
                audit_path=audit_path,
                include_condition_log=True,
                experiment_run_id=gov_run_id,
            )
            xai_tag = (
                f"{agent_id}_{run_id}_seed_{self.cfg.random_state}"
                if self.cfg.enable_weighted_secagg
                else f"{agent_id}_seed_{self.cfg.random_state}"
            )
            fig_tag = (
                f"hospital_09_forensic_{run_id}_seed_{self.cfg.random_state}"
                if self.cfg.enable_weighted_secagg
                else f"hospital_09_forensic_seed_{self.cfg.random_state}"
            )
            forensic_path = export_forensic_json(
                record, xai_dir / f"forensic_{xai_tag}.json"
            )
            fig_path = plot_hospital09_forensic(
                audit_path,
                agent_id=agent_id,
                weights=weights,
                out_path=xai_dir / f"{fig_tag}.png",
                policy_db=Path(self.policy_repository.db_path)
                if self.cfg.enable_governance else None,
                experiment_run_id=gov_run_id,
                decision_id=(record.get("governance") or {}).get("decision_id"),
            )
            if self.cfg.enable_governance:
                metrics["governance"].update(
                    summarize_governance_db(
                        Path(self.policy_repository.db_path),
                        experiment_run_id=gov_run_id,
                    )
                )
            metrics["xai"] = {
                "forensic_json": str(forensic_path),
                "figure": str(fig_path),
                "narrative": record.get("narrative"),
                "schema": record.get("schema"),
                "n_condition_logs": len(
                    (record.get("governance") or {}).get("condition_evaluation_log") or []
                ),
                "decision_id": (record.get("governance") or {}).get("decision_id"),
            }
            print(f"XAI forensic: {forensic_path}")
            print(f"XAI figure:   {fig_path}")
        except Exception as exc:  # noqa: BLE001 — XAI must not fail the experiment
            metrics["xai"] = {"error": str(exc)}
            print(f"XAI export skipped: {exc}")

        path = save_trustfed_agent_metrics(
            metrics, out_dir, self.cfg.approach, self.cfg.random_state, run_id=run_id
        )
        metrics["metrics_path"] = str(path)
        if self.cfg.enable_weighted_secagg:
            workstream = self.root / "privacy_hard_path" / "output"
            workstream.mkdir(parents=True, exist_ok=True)
            dest = workstream / path.name
            shutil.copy2(path, dest)
            metrics["privacy_hard_path_output"] = str(dest)

        return metrics
