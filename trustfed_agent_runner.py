"""
TrustFed-RL experiment runner (CLI entry point).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# macOS vecLib/OpenMP: sklearn/numpy matmul can SIGSEGV without this.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from agent_experiment_runner import AgentExperimentConfig, AgentExperimentRunner
from config_loader import resolve_dataset_configs, resolve_trust_config_path


def _default_data_dir(dataset: str = "iomt") -> str:
    if dataset == "iomt_natural":
        candidate = _ROOT / "data" / "CSVs" / "iomt_natural" / "clients"
    elif dataset == "wustl_ehms":
        candidate = _ROOT / "data" / "CSVs" / "wustl_ehms_clients"
    else:
        candidate = _ROOT / "data" / "CSVs" / "iomt_clients"
    return str(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustFed-RL experiment runner")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--approach",
        type=str,
        default="trustfed_agent",
        choices=["trustfed_agent", "trustfed_static", "trustfed_governance", "trustfed_rl_only", "trustfed_agent_privacy"],
    )
    parser.add_argument("--num-rounds", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--enable-lambda5", action="store_true", default=True)
    parser.add_argument("--no-lambda5", dest="enable_lambda5", action="store_false")
    parser.add_argument(
        "--no-cr-signals",
        action="store_true",
        help="Ablation B5−S: disable communication (C) and contextual risk (R) trust signals",
    )
    parser.add_argument(
        "--uniform-b-prior",
        action="store_true",
        help="Use uniform contextual-risk prior b_i=0.10 for all clients (no planted tier lookup)",
    )
    parser.add_argument("--enable-governance", action="store_true", default=True)
    parser.add_argument("--no-governance", dest="enable_governance", action="store_false")
    parser.add_argument("--enable-rl", action="store_true", default=True)
    parser.add_argument("--no-rl", dest="enable_rl", action="store_false")
    parser.add_argument("--adversary", choices=["static", "co_adaptive"], default="static")
    parser.add_argument("--dataset", choices=["iomt", "iomt_natural", "wustl_ehms"], default="iomt")
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--val-ref-csv", type=str, default=None)
    parser.add_argument(
        "--include-r-in-t",
        action="store_true",
        help="B5-R ablation: fuse contextual safety R into behavioural trust T (default off for iomt_natural)",
    )
    parser.add_argument(
        "--no-r-in-t",
        action="store_true",
        help="Force R∉T even on legacy iomt configs",
    )
    parser.add_argument(
        "--compromise-round",
        type=int,
        default=None,
        help="Late-compromise t_a (default from config/iomt_natural_adversary.json)",
    )
    parser.add_argument(
        "--flip-p",
        type=float,
        default=None,
        help="Label-flip probability for attackers after t_a",
    )
    parser.add_argument(
        "--poison-mode",
        choices=["label_flip", "sign_flip", "on_off", "label_flip_on_off", "comm_skip"],
        default=None,
        help="WP7 attack type (default label_flip)",
    )
    parser.add_argument(
        "--adversary-fraction",
        type=float,
        default=None,
        help="WP7 fraction of clients compromised (e.g. 0.1, 0.2, 0.3, 0.4); overrides attacker-ids",
    )
    parser.add_argument(
        "--on-off-attack-rounds",
        type=int,
        default=None,
        help="On–off: attack window length after t_a before recovery (default 5)",
    )
    parser.add_argument(
        "--on-off-period",
        type=int,
        default=None,
        help="On–off: alternate attack/benign blocks of this length (0=single window)",
    )
    parser.add_argument(
        "--attacker-ids",
        type=str,
        default=None,
        help="Comma-separated attacker client ids (e.g. client_03,client_07,client_11)",
    )
    parser.add_argument(
        "--no-late-compromise",
        action="store_true",
        help="Disable warm-up/late-compromise schedule (benign-only run)",
    )
    parser.add_argument("--trust-config", default=None)
    # None → resolve_dataset_configs picks IoMT-specific JSON for --dataset iomt
    parser.add_argument("--governance-config", default=None)
    parser.add_argument("--rl-config", default=None)
    parser.add_argument("--agent-config", default=str(_ROOT / "config" / "agent_config.json"))
    parser.add_argument(
        "--metrics-suffix",
        type=str,
        default=None,
        help="Append tag to metrics run_id (e.g. bakeoff) to avoid overwriting locked exports",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = _default_data_dir(args.dataset)
    if args.test_csv is None and args.dataset == "iomt_natural":
        t = _ROOT / "data" / "CSVs" / "iomt_natural" / "iomt_test_set.csv"
        if t.exists():
            args.test_csv = str(t)
    if args.val_ref_csv is None and args.dataset == "iomt_natural":
        v = _ROOT / "data" / "CSVs" / "iomt_natural" / "iomt_val_ref.csv"
        if v.exists():
            args.val_ref_csv = str(v)

    enable_governance = args.enable_governance
    enable_rl = args.enable_rl
    if args.approach == "trustfed_static":
        enable_governance, enable_rl = False, False
    elif args.approach == "trustfed_governance":
        enable_rl = False
    elif args.approach == "trustfed_rl_only":
        enable_governance = False

    gov_path, rl_path = resolve_dataset_configs(
        args.dataset, args.governance_config, args.rl_config, _ROOT,
    )
    trust_path = resolve_trust_config_path(args.dataset, args.trust_config, _ROOT)

    include_r: bool | None = None
    if args.include_r_in_t:
        include_r = True
    elif args.no_r_in_t or args.dataset == "iomt_natural":
        include_r = False

    late_cfg = None
    if args.dataset == "iomt_natural" and not args.no_late_compromise:
        from adversary.late_compromise import LateCompromiseConfig
        import json
        adv_path = _ROOT / "config" / "iomt_natural_adversary.json"
        raw = json.loads(adv_path.read_text()) if adv_path.exists() else {"enabled": True}
        if args.compromise_round is not None:
            raw["compromise_round"] = args.compromise_round
        if args.flip_p is not None:
            raw["flip_p"] = args.flip_p
        if args.poison_mode is not None:
            raw["poison_mode"] = args.poison_mode
        if args.adversary_fraction is not None:
            raw["adversary_fraction"] = args.adversary_fraction
        if args.on_off_attack_rounds is not None:
            raw["on_off_attack_rounds"] = args.on_off_attack_rounds
        if args.on_off_period is not None:
            raw["on_off_period"] = args.on_off_period
        if args.attacker_ids:
            raw["attacker_client_ids"] = [x.strip() for x in args.attacker_ids.split(",") if x.strip()]
        late_cfg = LateCompromiseConfig.from_dict(raw)
        late_cfg.seed = int(args.random_state)
    elif args.no_late_compromise:
        from adversary.late_compromise import LateCompromiseConfig
        late_cfg = LateCompromiseConfig(enabled=False)

    cfg = AgentExperimentConfig(
        data_dir=args.data_dir,
        approach=args.approach,
        num_rounds=args.num_rounds,
        random_state=args.random_state,
        enable_lambda5=args.enable_lambda5,
        disable_cr_signals=args.no_cr_signals,
        enable_governance=enable_governance,
        enable_rl=enable_rl,
        adversary_mode=args.adversary,
        dataset=args.dataset,
        test_csv=args.test_csv,
        val_ref_csv=args.val_ref_csv,
        trust_config_path=trust_path,
        governance_config_path=gov_path,
        rl_config_path=rl_path,
        agent_config_path=args.agent_config,
        enable_weighted_secagg=(args.approach == "trustfed_agent_privacy"),
        uniform_contextual_prior=bool(args.uniform_b_prior) or args.dataset == "iomt_natural",
        include_R_in_T=include_r,
        late_compromise=late_cfg,
        metrics_suffix=getattr(args, "metrics_suffix", None),
    )
    runner = AgentExperimentRunner(cfg, _ROOT)
    metrics = runner.run()
    print(f"Metrics saved: {metrics.get('metrics_path')}")
    print(f"Configs: gov={gov_path} rl={rl_path}")


if __name__ == "__main__":
    main()
