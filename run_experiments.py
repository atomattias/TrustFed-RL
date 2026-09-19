#!/usr/bin/env python3
"""
Unified experiment CLI for TrustFed-RL (IoMT).

Examples:
  python run_experiments.py setup
  python run_experiments.py check-data
  python run_experiments.py regression --dataset iomt
  python run_experiments.py agent --approach trustfed_agent --dataset iomt --seed 42
  python run_experiments.py matrix
  python run_experiments.py analyze
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _load_config() -> dict:
    path = ROOT / "config" / "experiment_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _resolve_paths(dataset: str, data_dir: str | None, test_csv: str | None) -> tuple[Path, Path]:
    cfg = _load_config()
    ds = cfg.get("dataset", {})
    if dataset == "iomt":
        data = ROOT / (data_dir or ds.get("iomt_dir", "data/CSVs/iomt_clients"))
        test = ROOT / (test_csv or ds.get("iomt_test_csv", "data/CSVs/iomt_test_set.csv"))
    elif dataset == "iomt_natural":
        data = ROOT / (data_dir or ds.get("iomt_natural_dir", "data/CSVs/iomt_natural/clients"))
        test = ROOT / (test_csv or ds.get("iomt_natural_test_csv", "data/CSVs/iomt_natural/iomt_test_set.csv"))
    elif dataset == "wustl_ehms":
        data = ROOT / (data_dir or ds.get("wustl_ehms_dir", "data/CSVs/wustl_ehms_clients"))
        test = ROOT / (test_csv or ds.get("wustl_ehms_test_csv", "data/CSVs/wustl_ehms_test_set.csv"))
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}; use iomt, iomt_natural, or wustl_ehms")
    return data.resolve(), test.resolve()


def cmd_setup(_args) -> int:
    return subprocess.call(["bash", str(ROOT / "scripts" / "setup.sh")])


def cmd_check_data(_args) -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "check_data.py")])


def cmd_verify(_args) -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "verify_install.py")])


def cmd_regression(args) -> int:
    sys.path.insert(0, str(ROOT))
    from experiment import ExperimentRunner
    from evaluation import resolve_metrics_run_id, save_trustfed_agent_metrics

    cfg = _load_config()
    dataset = args.dataset
    data_path, test_path = _resolve_paths(dataset, args.data_dir, args.test_csv)
    n_clients = len(list(data_path.glob("*.csv"))) if data_path.exists() else 0
    if n_clients == 0:
        print(f"\nERROR: No client CSVs in {data_path}")
        if dataset == "wustl_ehms":
            print("Run: python scripts/convert_wustl_ehms_to_clients.py")
        elif dataset == "iomt_natural":
            print("Run: python scripts/convert_iomt_natural_clients.py")
        else:
            print("On Colab, prepare data first:")
            print("  bash scripts/setup_iomt_data.sh --demo")
            print("See docs/IOMT_DATASET_SETUP.md")
        return 1
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42])
    out_dir = ROOT / "results" / "trustfed_agent" / "metrics"
    baseline = getattr(args, "baseline", None)
    default_rounds = int(cfg.get("model", {}).get("num_rounds", 30))
    num_rounds = args.num_rounds if args.num_rounds is not None else default_rounds

    for seed in seeds:
        trust_config_path = None
        if baseline == "fedavg":
            approach_key, result_key, label = "trustfed_fedavg", "federated_equal_weight", "B0 FedAvg"
            runner_kw = dict(baseline_only="fedavg", trust_only=False, use_multi_signal=False)
        elif baseline in ("median", "rm"):
            # Path AGG Rm: coordinate-wise median (same late-compromise loop as B0)
            approach_key, result_key, label = "trustfed_rm", "coordinate_median", "Rm coordinate-median"
            runner_kw = dict(baseline_only="median", trust_only=False, use_multi_signal=False)
        elif baseline == "b2":
            # Full behavioural T=f(V,S,D,U,C), R∉T (natural) / intermediate trust-FL
            approach_key, result_key, label = "trustfed_b2", "trust_aware", "B2 behavioural"
            runner_kw = dict(trust_only=True, use_multi_signal=True)
            if dataset == "iomt_natural":
                from config_loader import resolve_trust_config_path
                trust_config_path = resolve_trust_config_path(
                    dataset, trust_profile="b2", root=ROOT
                )
        elif baseline == "bc":
            # Path 1: C-only participation/skip-aware baseline under comm_skip
            approach_key, result_key, label = "trustfed_bc", "trust_aware", "Bc C-only"
            runner_kw = dict(trust_only=True, use_multi_signal=True)
            if dataset == "iomt_natural":
                from config_loader import resolve_trust_config_path
                trust_config_path = resolve_trust_config_path(
                    dataset, trust_profile="c_only", root=ROOT
                )
            else:
                raise SystemExit("--baseline bc is only supported for --dataset iomt_natural")
        else:
            # Default b1: V-only on iomt_natural; legacy multi-signal elsewhere
            if dataset == "iomt_natural":
                approach_key, result_key, label = "trustfed_b1v", "trust_aware", "B1 V-only"
                from config_loader import resolve_trust_config_path
                trust_config_path = resolve_trust_config_path(
                    dataset, trust_profile="b1", root=ROOT
                )
            else:
                approach_key, result_key, label = "trustfed_b1", "trust_aware", "B1"
            runner_kw = dict(trust_only=True, use_multi_signal=True)

        uniform_b = bool(getattr(args, "uniform_b_prior", False))
        if uniform_b and baseline in ("b1", "b2", "bc"):
            if baseline == "b1" and dataset == "iomt_natural":
                label = "B1 V-only (uniform b_i)"
            elif baseline == "b2":
                label = "B2 behavioural (uniform b_i)"
            elif baseline == "bc":
                label = "Bc C-only (uniform b_i)"
            else:
                label = "B1-U (uniform b_i=0.10)"
            runner_kw["uniform_contextual_prior"] = True
        elif uniform_b:
            print("WARNING: --uniform-b-prior only applies to baseline b1/b2/bc; ignoring for this run")

        if trust_config_path:
            runner_kw["trust_config_path"] = trust_config_path

        # Natural partition: benign vs poison tags must not collide on disk
        late_tag = None
        pending_late = None
        if dataset == "iomt_natural":
            from adversary.late_compromise import LateCompromiseConfig
            import json
            cfg_override = getattr(args, "late_compromise_config", None)
            if cfg_override:
                adv_path = Path(cfg_override)
                if not adv_path.is_absolute():
                    adv_path = ROOT / adv_path
            else:
                adv_path = ROOT / "config" / "iomt_natural_adversary.json"
            raw = json.loads(adv_path.read_text()) if adv_path.exists() else {"enabled": True}
            if getattr(args, "no_late_compromise", False):
                pending_late = LateCompromiseConfig(enabled=False)
                late_tag = "benign"
            else:
                if getattr(args, "compromise_round", None) is not None:
                    raw["compromise_round"] = args.compromise_round
                if getattr(args, "flip_p", None) is not None:
                    raw["flip_p"] = args.flip_p
                if getattr(args, "poison_mode", None) is not None:
                    raw["poison_mode"] = args.poison_mode
                if getattr(args, "adversary_fraction", None) is not None:
                    raw["adversary_fraction"] = args.adversary_fraction
                # Explicit attacker IDs (Option C): must clear fraction or resolve_attacker_ids ignores IDs
                attacker_ids_arg = getattr(args, "attacker_ids", None)
                if attacker_ids_arg:
                    ids = [x.strip() for x in str(attacker_ids_arg).split(",") if x.strip()]
                    raw["attacker_client_ids"] = ids
                    raw["adversary_fraction"] = None
                if getattr(args, "on_off_attack_rounds", None) is not None:
                    raw["on_off_attack_rounds"] = args.on_off_attack_rounds
                if getattr(args, "on_off_period", None) is not None:
                    raw["on_off_period"] = args.on_off_period
                if getattr(args, "benign_skip_prob", None) is not None:
                    bs = dict(raw.get("benign_straggler") or {})
                    bs["enabled"] = True
                    bs["skip_prob_q"] = float(args.benign_skip_prob)
                    raw["benign_straggler"] = bs
                pending_late = LateCompromiseConfig.from_dict(raw)
                pending_late.seed = int(seed)
                from adversary.late_compromise import fraction_tag
                # n_attackers resolved after clients exist; approximate tag from fraction/ids
                n_att = (
                    int(round(float(pending_late.adversary_fraction) * 12))
                    if pending_late.adversary_fraction is not None
                    else len(pending_late.attacker_client_ids)
                )
                late_tag = fraction_tag(pending_late.adversary_fraction, n_att, 12)
                mode = pending_late.poison_mode
                if mode and mode != "label_flip":
                    late_tag = f"{late_tag}_{mode}"
                if pending_late.benign_straggler_enabled:
                    late_tag = f"{late_tag}_strag"
                if pending_late.compromise_round > num_rounds:
                    late_tag = "benign"

        run_id = resolve_metrics_run_id(
            approach_key,
            dataset=dataset,
            uniform_contextual_prior=uniform_b and baseline in ("b1", "b2", "bc"),
            late_compromise_tag=late_tag,
        )
        suffix = getattr(args, "metrics_suffix", None)
        if suffix:
            run_id = f"{run_id}_{suffix}"
        out_path = out_dir / f"run_{run_id}_seed_{seed}.json"
        if out_path.exists():
            print(f"Skip {label} seed {seed} (exists: {out_path.name})")
            continue
        print(f"\n=== {label} ({dataset}) seed {seed} rounds={num_rounds} tag={late_tag} ===")
        import time

        t0 = time.perf_counter()
        runner = ExperimentRunner(
            data_dir=str(data_path),
            model_type="logistic_regression",
            random_state=seed,
            test_csv=str(test_path) if test_path.exists() else None,
            num_rounds=num_rounds,
            **runner_kw,
        )
        if pending_late is not None:
            runner._pending_late_cfg = pending_late
        results = runner.run_experiment()
        wall_sec = float(time.perf_counter() - t0)
        block = results.get(result_key) or results.get("trust_aware_no_retrain") or {}
        trust_mgr = getattr(runner, "trust_manager", None)
        from evaluation import (
            summarize_trust_metrics,
            compute_natural_trust_discrimination,
        )
        round_logs = block.get("round_logs") or getattr(runner, "round_logs", []) or []
        late_summary = (
            runner.late_compromise.summarize()
            if getattr(runner, "late_compromise", None)
            else (
                {
                    "enabled": pending_late.enabled,
                    "compromise_round": pending_late.compromise_round,
                    "poison_mode": pending_late.poison_mode,
                    "flip_p": pending_late.flip_p,
                    "attacker_client_ids": list(pending_late.attacker_client_ids),
                    "adversary_fraction": pending_late.adversary_fraction,
                }
                if pending_late is not None
                else None
            )
        )
        disc = None
        if dataset == "iomt_natural" and round_logs and late_summary and late_summary.get("enabled"):
            disc = compute_natural_trust_discrimination(
                round_logs,
                compromise_round=int(late_summary.get("compromise_round") or 20),
                attacker_ids=late_summary.get("attacker_client_ids") or [],
            )
        signal_weights = block.get("signal_weights") or getattr(
            runner, "resolved_signal_weights", None
        )
        metrics = {
            "approach": approach_key,
            "run_id": run_id,
            "dataset": dataset,
            "seed": seed,
            "num_rounds": num_rounds,
            "adversary": "static",
            "enable_lambda5": False,
            "enable_governance": False,
            "enable_rl": False,
            "uniform_contextual_prior": bool(uniform_b and baseline in ("b1", "b2", "bc")),
            "signal_weights": signal_weights,
            "late_compromise": late_summary,
            "round_logs": round_logs,
            "trust_discrimination": disc,
            "detection": {
                "f1": block.get("f1_score"),
                "fnr": block.get("false_negative_rate"),
                "fpr": block.get("false_positive_rate"),
                "precision": block.get("precision"),
                "recall": block.get("recall"),
                "accuracy": block.get("accuracy"),
            },
            "trust": summarize_trust_metrics(trust_mgr),
            "robustness": {
                "detection_f1_final": block.get("f1_score"),
                "adversary": None,
            },
            "resources": {
                "rounds_measured": num_rounds,
                "mean_round_latency_sec": wall_sec / max(num_rounds, 1),
                "total_round_latency_sec": wall_sec,
                "total_uplink_bytes": None,
                "total_downlink_bytes": None,
                "total_communication_bytes": None,
                "mean_messages_per_round": None,
                "resource_aware_measured": True,
                "note": "B0/B1 wall-clock only; per-update byte accounting is in agent runs",
            },
            "response": {
                "precision": None,
                "fp_response_rate": None,
                "cumulative_defense_utility": None,
            },
            "governance": {
                "violation_rate": None,
                "compliance_rate": None,
                "audit_completeness": None,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = save_trustfed_agent_metrics(
            metrics, out_dir, approach_key, seed, run_id=run_id
        )
        print(f"{label} metrics saved: {path}")
    return 0


def cmd_agent(args) -> int:
    cfg = _load_config()
    data_path, test_path = _resolve_paths(args.dataset, args.data_dir, args.test_csv)
    n_clients = len(list(data_path.glob("*.csv"))) if data_path.exists() else 0
    if n_clients == 0:
        print(f"\nERROR: No client CSVs in {data_path}")
        if args.dataset == "wustl_ehms":
            print("Run: python scripts/convert_wustl_ehms_to_clients.py")
        elif args.dataset == "iomt_natural":
            print("Run: python scripts/convert_iomt_natural_clients.py")
        else:
            print("Run: bash scripts/setup_iomt_data.sh --demo  (or with CICIOMT_SOURCE)")
        return 1
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42])
    default_rounds = int(cfg.get("model", {}).get("num_rounds", 30))
    num_rounds = args.num_rounds if args.num_rounds is not None else default_rounds
    for seed in seeds:
        from evaluation import resolve_metrics_run_id
        run_id = resolve_metrics_run_id(
            args.approach,
            enable_lambda5=not args.no_lambda5,
            adversary_mode=args.adversary,
            dataset=args.dataset,
            disable_cr_signals=getattr(args, "no_cr_signals", False),
            uniform_contextual_prior=getattr(args, "uniform_b_prior", False) or args.dataset == "iomt_natural",
        )
        out_path = ROOT / "results" / "trustfed_agent" / "metrics" / f"run_{run_id}_seed_{seed}.json"
        if out_path.exists():
            print(f"Skip {run_id} seed {seed} (exists)")
            continue
        cmd = [
            sys.executable, str(ROOT / "trustfed_agent_runner.py"),
            "--data-dir", str(data_path),
            "--approach", args.approach,
            "--dataset", args.dataset,
            "--test-csv", str(test_path),
            "--num-rounds", str(num_rounds),
            "--random-state", str(seed),
            "--adversary", args.adversary,
        ]
        val_ref = ROOT / "data" / "CSVs" / "iomt_natural" / "iomt_val_ref.csv"
        if args.dataset == "iomt_natural" and val_ref.exists():
            cmd.extend(["--val-ref-csv", str(val_ref)])
        if args.no_lambda5:
            cmd.append("--no-lambda5")
        if getattr(args, "no_cr_signals", False):
            cmd.append("--no-cr-signals")
        if args.no_governance:
            cmd.append("--no-governance")
        if args.no_rl:
            cmd.append("--no-rl")
        if getattr(args, "uniform_b_prior", False) or args.dataset == "iomt_natural":
            cmd.append("--uniform-b-prior")
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


def cmd_matrix(_args) -> int:
    return subprocess.call(["bash", str(ROOT / "scripts" / "run_trustfed_rl_iomt_matrix.sh")])


def cmd_analyze(_args) -> int:
    return subprocess.call([
        sys.executable, str(ROOT / "scripts" / "analyze_trustfed_agent_results.py"),
        "--input", str(ROOT / "results" / "trustfed_agent" / "metrics"),
        "--export", str(ROOT / "results" / "trustfed_agent" / "summary.json"),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustFed-RL experiment orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Create venv and install dependencies")
    sub.add_parser("verify", help="Verify package imports")
    sub.add_parser("check-data", help="Check IoMT dataset paths")
    sub.add_parser("matrix", help="Run TrustFed-RL IoMT matrix (B1–B5†)")
    sub.add_parser("analyze", help="Summarize metrics JSON files")

    p_reg = sub.add_parser("regression", help="B0/B1 FL baselines (IoMT)")
    p_reg.add_argument("--seed", type=int, default=None)
    p_reg.add_argument("--data-dir", type=str, default=None)
    p_reg.add_argument("--test-csv", type=str, default=None)
    p_reg.add_argument("--dataset", choices=["iomt", "iomt_natural", "wustl_ehms"], default="iomt")
    p_reg.add_argument("--num-rounds", type=int, default=None, help="Default from experiment_config (30)")
    p_reg.add_argument(
        "--baseline",
        choices=["b1", "b2", "bc", "fedavg", "median", "rm"],
        default="b1",
        help=(
            "iomt_natural: b1=V-only, b2=behavioural, bc=C-only, fedavg=B0, "
            "rm/median=coordinate-median (Path AGG Rm)"
        ),
    )
    p_reg.add_argument(
        "--uniform-b-prior",
        action="store_true",
        help="B1-U ablation: set contextual-risk prior b_i=0.10 for all clients (no planted tier lookup)",
    )
    p_reg.add_argument("--compromise-round", type=int, default=None)
    p_reg.add_argument("--flip-p", type=float, default=None)
    p_reg.add_argument(
        "--poison-mode",
        choices=["label_flip", "sign_flip", "on_off", "label_flip_on_off", "comm_skip"],
        default=None,
    )
    p_reg.add_argument("--adversary-fraction", type=float, default=None)
    p_reg.add_argument(
        "--attacker-ids",
        type=str,
        default=None,
        help="Comma-separated client IDs (e.g. client_02,client_07). "
        "When set, clears adversary_fraction so IDs are not overridden.",
    )
    p_reg.add_argument("--on-off-attack-rounds", type=int, default=None)
    p_reg.add_argument("--on-off-period", type=int, default=None)
    p_reg.add_argument("--no-late-compromise", action="store_true")
    p_reg.add_argument(
        "--late-compromise-config",
        type=str,
        default=None,
        help=(
            "JSON path for late-compromise schedule (default: "
            "config/iomt_natural_adversary.json). Use "
            "config/iomt_natural_adversary_path_strag.json for Path STRAG."
        ),
    )
    p_reg.add_argument(
        "--benign-skip-prob",
        type=float,
        default=None,
        help="Override benign_straggler.skip_prob_q (Path STRAG explore retune only).",
    )
    p_reg.add_argument(
        "--metrics-suffix",
        type=str,
        default=None,
        help="Append tag to metrics run_id (e.g. budget2x) to avoid overwriting locked exports",
    )

    p_agent = sub.add_parser("agent", help="Run TrustFed-RL approach")
    p_agent.add_argument(
        "--approach",
        default="trustfed_agent",
        choices=[
            "trustfed_agent",
            "trustfed_static",
            "trustfed_governance",
            "trustfed_rl_only",
            "trustfed_agent_privacy",
        ],
    )
    p_agent.add_argument("--seed", type=int, default=None)
    p_agent.add_argument("--data-dir", type=str, default=None)
    p_agent.add_argument("--test-csv", type=str, default=None)
    p_agent.add_argument("--dataset", choices=["iomt", "iomt_natural", "wustl_ehms"], default="iomt")
    p_agent.add_argument("--num-rounds", type=int, default=None)
    p_agent.add_argument("--adversary", default="static", choices=["static", "co_adaptive"])
    p_agent.add_argument("--no-lambda5", action="store_true")
    p_agent.add_argument("--no-cr-signals", action="store_true", help="B5−S ablation: disable C and R signals")
    p_agent.add_argument("--no-governance", action="store_true")
    p_agent.add_argument("--no-rl", action="store_true")
    p_agent.add_argument(
        "--uniform-b-prior",
        action="store_true",
        help="Use uniform contextual-risk prior b_i=0.10 for all clients (no planted tier lookup)",
    )

    args = parser.parse_args()
    handlers = {
        "setup": cmd_setup,
        "verify": cmd_verify,
        "check-data": cmd_check_data,
        "regression": cmd_regression,
        "agent": cmd_agent,
        "matrix": cmd_matrix,
        "analyze": cmd_analyze,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
