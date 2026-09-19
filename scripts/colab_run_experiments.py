#!/usr/bin/env python3
"""
Google Colab runner for TrustFed-RL IoMT master matrix.

Mounts Drive, restores prior metrics, runs phases seed-by-seed with backup
after each seed, and optionally downloads a results zip.

Usage (Colab) — notebook already mounts Drive in cell 1, then:
  %cd /content/TrustFed-Agent
  !python scripts/colab_run_experiments.py --task trustfed_rl_matrix --no-mount --download

(--no-mount is required when Drive was mounted in a notebook cell: drive.mount
cannot run inside a !python subprocess.)

Notebook: colab/Run_TrustFed_RL_IoMT_Matrix.ipynb
Pack on laptop: bash scripts/pack_colab_handoff.sh

Tasks:
  trustfed_rl_matrix  Master IoMT matrix: B0,B1,B3,B4,B5S,B5,B5†,B4† + analyze
  agent_iomt          IoMT B5 only (legacy; prefer trustfed_rl_matrix)
  regression_iomt     IoMT B1 only
  recommended         B1 + B5 + analyze
  analyze             Summarize existing metrics only

Phase ids for --phases: B0,B1,B3,B4,B5S,B5,B5star(=B5†),B4star(=B4†)
  Opt-in: B1U (uniform b_i), B5P (weighted update hiding)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/trustfed")
METRICS_DIR = ROOT / "results" / "trustfed_agent" / "metrics"
RESULTS_DIR = ROOT / "results" / "trustfed_agent"

# Master matrix only (EXPERIMENT_PLAN §3). No B0m / B5−λ.
MATRIX_PHASES: list[tuple[str, str, list[str]]] = [
    (
        "B0",
        "run_trustfed_fedavg_iomt_seed_{seed}.json",
        ["regression", "--baseline", "fedavg", "--dataset", "iomt"],
    ),
    (
        "B1",
        "run_trustfed_b1_iomt_seed_{seed}.json",
        ["regression", "--baseline", "b1", "--dataset", "iomt"],
    ),
    (
        "B3",
        "run_trustfed_governance_iomt_seed_{seed}.json",
        ["agent", "--approach", "trustfed_governance", "--dataset", "iomt"],
    ),
    (
        "B4",
        "run_trustfed_rl_only_iomt_seed_{seed}.json",
        ["agent", "--approach", "trustfed_rl_only", "--dataset", "iomt"],
    ),
    (
        "B5S",
        "run_trustfed_agent_no_cr_iomt_seed_{seed}.json",
        ["agent", "--approach", "trustfed_agent", "--no-cr-signals", "--dataset", "iomt"],
    ),
    (
        "B5",
        "run_trustfed_agent_iomt_seed_{seed}.json",
        ["agent", "--approach", "trustfed_agent", "--dataset", "iomt"],
    ),
    (
        "B5star",
        "run_trustfed_agent_co_adaptive_iomt_seed_{seed}.json",
        [
            "agent",
            "--approach",
            "trustfed_agent",
            "--adversary",
            "co_adaptive",
            "--dataset",
            "iomt",
        ],
    ),
    (
        "B4star",
        "run_trustfed_rl_only_co_adaptive_iomt_seed_{seed}.json",
        [
            "agent",
            "--approach",
            "trustfed_rl_only",
            "--adversary",
            "co_adaptive",
            "--dataset",
            "iomt",
        ],
    ),
]


OPTIONAL_PHASES: list[tuple[str, str, list[str]]] = [
    (
        "B1U",
        "run_trustfed_b1_uniform_iomt_seed_{seed}.json",
        ["regression", "--baseline", "b1", "--uniform-b-prior", "--dataset", "iomt"],
    ),
    (
        "B5P",
        "run_trustfed_agent_privacy_iomt_seed_{seed}.json",
        ["agent", "--approach", "trustfed_agent_privacy", "--dataset", "iomt"],
    ),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def mount_drive() -> None:
    """Mount Google Drive when needed.

    Notebooks usually mount Drive in an earlier cell. Calling ``drive.mount``
    from a ``!python`` subprocess fails with::

        AttributeError: 'NoneType' object has no attribute 'kernel'

    because there is no interactive IPython kernel. If ``/content/drive``
    already exists, we skip mounting. Pass ``--no-mount`` from the notebook
    after mounting in a prior cell (recommended).
    """
    drive_root = Path("/content/drive")
    if drive_root.exists():
        _log(f"Drive already available at {drive_root} — skipping mount")
        return

    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "google.colab is not available. Run this script in Google Colab, "
            "or pass --no-mount and set --drive-dir to a local/Drive path."
        ) from exc

    _log("Mounting Google Drive...")
    try:
        drive.mount(str(drive_root), force_remount=False)
    except AttributeError as exc:
        # Typical when invoked via !python (subprocess) after notebook mount failed
        # or kernel helper is unavailable.
        raise SystemExit(
            "drive.mount() cannot run inside a !python subprocess.\n"
            "Mount Drive in a notebook cell first, then run with --no-mount:\n"
            "  !python scripts/colab_run_experiments.py "
            "--task trustfed_rl_matrix --no-mount --download"
        ) from exc


def load_seeds() -> list[int]:
    cfg_path = ROOT / "config" / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return list(cfg.get("seeds", [42]))


def ensure_deps() -> None:
    req = ROOT / "requirements.txt"
    _log("Installing dependencies from requirements.txt...")
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if req.exists():
        cmd.extend(["-r", str(req)])
    else:
        cmd.extend(["stable-baselines3", "sb3-contrib", "gymnasium", "scikit-learn", "pandas"])
    subprocess.check_call(cmd)


def set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def restore_metrics_zip(drive_dir: Path) -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = drive_dir / "iomt_metrics_latest.zip"
    # Also accept older backup names
    candidates = [
        zip_path,
        drive_dir / "iomt_results_backup.zip",
        drive_dir / "iomt_b5_complete.zip",
    ]
    zip_found = next((p for p in candidates if p.exists()), None)
    if zip_found is None:
        _log(f"No prior metrics backup under {drive_dir}")
        return len(list(METRICS_DIR.glob("run_*.json")))

    _log(f"Restoring metrics from {zip_found}")
    with zipfile.ZipFile(zip_found, "r") as zf:
        zf.extractall("/tmp/trustfed_restore")
    for path in Path("/tmp/trustfed_restore").rglob("run_*.json"):
        dest = METRICS_DIR / path.name
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
    count = len(list(METRICS_DIR.glob("run_*.json")))
    _log(f"Metrics on disk: {count} JSON file(s)")
    return count


def backup_metrics_zip(drive_dir: Path, label: str = "iomt_metrics_latest") -> Path:
    drive_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(METRICS_DIR.glob("run_*.json"))
    if not json_files:
        _log("No metric JSON files to back up yet.")
        return drive_dir / f"{label}.zip"

    latest = drive_dir / f"{label}.zip"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stamped = drive_dir / f"{label}_{stamp}.zip"

    for target in (latest, stamped):
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in json_files:
                zf.write(path, arcname=path.name)
        _log(f"Backed up {len(json_files)} metric(s) -> {target}")
    return latest


def backup_full_results(drive_dir: Path) -> Path:
    drive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = drive_dir / f"trustfed_results_{stamp}.zip"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in RESULTS_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(RESULTS_DIR.parent)))
    _log(f"Full results archive -> {archive}")
    return archive


def setup_iomt_data(drive_dir: Path, demo: bool) -> None:
    bundle = drive_dir / "iomt_data_bundle.zip"
    clients = ROOT / "data" / "CSVs" / "iomt_clients"
    n_clients = len(list(clients.glob("*.csv"))) if clients.exists() else 0
    if n_clients > 0:
        _log(f"IoMT client data already present ({n_clients} hospitals).")
        return
    if bundle.exists():
        _log(f"Extracting IoMT data bundle from {bundle}")
        with zipfile.ZipFile(bundle, "r") as zf:
            zf.extractall(ROOT)
    elif demo:
        _log("No Drive bundle found; generating synthetic IoMT demo data...")
        subprocess.check_call(["bash", str(ROOT / "scripts" / "setup_iomt_data.sh"), "--demo"])
    else:
        raise SystemExit(
            "IoMT data missing. Upload iomt_data_bundle.zip to "
            f"{drive_dir} or pass --demo for a quick smoke test."
        )
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "check_data.py")])


def run_seed(command: list[str]) -> int:
    return subprocess.call(command, cwd=ROOT, env=os.environ.copy())


def run_regression_iomt(drive_dir: Path) -> None:
    seeds = load_seeds()
    for seed in seeds:
        out = METRICS_DIR / f"run_trustfed_b1_iomt_seed_{seed}.json"
        if out.exists():
            _log(f"Skip B1 seed {seed} (exists)")
            continue
        _log(f"\n=== B1 regression (iomt) seed {seed} ===")
        rc = run_seed(
            [sys.executable, str(ROOT / "run_experiments.py"), "regression", "--dataset", "iomt", "--seed", str(seed)]
        )
        if rc != 0:
            raise SystemExit(rc)
        backup_metrics_zip(drive_dir)


def run_agent_iomt(drive_dir: Path) -> None:
    seeds = load_seeds()
    for seed in seeds:
        out = METRICS_DIR / f"run_trustfed_agent_iomt_seed_{seed}.json"
        if out.exists():
            _log(f"Skip B5 seed {seed} (exists)")
            continue
        _log(f"\n=== B5 trustfed_agent (iomt) seed {seed} ===")
        rc = run_seed(
            [
                sys.executable,
                str(ROOT / "run_experiments.py"),
                "agent",
                "--approach",
                "trustfed_agent",
                "--dataset",
                "iomt",
                "--seed",
                str(seed),
            ]
        )
        if rc != 0:
            raise SystemExit(rc)
        backup_metrics_zip(drive_dir)


def run_trustfed_rl_matrix(drive_dir: Path, phases: set[str] | None = None) -> None:
    """Run full TrustFed-RL IoMT matrix seed-by-seed with Drive backup."""
    seeds = load_seeds()
    catalog = list(MATRIX_PHASES)
    if phases is not None and ("B5P" in phases or "B1U" in phases):
        catalog.extend(OPTIONAL_PHASES)
    for phase_id, out_tmpl, argv in catalog:
        if phases is not None and phase_id not in phases:
            continue
        _log(f"\n######## Phase {phase_id} ########")
        for seed in seeds:
            out = METRICS_DIR / out_tmpl.format(seed=seed)
            if out.exists():
                _log(f"Skip {phase_id} seed {seed} (exists)")
                continue
            _log(f"\n=== {phase_id} seed {seed} ===")
            cmd = [sys.executable, str(ROOT / "run_experiments.py"), *argv, "--seed", str(seed)]
            rc = run_seed(cmd)
            if rc != 0:
                raise SystemExit(rc)
            backup_metrics_zip(drive_dir)
    run_analyze()


def run_analyze() -> None:
    rc = run_seed([sys.executable, str(ROOT / "run_experiments.py"), "analyze"])
    if rc != 0:
        raise SystemExit(rc)


def download_file(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore
    except ImportError:
        _log(f"Not in Colab; download manually from Drive: {path}")
        return
    _log(f"Starting browser download: {path.name}")
    files.download(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab runner with Drive backup + download")
    parser.add_argument(
        "--task",
        choices=["trustfed_rl_matrix", "agent_iomt", "regression_iomt", "recommended", "analyze"],
        default="trustfed_rl_matrix",
        help="Which experiment phase to run",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="",
        help="Comma subset of master phases: B0,B1,B3,B4,B5S,B5,B5star,B4star (opt-in B1U,B5P)",
    )
    parser.add_argument("--drive-dir", type=Path, default=DEFAULT_DRIVE_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-mount", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--skip-restore", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_thread_limits()
    os.chdir(ROOT)

    if not args.no_mount:
        mount_drive()

    args.drive_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Drive workspace: {args.drive_dir}")

    if not args.skip_restore:
        restore_metrics_zip(args.drive_dir)

    if args.task != "analyze":
        ensure_deps()
        setup_iomt_data(args.drive_dir, demo=args.demo)

    if args.task == "regression_iomt":
        run_regression_iomt(args.drive_dir)
    elif args.task == "agent_iomt":
        run_agent_iomt(args.drive_dir)
    elif args.task == "recommended":
        run_regression_iomt(args.drive_dir)
        run_agent_iomt(args.drive_dir)
        run_analyze()
    elif args.task == "trustfed_rl_matrix":
        phases = {p.strip() for p in args.phases.split(",") if p.strip()} or None
        run_trustfed_rl_matrix(args.drive_dir, phases=phases)
    elif args.task == "analyze":
        run_analyze()

    backup_metrics_zip(args.drive_dir)
    archive = backup_full_results(args.drive_dir)

    _log("\n=== Done ===")
    _log(f"Metrics backup: {args.drive_dir / 'iomt_metrics_latest.zip'}")
    _log(f"Full archive:   {archive}")
    if args.download:
        download_file(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
