# TrustFed-RL

Source code for the experiments reported in the TrustFed-RL paper
(governance- and RL-based multi-signal federated learning for trustworthy
autonomous cyber defence in IoMT settings).

This repository contains **code, configs, and runners only**.
It does **not** ship CICIoMT / ECU raw datasets, experiment result dumps,
paper drafts, or internal planning notes.

## What is included

```
src/                  # FL, trust, governance, RL, adversary modules
config/               # Locked experiment / trust / RL / governance configs
scripts/              # Data conversion + confirmatory experiment runners
tests/                # Unit / integration tests
experiment.py         # Trust-weighted FL engine
run_experiments.py    # Unified CLI
trustfed_agent_runner.py
docs/IOMT_DATASET_SETUP.md
```

## What is not included

- Raw datasets (`data/CSVs`, `data/train`, …) — obtain CICIoMT2024 separately
- Large result archives / metric dumps
- Paper LaTeX drafts and internal `*PLAN*` / QA documents
- Virtualenvs and local caches

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download / place CICIoMT, then convert structured clients
bash scripts/setup_iomt_data.sh          # see docs/IOMT_DATASET_SETUP.md
python run_experiments.py check-data

# Confirmatory IoMT matrix (paper primary path)
bash scripts/run_trustfed_rl_iomt_matrix.sh
```

Additional locked paths used in the paper:

| Path | Script |
|------|--------|
| H1 confirmatory | `scripts/run_h1_confirmatory.sh` |
| Path AGG (`label_flip`) | `scripts/run_path_agg_malicious_robust.sh` |
| Path STRAG (benign straggler) | `scripts/run_path_strag.sh` |
| Path 1 C-only | `scripts/run_path1_c_only_comm_skip.sh` |

See **INSTALL.md** for environment details.

## License

See `LICENSE`.
