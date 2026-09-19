# TrustFed-RL — Installation & Experiments

## 1. Prerequisites

- Python **3.9–3.12** (PyTorch/stable-baselines3 may not support 3.13+ yet)
- CICIoMT2024 hospital CSVs (or demo data). See [docs/IOMT_DATASET_SETUP.md](docs/IOMT_DATASET_SETUP.md).

## 2. Quick setup

```bash
cd TrustFed-Agent
./scripts/setup.sh
source .venv/bin/activate   # or .venv312 if present
bash scripts/setup_iomt_data.sh          # needs CICIOMT_SOURCE
# or: bash scripts/setup_iomt_data.sh --demo
python scripts/verify_install.py
python scripts/check_data.py
```

## 3. Run experiments

### Full IoMT matrix (recommended)

```bash
bash scripts/run_trustfed_rl_iomt_matrix.sh
# subset: PHASES="B0,B1,B5" bash scripts/run_trustfed_rl_iomt_matrix.sh
python scripts/check_trustfed_rl_iomt_status.py
python run_experiments.py analyze
```

### Unified CLI

```bash
python run_experiments.py verify
python run_experiments.py check-data
python run_experiments.py regression --dataset iomt --seed 42
python run_experiments.py agent --approach trustfed_agent --dataset iomt --seed 42
python run_experiments.py matrix                        # IoMT B1–B5† matrix
python run_experiments.py analyze
```

### Colab

1. Pack data: `bash scripts/pack_iomt_data_bundle.sh` → upload zip to `My Drive/trustfed/`
2. Open [`colab/Run_TrustFed_RL_IoMT_Matrix.ipynb`](colab/Run_TrustFed_RL_IoMT_Matrix.ipynb) in Google Colab
3. Follow [`colab/COLAB_INSTRUCTIONS.md`](colab/COLAB_INSTRUCTIONS.md) (30 rounds, master matrix)

## 4. Configuration

| File | Purpose |
|------|---------|
| `config/experiment_config.json` | Seeds, IoMT paths, approach matrix |
| `config/trust_config.json` | Six-signal trust weights (V,S,D,U,C,R) |
| `config/iomt_governance_config.json` | Healthcare governance (IoMT) |
| `config/iomt_rl_config.json` | PPO / reward (IoMT) |
| `config/governance_config.json` | Shared governance defaults |
| `config/rl_config.json` | Shared RL defaults |

## 5. Outputs

| Path | Content |
|------|---------|
| `results/trustfed_agent/metrics/run_*_iomt_*.json` | Per-run IoMT metrics |
| `results/trustfed_agent/audit/audit_*.jsonl` | Governance audit log |
| `results/trustfed_agent/summary.json` | Aggregated summary (after analyze) |
| `results/trustfed_agent/logs/` | Matrix runner logs |

## 6. Package layout

```
src/
  agent_experiment_runner.py   # Full FL + governance + response loop
  agents/                      # AutonomousAgent
  governance/                  # Policy, compliance, validator
  rl/                          # Simulator, PPO, gym env, Δτ
  adversary/                   # Co-adaptive adversary
  federated_*.py               # Trust-weighted FL core
```

## 7. Approaches (master IoMT matrix)

| ID | CLI | Trust | Gov | RL | Adversary |
|----|-----|-------|-----|-----|-----------|
| B0 | `regression --baseline fedavg` | equal | — | — | static |
| B1 | `regression --baseline b1` | 6-signal | — | — | static |
| B3 | `trustfed_governance` | 6-signal | ✓ | — | static |
| B4 | `trustfed_rl_only` | 6-signal | — | ✓ | static |
| B5−S | `trustfed_agent --no-cr-signals` | no C,R | ✓ | ✓ | static |
| B5 | `trustfed_agent` | 6-signal | ✓ | ✓ | static |
| B5† | `trustfed_agent --adversary co_adaptive` | 6-signal | ✓ | ✓ | co-adaptive |
| B4† | `trustfed_rl_only --adversary co_adaptive` | 6-signal | — | ✓ | co-adaptive |

## 8. Troubleshooting

- **No CSV files:** `bash scripts/setup_iomt_data.sh` or `--demo`; see `docs/IOMT_DATASET_SETUP.md`
- **torch install slow:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **sb3 import error:** `pip install stable-baselines3 sb3-contrib gymnasium`
- **macOS BLAS segfault:** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`
