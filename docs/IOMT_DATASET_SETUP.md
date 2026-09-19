# CICIoMT2024 Dataset Setup

TrustFed-Agent uses the [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) dataset for federated hospital IoMT intrusion detection experiments.

## 1. Request access

1. Open the [CIC IoMT 2024 dataset page](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html).
2. Scroll to the download button and complete the CIC request form.
3. Extract the archive locally, e.g.:

```text
~/datasets/CICIoMT2024/
  WiFi_and_MQTT/
    attacks/
      csv/
        train/
        test/
  Bluetooth/
    attacks/
      csv/
        ...
```

## 2. Convert to hospital federated clients

From the TrustFed-Agent root:

```bash
export CICIOMT_SOURCE="$HOME/datasets/CICIoMT2024/WiFi_and_MQTT/attacks/csv/train"

python3 scripts/convert_iomt_to_hospital_clients.py \
  --source-dir "$CICIOMT_SOURCE" \
  --output-dir ../data/CSVs/iomt_clients \
  --test-output ../data/CSVs/iomt_test_set.csv \
  --num-hospitals 12 \
  --seed 42
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--source-dir` | required | Directory with CICIoMT CSV feature files (searched recursively) |
| `--output-dir` | `../data/CSVs/iomt_clients` | Per-hospital client CSV output |
| `--test-output` | `../data/CSVs/iomt_test_set.csv` | Held-out test set |
| `--num-hospitals` | 12 | Number of federated hospital silos |
| `--partition` | `protocol` | `protocol`, `attack`, or `hash` |
| `--demo` | off | Generate synthetic IoMT-like data (no download required) |

## 3. Verify

```bash
python3 run_experiments.py check-data
python3 run_experiments.py regression --dataset iomt --seed 42 --num-rounds 2
```

## 4. Run healthcare experiments

```bash
python3 run_experiments.py agent --approach trustfed_agent --dataset iomt --seed 42
python3 run_experiments.py paper-claims
```

## Hospital partition semantics

| Scheme | Federated story |
|--------|-----------------|
| `protocol` | Hospitals specialize by Wi-Fi / MQTT / Bluetooth traffic mix |
| `attack` | Hospitals see different attack exposure (non-IID) |
| `hash` | Balanced random split across 12 silos |

Compromised hospitals (`*_compromised_*` in filename) are used for co-adaptive adversary experiments.

## Citation

```bibtex
@article{dadkhah2024ciciomt,
  title={CICIoMT2024: Attack Vectors in Healthcare devices-A Multi-Protocol Dataset for Assessing IoMT Device Security},
  author={Dadkhah, Sajjad and Neto, Elias C P and Ferreira, Rafael and others},
  journal={Internet of Things},
  volume={28},
  year={2024}
}
```
