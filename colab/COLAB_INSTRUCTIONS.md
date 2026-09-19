# Google Colab — full D1 IoMT master matrix

**Notebook:** [`Run_TrustFed_RL_IoMT_Matrix.ipynb`](Run_TrustFed_RL_IoMT_Matrix.ipynb)  
**Do not use:** `Run_IoMT_Experiments.ipynb` (deprecated)

**Runner:** `scripts/colab_run_experiments.py --task trustfed_rl_matrix`  
**Packer:** `bash scripts/pack_colab_handoff.sh`  
**Config:** 30 rounds · seeds 42–53 · governance `colleague_v1`

Matrix: **B0 → B1 → B3 → B4 → B5−S → B5 → B5† → B4†**

---

## Goal: run the whole matrix on Colab

Local laptop results and Drive are **separate**. You can leave the laptop job running; Colab will not touch those files.  
At the end, **use the Colab / Drive metrics** as the paper source of truth (ignore or archive local).

### Two modes

| Mode | Upload `iomt_metrics_latest.zip`? | Effect |
|------|-----------------------------------|--------|
| **A — Full from scratch (recommended if you want Colab-only)** | **No** (or empty Drive `trustfed/` metrics) | Re-runs every phase/seed on Colab |
| **B — Full matrix but skip what you already finished** | **Yes** | Skips B0–B4 / B5−S / any B5 seeds already in the zip; still runs the rest |

For a clean Colab-only paper run, choose **A**.

---

## 1. Laptop prep

```bash
cd TrustFed-Agent
bash scripts/pack_colab_handoff.sh
```

You need at least:

| File | Mode A (scratch) | Mode B (resume) |
|------|------------------|-----------------|
| `TrustFed-Agent-colab.zip` | upload | upload |
| `iomt_data_bundle.zip` | upload | upload |
| `iomt_metrics_latest.zip` | **do not upload** (or delete it on Drive) | upload |
| `colab/Run_TrustFed_RL_IoMT_Matrix.ipynb` | upload notebook in Colab | same |

Upload zips to **`My Drive/trustfed/`**.

---

## 2. Colab

1. *File → Upload notebook* → `Run_TrustFed_RL_IoMT_Matrix.ipynb`  
2. Runtime → **CPU** (keep tab open / Colab Pro for multi-day)  
3. Run sections **1 → 4** (Drive, install, data, short-round cleanup)  
4. Run section **5 — full matrix**:

```bash
!python scripts/colab_run_experiments.py --task trustfed_rl_matrix --no-mount --download
```

**Why `--no-mount`?** Section 1 already mounts Drive. A second `drive.mount()` inside
`!python scripts/...` runs as a subprocess without an IPython kernel and fails with
`AttributeError: 'NoneType' object has no attribute 'kernel'`. Do not `sed`-patch the
runner; use `--no-mount` (the runner also auto-skips mount if `/content/drive` exists).

That runs all phases. After each seed it updates  
`My Drive/trustfed/iomt_metrics_latest.zip`.

If the session disconnects: **re-run the same cell** — finished seeds are restored from Drive and skipped.

### Phase ids (only if you ever need a subset)

`B0`, `B1`, `B3`, `B4`, `B5S`, `B5`, `B5star` (=B5†), `B4star` (=B4†)

---

## 3. When READY

```bash
!python scripts/check_trustfed_rl_iomt_status.py
!python run_experiments.py analyze
```

Expect **READY FOR PAPER TABLES** (12/12 each phase).

Copy Drive `iomt_metrics_latest.zip` / `trustfed_results_*.zip` back to the laptop if you want local plots:

`TrustFed-Agent/results/trustfed_agent/metrics/`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Still skipping old seeds on “scratch” run | Delete `My Drive/trustfed/iomt_metrics_latest.zip` and any old metric zips, then re-run |
| `colleague_seed_v1.json` missing | Re-run `pack_colab_handoff.sh`, re-upload **code** zip |
| IoMT data missing | Upload `iomt_data_bundle.zip` |
| NumPy / torch error | Re-run install cell; keep `requirements.txt` NumPy 1.x |
| Session died | Re-run full-matrix cell with `--no-mount`; resume is automatic |
| `AttributeError: ... 'kernel'` on mount | Drive already mounted in section 1 — add `--no-mount` (do not sed the script) |
| `Streaming output truncated to the last 5000 lines` | **Not an error** — Colab UI limit. Check Drive `iomt_metrics_latest.zip` / “metrics saved” lines |
| `Could not extract source filename from: hospital_*.csv` | **Not an error** on IoMT — leftover CICIDS clean-source probe; clients still set up OK |

---

## Time note

Full 8 phases × 12 seeds × ~1–1.5 h/seed is **multi-day** on Colab CPU. Prefer Colab Pro / keep-alive; always rely on Drive resume.
