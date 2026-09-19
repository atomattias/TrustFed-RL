#!/usr/bin/env python3
"""
Convert ECU-IoHT into constructed FL client CSVs for cross-dataset B1/B2 rematch.

Plan: docs/ECU_CROSS_DATASET_REPLICATION_PLAN.md (WP1)
Option C: docs/ECU_OPTION_C_PARTITION_ATTACKER_PLAN.md (WP-C1)

Important:
  - ECU has NO hospitals; clients are Source-IP silos (constructed).
  - Drop leaky columns from the *model* feature set (Protocol, Source, Destination, Info, No., raw Time).
  - Source is used only for partitioning, then discarded from exported CSVs.
  - Never writes into data/CSVs/iomt_natural/.

Usage:
  python scripts/convert_ecu_ioht_to_clients.py
  python scripts/convert_ecu_ioht_to_clients.py --source-select offset --source-offset 1 --seed 43 \\
      --out-dir data/CSVs/ecu_ioht_clients_c_p1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

MUST_DROP_MODEL = {"No.", "Protocol", "Source", "Destination", "Info", "Time"}


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer non-Protocol features; keep Source only for partitioning."""
    out = df.copy()
    out["label"] = (out["Type"].astype(str) == "Attack").astype(int)
    out["Length"] = pd.to_numeric(out["Length"], errors="coerce").fillna(0.0)
    out["log1p_length"] = np.log1p(out["Length"].clip(lower=0))
    # Source-local inter-arrival (partition key only later)
    out = out.sort_values(["Source", "Time"], kind="mergesort")
    out["delta_t"] = out.groupby("Source", sort=False)["Time"].diff().fillna(0.0)
    out["delta_t"] = out["delta_t"].clip(lower=0.0, upper=out["delta_t"].quantile(0.99))
    out["log1p_delta_t"] = np.log1p(out["delta_t"])
    out["uid"] = np.arange(len(out), dtype=np.int64)
    return out


def _select_sources(
    eligible: list,
    src_counts: pd.Series,
    n_clients: int,
    source_select: str,
    source_offset: int,
    seed: int,
) -> tuple[list, str]:
    """Return chosen Sources and a short selection note for the manifest."""
    mode = (source_select or "top").strip().lower()
    offset = max(0, int(source_offset))
    if mode == "top":
        chosen = list(eligible[:n_clients])
        note = f"top:{n_clients}"
        return chosen, note
    if mode == "offset":
        need = n_clients + offset
        if len(eligible) >= need:
            chosen = list(eligible[offset : offset + n_clients])
            note = f"offset:{offset}:eligible[{offset}:{offset + n_clients}]"
            return chosen, note
        # Fallback: sample n_clients from eligible
        mode = "sample"
    if mode == "sample":
        pool = (
            list(eligible)
            if len(eligible) >= n_clients
            else list(src_counts.head(max(n_clients, len(src_counts))).index)
        )
        if len(pool) < n_clients:
            raise SystemExit(f"Need ≥{n_clients} Sources; found {len(pool)}")
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(pool), size=n_clients, replace=False)
        chosen = [pool[int(i)] for i in sorted(idxs.tolist())]
        note = f"sample:n={n_clients}:seed={seed}"
        return chosen, note
    raise SystemExit(f"Unknown --source-select {source_select!r}; use top|offset|sample")


def convert_ecu_ioht_to_clients(
    source: Path,
    out_dir: Path,
    n_clients: int = 8,
    min_source_rows: int = 300,
    max_rows_per_client: int = 5000,
    seed: int = 42,
    source_select: str = "top",
    source_offset: int = 0,
) -> dict:
    df = pd.read_excel(source, engine="openpyxl")
    feat = _build_feature_frame(df)

    # Shared hold-outs BEFORE client assignment (plan §3)
    train_pool, temp = train_test_split(
        feat,
        test_size=0.30,
        random_state=seed,
        stratify=feat["label"],
    )
    val_ref, test_set = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp["label"],
    )

    model_cols = ["Length", "log1p_length", "delta_t", "log1p_delta_t", "label"]

    # Choose Sources in the *train* pool for constructed silos
    src_counts = train_pool["Source"].value_counts()
    eligible = src_counts[src_counts >= min_source_rows].index.tolist()
    if len(eligible) < n_clients:
        eligible = src_counts.head(n_clients).index.tolist()
    chosen, select_note = _select_sources(
        eligible, src_counts, n_clients, source_select, source_offset, seed
    )
    if len(chosen) < n_clients:
        raise SystemExit(f"Selected only {len(chosen)} Sources; need {n_clients}")

    clients_dir = out_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    client_manifest = {}
    assigned = 0
    for i, src in enumerate(chosen, start=1):
        part = train_pool[train_pool["Source"] == src][model_cols].copy()
        if max_rows_per_client > 0 and len(part) > max_rows_per_client:
            if part["label"].nunique() > 1:
                part, _ = train_test_split(
                    part,
                    train_size=max_rows_per_client,
                    random_state=seed + i,
                    stratify=part["label"],
                )
            else:
                part = part.sample(n=max_rows_per_client, random_state=seed + i)
        if float(part["Length"].std() or 0.0) < 1e-12:
            rng = np.random.default_rng(seed + i)
            part = part.copy()
            part["Length"] = part["Length"].astype(float) + rng.normal(0.0, 1e-3, len(part))
            part["log1p_length"] = np.log1p(part["Length"].clip(lower=0))
        path = clients_dir / f"client_{i:02d}.csv"
        part.to_csv(path, index=False)
        client_manifest[f"client_{i:02d}"] = {
            "partition_source": str(src),
            "n_train_rows": int(len(part)),
            "attack_rate": float(part["label"].mean()) if len(part) else 0.0,
            "note": "constructed Source silo; not a hospital",
        }
        assigned += len(part)

    val_path = out_dir / "ecu_val_ref.csv"
    test_path = out_dir / "ecu_test_set.csv"
    val_ref[model_cols].to_csv(val_path, index=False)
    test_set[model_cols].to_csv(test_path, index=False)
    val_ref[model_cols].to_csv(out_dir / "iomt_val_ref.csv", index=False)
    test_set[model_cols].to_csv(out_dir / "iomt_test_set.csv", index=False)

    unused_train = int(len(train_pool) - assigned)
    manifest = {
        "schema_version": "ecu_ioht_clients_v2_option_c",
        "plan": "docs/ECU_OPTION_C_PARTITION_ATTACKER_PLAN.md WP-C1",
        "source_file": str(source),
        "claim": "constructed FL clients from ECU-IoHT Source silos; NOT multi-hospital",
        "dropped_from_model": sorted(MUST_DROP_MODEL),
        "model_features": [c for c in model_cols if c != "label"],
        "n_clients": len(chosen),
        "min_source_rows": min_source_rows,
        "max_rows_per_client": max_rows_per_client,
        "source_select": source_select,
        "source_offset": int(source_offset),
        "source_select_note": select_note,
        "eligible_sources_n": int(len(eligible)),
        "split": {
            "train_pool": int(len(train_pool)),
            "assigned_to_clients": assigned,
            "unused_train_other_sources": unused_train,
            "val_ref": int(len(val_ref)),
            "test": int(len(test_set)),
            "train_utilization_of_train_pool": assigned / max(len(train_pool), 1),
        },
        "attack_rates": {
            "train_assigned": float(
                pd.concat(
                    [
                        pd.read_csv(clients_dir / f"client_{i:02d}.csv")
                        for i in range(1, len(chosen) + 1)
                    ]
                )["label"].mean()
            ),
            "val_ref": float(val_ref["label"].mean()),
            "test": float(test_set["label"].mean()),
        },
        "clients": client_manifest,
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "ECU_IoHT.xlsx")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "CSVs" / "ecu_ioht_clients")
    ap.add_argument("--n-clients", type=int, default=8)
    ap.add_argument("--min-source-rows", type=int, default=300)
    ap.add_argument("--max-rows-per-client", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--source-select",
        choices=["top", "offset", "sample"],
        default="top",
        help="How to pick constructed Source silos (default top = legacy WP1)",
    )
    ap.add_argument(
        "--source-offset",
        type=int,
        default=0,
        help="For --source-select offset: take eligible[K:K+n_clients]",
    )
    args = ap.parse_args()

    if "iomt_natural" in str(args.out_dir):
        raise SystemExit("Refusing to write into iomt_natural path")

    man = convert_ecu_ioht_to_clients(
        args.source,
        args.out_dir,
        args.n_clients,
        args.min_source_rows,
        args.max_rows_per_client,
        args.seed,
        args.source_select,
        args.source_offset,
    )
    print(
        json.dumps(
            {
                k: man[k]
                for k in (
                    "n_clients",
                    "split",
                    "claim",
                    "model_features",
                    "source_select",
                    "source_offset",
                    "source_select_note",
                )
            },
            indent=2,
        )
    )
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
