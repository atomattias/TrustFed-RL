#!/usr/bin/env python3
"""
Convert CICIoMT2024 train CSVs into natural FL clients with a strict 70/15/15 split.

Plan: docs/NATURAL_PARTITION_IMPLEMENTATION_PLAN.md (v2.0 WP1)

  raw train → stratified 70% train / 15% val_ref / 15% test  (BEFORE clients)
  train only → 12 protocol/attack-family clients (no quality / compromised tiers)

Usage:
  python scripts/convert_iomt_natural_clients.py
  python scripts/convert_iomt_natural_clients.py --source-dir data/train --max-rows-per-file 8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FAMILIES = ("benign", "mqtt", "tcp_ip", "recon", "arp")

# Target composition per client (rows sum to 1). Observable non-IID mix — not quality tiers.
CLIENT_MIX: Dict[int, Dict[str, float]] = {
    1: {"benign": 0.15, "mqtt": 0.45, "tcp_ip": 0.25, "recon": 0.10, "arp": 0.05},
    2: {"benign": 0.15, "mqtt": 0.45, "tcp_ip": 0.25, "recon": 0.10, "arp": 0.05},
    3: {"benign": 0.15, "mqtt": 0.45, "tcp_ip": 0.25, "recon": 0.10, "arp": 0.05},
    4: {"benign": 0.15, "mqtt": 0.10, "tcp_ip": 0.50, "recon": 0.15, "arp": 0.10},
    5: {"benign": 0.15, "mqtt": 0.10, "tcp_ip": 0.50, "recon": 0.15, "arp": 0.10},
    6: {"benign": 0.15, "mqtt": 0.10, "tcp_ip": 0.50, "recon": 0.15, "arp": 0.10},
    7: {"benign": 0.20, "mqtt": 0.10, "tcp_ip": 0.25, "recon": 0.40, "arp": 0.05},
    8: {"benign": 0.20, "mqtt": 0.10, "tcp_ip": 0.25, "recon": 0.40, "arp": 0.05},
    9: {"benign": 0.25, "mqtt": 0.15, "tcp_ip": 0.25, "recon": 0.15, "arp": 0.20},
    10: {"benign": 0.45, "mqtt": 0.15, "tcp_ip": 0.25, "recon": 0.10, "arp": 0.05},
    11: {"benign": 0.45, "mqtt": 0.15, "tcp_ip": 0.25, "recon": 0.10, "arp": 0.05},
    12: {"benign": 0.45, "mqtt": 0.15, "tcp_ip": 0.25, "recon": 0.10, "arp": 0.05},
}

# Single-file families need higher caps so CLIENT_MIX stays realizable (Benign≈193k, ARP≈16k).
DEFAULT_FAMILY_CAPS = {
    "benign": 50000,
    "arp": 16000,
}


def _infer_family(stem: str) -> str:
    s = stem.lower()
    if "benign" in s:
        return "benign"
    if s.startswith("mqtt") or "mqtt" in s:
        return "mqtt"
    if s.startswith("recon") or "recon" in s:
        return "recon"
    if s.startswith("arp") or "arp" in s:
        return "arp"
    if s.startswith("tcp_ip") or s.startswith("tcp") or "ddos" in s or "dos" in s:
        return "tcp_ip"
    return "other"


def _label_from_filename(path: Path) -> int:
    return 0 if "benign" in path.stem.lower() else 1


def _normalize_frame(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    df = df.copy()
    label_col = None
    for col in df.columns:
        if col.lower() in ("label", "labels", "class", "attack", "category"):
            label_col = col
            break
    if label_col is not None:
        def _map(v) -> int:
            t = str(v).strip().lower()
            if t in {"benign", "normal", "0", "legitimate"}:
                return 0
            try:
                x = float(t)
                if x in (0.0, 1.0):
                    return int(x)
            except ValueError:
                pass
            return 1

        df["label"] = df[label_col].map(_map).astype(int)
        drop = [c for c in df.columns if c != "label" and c.lower() in (
            "label", "labels", "class", "attack", "category", "flow_id", "timestamp",
        )]
        df = df.drop(columns=drop, errors="ignore")
    else:
        df["label"] = _label_from_filename(source_path)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" not in numeric:
        numeric.append("label")
    df = df[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["label"] = df["label"].astype(int)
    return df


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_pool(
    source_dir: Path,
    max_rows_per_file: Optional[int],
    seed: int,
    family_row_caps: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        files = sorted(source_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {source_dir}")

    family_row_caps = family_row_caps or {}
    frames: List[pd.DataFrame] = []
    for path in files:
        family = _infer_family(path.stem)
        if family == "other":
            print(f"  skip (unknown family): {path.name}")
            continue
        cap = family_row_caps.get(family, max_rows_per_file)
        try:
            if cap is not None:
                df = pd.read_csv(path, nrows=int(cap) * 3)
                if len(df) > cap:
                    df = df.sample(n=int(cap), random_state=seed)
            else:
                df = pd.read_csv(path)
            if df.empty:
                continue
            df = _normalize_frame(df, path)
            df["_family"] = family
            df["_source_file"] = path.name
            frames.append(df)
            print(
                f"  loaded {path.name}: {len(df):,} rows family={family} "
                f"attack_rate={df['label'].mean():.3f}"
            )
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")

    if not frames:
        raise RuntimeError(f"No usable CSVs loaded from {source_dir}")

    feat_cols: List[str] = []
    seen = set()
    for fr in frames:
        for c in fr.columns:
            if c in ("label", "_family", "_source_file"):
                continue
            if c not in seen:
                seen.add(c)
                feat_cols.append(c)

    aligned = []
    for fr in frames:
        out = fr.reindex(columns=feat_cols + ["label", "_family", "_source_file"], fill_value=0.0)
        out[feat_cols] = out[feat_cols].astype(np.float64)
        out["label"] = out["label"].astype(int)
        aligned.append(out)

    pooled = pd.concat(aligned, ignore_index=True)
    pooled.insert(0, "_uid", np.arange(len(pooled), dtype=np.int64))
    pooled = pooled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"Pooled {len(pooled):,} rows, {len(feat_cols)} features, "
        f"families={pooled['_family'].value_counts().to_dict()}"
    )
    return pooled


def stratified_70_15_15(
    df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-family 70/15/15, stratify by label when both classes exist."""
    train_parts, val_parts, test_parts = [], [], []
    for family, g in df.groupby("_family", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 10:
            train_parts.append(g)
            print(f"  warn: family {family} n={len(g)} — all to train")
            continue
        strat = g["label"] if g["label"].nunique() > 1 else None
        train_g, temp = train_test_split(
            g, test_size=0.30, random_state=seed, stratify=strat,
        )
        strat_t = temp["label"] if temp["label"].nunique() > 1 else None
        if len(temp) < 4:
            val_g, test_g = temp.iloc[: len(temp) // 2], temp.iloc[len(temp) // 2 :]
        else:
            val_g, test_g = train_test_split(
                temp, test_size=0.50, random_state=seed, stratify=strat_t,
            )
        train_parts.append(train_g)
        val_parts.append(val_g)
        test_parts.append(test_g)
        print(
            f"  family {family}: train={len(train_g):,} val={len(val_g):,} test={len(test_g):,} "
            f"(attack_rate train={train_g['label'].mean():.3f})"
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else train_df.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else train_df.iloc[0:0]
    return train_df, val_df, test_df


def mix_faithful_client_target(
    train_df: pd.DataFrame,
    num_clients: int,
    max_rows_per_client: Optional[int],
) -> int:
    """Largest equal client size whose mix demands fit every family supply."""
    supplies = {f: int((train_df["_family"] == f).sum()) for f in FAMILIES}
    limits = []
    for f in FAMILIES:
        w = sum(CLIENT_MIX[k][f] for k in range(1, num_clients + 1))
        if w <= 0:
            continue
        limits.append(supplies[f] / w)
    if not limits:
        raise RuntimeError("No family supplies for mix-faithful sizing")
    target = int(np.floor(min(limits)))
    if max_rows_per_client is not None:
        target = min(target, max_rows_per_client)
    target = max(target, 1)
    print(
        f"  mix-faithful client_target={target} "
        f"(binding={min(limits):.1f}; supplies={supplies})"
    )
    return target


def assign_clients(
    train_df: pd.DataFrame,
    num_clients: int,
    max_rows_per_client: Optional[int],
    seed: int,
) -> Dict[int, pd.DataFrame]:
    """Allocate train rows by CLIENT_MIX with mix-faithful sizing (no leftover dump)."""
    if num_clients != 12:
        raise ValueError("v1 mix table is defined for exactly 12 clients")

    client_target = mix_faithful_client_target(train_df, num_clients, max_rows_per_client)

    need: Dict[int, Dict[str, int]] = {}
    for k in range(1, num_clients + 1):
        mix = CLIENT_MIX[k]
        need[k] = {f: int(round(client_target * mix[f])) for f in FAMILIES}

    pools = {
        f: train_df[train_df["_family"] == f].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for f in FAMILIES
    }
    for f in FAMILIES:
        total_need = sum(need[k][f] for k in need)
        supply = len(pools[f])
        if total_need > supply:
            scale = supply / max(total_need, 1)
            for k in need:
                need[k][f] = int(need[k][f] * scale)
            used = sum(need[k][f] for k in need)
            rem = supply - used
            k_cycle = 1
            while rem > 0:
                need[k_cycle][f] += 1
                rem -= 1
                k_cycle = k_cycle % num_clients + 1

    cursors = {f: 0 for f in FAMILIES}
    clients: Dict[int, pd.DataFrame] = {}
    for k in range(1, num_clients + 1):
        parts = []
        for f in FAMILIES:
            n = need[k][f]
            if n <= 0:
                continue
            start = cursors[f]
            end = start + n
            parts.append(pools[f].iloc[start:end])
            cursors[f] = end
        if not parts:
            raise RuntimeError(f"Client {k} received zero rows — check mix/supply")
        cdf = pd.concat(parts, ignore_index=True)
        cdf = cdf.sample(frac=1.0, random_state=seed + k).reset_index(drop=True)
        clients[k] = cdf
        fam_counts = cdf["_family"].value_counts().to_dict()
        realized = {f: fam_counts.get(f, 0) / len(cdf) for f in FAMILIES}
        max_dev = max(abs(realized[f] - CLIENT_MIX[k][f]) for f in FAMILIES)
        print(
            f"  client_{k:02d}: n={len(cdf):,} attack_rate={cdf['label'].mean():.3f} "
            f"max|Δmix|={max_dev:.3f} "
            f"realized={{{', '.join(f'{f}={realized[f]:.2f}' for f in FAMILIES)}}}"
        )
        if max_dev > 0.05:
            raise RuntimeError(
                f"client_{k:02d} mix deviation {max_dev:.3f} > 0.05 — check caps/sizing"
            )
    unused = {f: len(pools[f]) - cursors[f] for f in FAMILIES}
    print(f"  unused train by family (intentional; preserves mix): {unused}")
    return clients


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ("_family", "_source_file", "_uid") if c in df.columns]
    out = df.drop(columns=drop)
    cols = [c for c in out.columns if c != "label"] + ["label"]
    return out[cols]


def convert(
    source_dir: Path,
    output_dir: Path,
    seed: int = 42,
    num_clients: int = 12,
    max_rows_per_file: Optional[int] = 8000,
    max_rows_per_client: Optional[int] = 25000,
    family_row_caps: Optional[Dict[str, int]] = None,
) -> None:
    family_row_caps = family_row_caps if family_row_caps is not None else dict(DEFAULT_FAMILY_CAPS)
    print(f"Loading pool from {source_dir} ...")
    print(f"  family_row_caps={family_row_caps} default_per_file={max_rows_per_file}")
    pooled = load_train_pool(
        source_dir,
        max_rows_per_file=max_rows_per_file,
        seed=seed,
        family_row_caps=family_row_caps,
    )

    print("Splitting 70/15/15 per family (before client construction) ...")
    train_df, val_df, test_df = stratified_70_15_15(pooled, seed=seed)
    n = len(pooled)
    print(
        f"Split sizes: train={len(train_df):,} ({len(train_df)/n:.1%}) "
        f"val={len(val_df):,} ({len(val_df)/n:.1%}) test={len(test_df):,} ({len(test_df)/n:.1%})"
    )

    print("Assigning natural clients ...")
    clients = assign_clients(
        train_df, num_clients=num_clients, max_rows_per_client=max_rows_per_client, seed=seed,
    )

    clients_dir = output_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_out = _feature_frame(val_df)
    test_out = _feature_frame(test_df)
    feature_cols = [c for c in val_out.columns if c != "label"]

    val_path = output_dir / "iomt_val_ref.csv"
    test_path = output_dir / "iomt_test_set.csv"
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    client_paths = {}
    for k, cdf in clients.items():
        frame = _feature_frame(cdf)
        frame = frame.reindex(columns=feature_cols + ["label"], fill_value=0.0)
        frame["label"] = frame["label"].astype(int)
        path = clients_dir / f"client_{k:02d}.csv"
        frame.to_csv(path, index=False)
        client_paths[k] = str(path.relative_to(output_dir))

    assigned = sum(len(c) for c in clients.values())
    print(
        f"Assigned {assigned:,} / {len(train_df):,} train rows to clients "
        f"({assigned / max(len(train_df), 1):.1%})"
    )

    print("Checking train/val/test uid disjointness ...")
    train_uids = set(train_df["_uid"].tolist())
    val_uids = set(val_df["_uid"].tolist())
    test_uids = set(test_df["_uid"].tolist())
    client_uids = set()
    for k, cdf in clients.items():
        hs = set(cdf["_uid"].tolist())
        if client_uids & hs:
            raise RuntimeError(f"Duplicate uid across clients involving client_{k:02d}")
        client_uids |= hs
    if not client_uids.issubset(train_uids):
        raise RuntimeError("Client rows include uids outside the train partition")
    if train_uids & val_uids or train_uids & test_uids or val_uids & test_uids:
        raise RuntimeError(
            f"Split leakage: train∩val={len(train_uids & val_uids)}, "
            f"train∩test={len(train_uids & test_uids)}, "
            f"val∩test={len(val_uids & test_uids)}"
        )
    print("  OK: train/val/test partitions are uid-disjoint; clients ⊆ train.")

    realized_mix = {}
    for k, cdf in clients.items():
        rates = cdf["_family"].value_counts(normalize=True).to_dict()
        realized_mix[f"client_{k:02d}"] = {f: float(rates.get(f, 0.0)) for f in FAMILIES}

    manifest = {
        "version": "iomt_natural_v1",
        "seed": seed,
        "source_dir": str(source_dir),
        "split": {"train": 0.70, "val_ref": 0.15, "test": 0.15},
        "max_rows_per_file": max_rows_per_file,
        "family_row_caps": family_row_caps,
        "max_rows_per_client": max_rows_per_client,
        "num_clients": num_clients,
        "families": list(FAMILIES),
        "client_mix_target": {str(k): CLIENT_MIX[k] for k in CLIENT_MIX},
        "client_mix_realized": realized_mix,
        "counts": {
            "pooled": int(len(pooled)),
            "train": int(len(train_df)),
            "val_ref": int(len(val_df)),
            "test": int(len(test_df)),
            "assigned_to_clients": int(assigned),
            "train_utilization": float(assigned / max(len(train_df), 1)),
            "per_client": {f"client_{k:02d}": int(len(clients[k])) for k in clients},
        },
        "family_rates": {
            "pooled": pooled["_family"].value_counts(normalize=True).to_dict(),
            "val_ref": val_df["_family"].value_counts(normalize=True).to_dict(),
            "test": test_df["_family"].value_counts(normalize=True).to_dict(),
        },
        "attack_rates": {
            "train": float(train_df["label"].mean()),
            "val_ref": float(val_df["label"].mean()),
            "test": float(test_df["label"].mean()),
        },
        "feature_columns": feature_cols,
        "paths": {
            "val_ref": "iomt_val_ref.csv",
            "test": "iomt_test_set.csv",
            "clients": client_paths,
        },
        "note": (
            "Attacker IDs / t_a live in runtime config, not in this manifest. "
            "Unused train rows (mostly tcp_ip surplus) are intentional to preserve CLIENT_MIX."
        ),
    }

    manifest["sha256"] = {
        "iomt_val_ref.csv": _file_sha256(val_path),
        "iomt_test_set.csv": _file_sha256(test_path),
        **{f"client_{k:02d}.csv": _file_sha256(clients_dir / f"client_{k:02d}.csv") for k in clients},
    }

    man_path = output_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nWrote {val_path}")
    print(f"Wrote {test_path}")
    print(f"Wrote {clients_dir} ({num_clients} clients)")
    print(f"Wrote {man_path}")

    assert "quality" not in str(client_paths).lower()
    assert "compromised" not in str(client_paths).lower()
    for k in clients:
        ccols = list(pd.read_csv(clients_dir / f"client_{k:02d}.csv", nrows=0).columns)
        assert ccols == feature_cols + ["label"], f"schema mismatch client {k}"
    vcols = list(pd.read_csv(val_path, nrows=0).columns)
    tcols = list(pd.read_csv(test_path, nrows=0).columns)
    assert vcols == tcols == feature_cols + ["label"]
    print("Acceptance: schema aligned; no quality/compromised names; mix-faithful.")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="CICIoMT → natural FL clients (70/15/15)")
    p.add_argument("--source-dir", type=Path, default=root / "data" / "train")
    p.add_argument("--output-dir", type=Path, default=root / "data" / "CSVs" / "iomt_natural")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=12)
    p.add_argument("--max-rows-per-file", type=int, default=8000)
    p.add_argument("--max-rows-per-client", type=int, default=25000)
    p.add_argument("--max-rows-benign", type=int, default=DEFAULT_FAMILY_CAPS["benign"])
    p.add_argument("--max-rows-arp", type=int, default=DEFAULT_FAMILY_CAPS["arp"])
    p.add_argument("--no-cap", action="store_true", help="Disable per-file row cap")
    args = p.parse_args()

    family_caps = None if args.no_cap else {
        "benign": args.max_rows_benign,
        "arp": args.max_rows_arp,
    }
    convert(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        num_clients=args.num_clients,
        max_rows_per_file=None if args.no_cap else args.max_rows_per_file,
        max_rows_per_client=args.max_rows_per_client,
        family_row_caps=family_caps,
    )


if __name__ == "__main__":
    main()
