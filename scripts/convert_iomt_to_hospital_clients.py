#!/usr/bin/env python3
"""
Convert CICIoMT2024 CSV features into federated hospital client CSVs for TrustFed-Agent.

Usage:
  python scripts/convert_iomt_to_hospital_clients.py --source-dir /path/to/CICIoMT2024/.../csv/train
  python scripts/convert_iomt_to_hospital_clients.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BENIGN_TOKENS = {
    "benign", "normal", "legitimate", "legit", "0", "no_attack", "background",
}
QUALITY_TIERS = [
    ("high", 0.0, 0.0),
    ("high", 0.0, 0.0),
    ("high", 0.0, 0.0),
    ("medium", 0.05, 0.02),
    ("medium", 0.05, 0.02),
    ("medium", 0.10, 0.05),
    ("low", 0.15, 0.10),
    ("low", 0.15, 0.10),
    ("compromised", 0.99, 0.95),
    ("compromised", 0.99, 0.95),
    ("high", 0.0, 0.0),
    ("medium", 0.05, 0.02),
]


def _find_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in ("label", "labels", "class", "attack", "category"):
            return col
    raise ValueError(f"No label column found in columns: {list(df.columns)[:20]}")


def _to_binary_label(value) -> int:
    text = str(value).strip().lower()
    if text in BENIGN_TOKENS:
        return 0
    try:
        num = float(text)
        if num in (0.0, 1.0):
            return int(num)
    except ValueError:
        pass
    return 0 if text in BENIGN_TOKENS else 1


def _label_from_filename(path: Path) -> int:
    name = path.stem.lower()
    if "benign" in name:
        return 0
    return 1


def _infer_protocol(stem: str) -> str:
    s = stem.lower()
    if s.startswith("mqtt"):
        return "mqtt"
    if s.startswith("recon"):
        return "recon"
    if s.startswith("arp"):
        return "arp"
    if s.startswith("tcp_ip") or s.startswith("tcp"):
        return "tcp_ip"
    if "bluetooth" in s or "ble" in s:
        return "bluetooth"
    if "benign" in s:
        return "benign"
    return "other"


def _normalize_dataframe(df: pd.DataFrame, source_path: Optional[Path] = None) -> pd.DataFrame:
    df = df.copy()
    label_col = None
    for col in df.columns:
        if col.lower() in ("label", "labels", "class", "attack", "category"):
            label_col = col
            break

    if label_col is not None:
        df["label"] = df[label_col].map(_to_binary_label).astype(int)
    elif source_path is not None:
        df["label"] = _label_from_filename(source_path)
    else:
        raise ValueError("No label column and no source path for filename-based labelling")

    drop_cols = [c for c in df.columns if c != "label" and c.lower() in (
        "label", "labels", "class", "attack", "category", "flow_id", "timestamp",
    )]
    df = df.drop(columns=drop_cols, errors="ignore")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" not in numeric:
        numeric.append("label")
    df = df[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _load_csv_dir(
    source_dir: Path,
    max_rows_per_file: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        files = sorted(source_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {source_dir}")
    rng = np.random.default_rng(seed)
    frames = []
    for path in files:
        try:
            if max_rows_per_file is not None:
                df = pd.read_csv(path, nrows=max_rows_per_file * 3)
                if len(df) > max_rows_per_file:
                    df = df.sample(n=max_rows_per_file, random_state=seed)
            else:
                df = pd.read_csv(path)
            if df.empty:
                continue
            df = _normalize_dataframe(df, source_path=path)
            df["_source_file"] = path.stem
            df["_source_path"] = str(path)
            df["_protocol"] = _infer_protocol(path.stem)
            frames.append(df)
            print(f"  loaded {path.name}: {len(df):,} rows, label={df['label'].iloc[0]}")
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
    if not frames:
        raise RuntimeError(f"Could not load any valid CSV from {source_dir}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(frames)} CSV files")
    return combined


def _partition_key(row: pd.Series, scheme: str) -> str:
    if scheme == "protocol" and "_protocol" in row.index:
        return str(row["_protocol"])
    path = str(row.get("_source_path", "")).lower()
    stem = str(row.get("_source_file", "")).lower()
    if scheme == "protocol":
        return _infer_protocol(stem)
    if scheme == "attack":
        return stem or "unknown"
    digest = hashlib.md5(f"{row.name}".encode()).hexdigest()
    return digest[:8]


def _assign_hospital_ids(df: pd.DataFrame, num_hospitals: int, scheme: str, seed: int = 42) -> pd.Series:
    keys = df.apply(lambda r: _partition_key(r, scheme), axis=1)
    rng = np.random.default_rng(seed)
    hospitals = pd.Series(index=df.index, dtype=str)
    unique_keys = sorted(keys.unique())

    if scheme == "hash" or len(unique_keys) < num_hospitals:
        perm = rng.permutation(len(df))
        for i, idx in enumerate(df.index):
            hospitals.loc[idx] = f"hospital_{(perm[i] % num_hospitals) + 1:02d}"
        return hospitals

    groups = np.array_split(unique_keys, num_hospitals)
    mapping = {}
    for i, group in enumerate(groups):
        hid = f"hospital_{i + 1:02d}"
        for k in group:
            mapping[k] = hid
    hospitals = keys.map(mapping)
    return hospitals


def _stratified_sample(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Cap rows per class without losing the label column (groupby.apply drops it)."""
    if len(df) <= max_rows:
        return df.copy()
    parts = []
    for label_val, group in df.groupby("label", sort=False):
        n = max(1, int(max_rows * len(group) / len(df)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _flip_labels(df: pd.DataFrame, noise_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n_flip = int(len(df) * noise_ratio)
    if n_flip <= 0:
        return df
    idx = rng.choice(len(df), size=n_flip, replace=False)
    df.iloc[idx, df.columns.get_loc("label")] = 1 - df.iloc[idx]["label"]
    return df


def _corrupt_features(df: pd.DataFrame, ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    feat_cols = [c for c in df.columns if c != "label"]
    if not feat_cols:
        return df
    n_rows = int(len(df) * ratio)
    if n_rows <= 0:
        return df
    rows = rng.choice(len(df), size=n_rows, replace=False)
    for i in rows:
        col = rng.choice(feat_cols)
        std = float(df[col].std()) or 1.0
        df.iat[i, df.columns.get_loc(col)] += rng.normal(0, std * 2)
    return df


def _apply_quality_tier(df: pd.DataFrame, tier: str, label_noise: float, feature_noise: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if label_noise > 0:
        out = _flip_labels(out, label_noise, rng)
    if feature_noise > 0:
        out = _corrupt_features(out, feature_noise, rng)
    return out


def _generate_demo_data(n_samples: int = 25000, n_features: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_attack = int(n_samples * 0.12)
    rows = []
    protocols = ["wifi", "mqtt", "bluetooth"]
    attacks = ["ddos", "dos", "recon", "mqtt_flood", "spoofing", "benign"]
    for i in range(n_samples):
        protocol = protocols[i % len(protocols)]
        attack = attacks[rng.integers(0, len(attacks))]
        is_attack = 1 if (i < n_attack or attack != "benign") and rng.random() < 0.15 else 0
        if attack == "benign":
            is_attack = 0
        row = {f"feat_{j}": float(rng.normal(is_attack * 2, 1.0)) for j in range(n_features)}
        row["label"] = is_attack
        row["_source_file"] = f"{protocol}_{attack}"
        row["_source_path"] = f"/demo/{protocol}/{attack}.csv"
        rows.append(row)
    return pd.DataFrame(rows)


def convert_iomt_to_hospitals(
    source_dir: Optional[Path],
    output_dir: Path,
    test_output: Path,
    test_dir: Optional[Path] = None,
    num_hospitals: int = 12,
    partition: str = "protocol",
    test_size: float = 0.2,
    max_rows_per_file: Optional[int] = 5000,
    max_test_rows: Optional[int] = 50000,
    max_rows_per_hospital: Optional[int] = 25000,
    seed: int = 42,
    demo: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    if demo:
        print("Generating synthetic IoMT demo data...")
        combined = _generate_demo_data(seed=seed)
        test_df = None
    else:
        if source_dir is None or not source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found: {source_dir}. Use --demo or set --source-dir."
            )
        print(f"Loading train from {source_dir} ...")
        combined = _load_csv_dir(source_dir, max_rows_per_file=max_rows_per_file, seed=seed)
        test_df = None
        if test_dir is not None and test_dir.exists():
            print(f"Loading official test from {test_dir} ...")
            test_df = _load_csv_dir(test_dir, max_rows_per_file=max_rows_per_file, seed=seed)
            if max_test_rows and len(test_df) > max_test_rows:
                test_df = _stratified_sample(test_df, max_test_rows, seed=seed)

    meta_cols = ["_source_file", "_source_path", "_protocol"]
    if test_df is None:
        stratify = combined["label"] if combined["label"].nunique() > 1 else None
        train_df, test_df = train_test_split(
            combined, test_size=test_size, random_state=seed, stratify=stratify,
        )
    else:
        train_df = combined

    train_df = train_df.copy()
    train_df["hospital_id"] = _assign_hospital_ids(train_df, num_hospitals, partition, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    hospital_ids = sorted(train_df["hospital_id"].unique())[:num_hospitals]
    tier_cycle = (QUALITY_TIERS * ((len(hospital_ids) // len(QUALITY_TIERS)) + 1))[:len(hospital_ids)]

    for hid, (tier, label_noise, feat_noise) in zip(hospital_ids, tier_cycle):
        chunk = train_df[train_df["hospital_id"] == hid].drop(columns=meta_cols + ["hospital_id"])
        if max_rows_per_hospital and len(chunk) > max_rows_per_hospital:
            chunk = _stratified_sample(chunk, max_rows_per_hospital, seed=seed)
        chunk = _apply_quality_tier(chunk, tier, label_noise, feat_noise, rng)
        protocol = hid.replace("hospital_", "")
        fname = f"{hid}_{tier}_quality_iomt.csv"
        if tier == "compromised":
            fname = f"{hid}_compromised_iomt.csv"
        out_path = output_dir / fname
        chunk.to_csv(out_path, index=False)
        print(f"  wrote {out_path.name}: {len(chunk):,} rows, tier={tier}, attacks={chunk['label'].sum():,}")

    test_clean = test_df.drop(columns=meta_cols + (["hospital_id"] if "hospital_id" in test_df.columns else []))
    test_clean.to_csv(test_output, index=False)
    print(f"\nTest set: {test_output} ({len(test_clean):,} rows)")
    print(f"Clients: {output_dir} ({len(list(output_dir.glob('*.csv')))} files)")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Convert CICIoMT2024 to federated hospital CSV clients")
    parser.add_argument("--source-dir", type=Path, default=None, help="CICIoMT2024 CSV train directory")
    parser.add_argument("--test-dir", type=Path, default=None, help="CICIoMT2024 official CSV test directory")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "CSVs" / "iomt_clients")
    parser.add_argument("--test-output", type=Path, default=root / "data" / "CSVs" / "iomt_test_set.csv")
    parser.add_argument("--num-hospitals", type=int, default=12)
    parser.add_argument("--partition", choices=["protocol", "attack", "hash"], default="protocol")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows-per-file", type=int, default=5000,
                        help="Sample cap per source CSV (full dataset is ~7M rows)")
    parser.add_argument("--max-test-rows", type=int, default=50000)
    parser.add_argument("--max-rows-per-hospital", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data (no CICIoMT download)")
    args = parser.parse_args()

    convert_iomt_to_hospitals(
        source_dir=args.source_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir.resolve(),
        test_output=args.test_output.resolve(),
        num_hospitals=args.num_hospitals,
        partition=args.partition,
        test_size=args.test_size,
        max_rows_per_file=args.max_rows_per_file,
        max_test_rows=args.max_test_rows,
        max_rows_per_hospital=args.max_rows_per_hospital,
        seed=args.seed,
        demo=args.demo,
    )


if __name__ == "__main__":
    main()
