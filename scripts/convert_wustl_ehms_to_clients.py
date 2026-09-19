#!/usr/bin/env python3
"""
Convert WUSTL-EHMS-2020 (with attack categories) into federated client CSVs for TrustFed-RL (P4).

Safe feature policy (mandatory):
  DROP: SrcAddr, DstAddr, SrcMac, DstMac, Sport, Packet_num  (identity / perfect attack separator)
  DROP from training label space: Attack Category (keep only for stratification)
  KEEP: flow numerics + Flgs encoding + vitals (Temp, SpO2, ...)

Usage:
  python scripts/convert_wustl_ehms_to_clients.py
  python scripts/convert_wustl_ehms_to_clients.py --source data/wustl-ehms-2020_with_attacks_categories.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

LEAK_COLS = {
    "SrcAddr",
    "DstAddr",
    "SrcMac",
    "DstMac",
    "Sport",
    "Packet_num",
    "Dir",  # constant in this release
}


def _stratified_sample(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    parts = []
    for _, group in df.groupby("label", sort=False):
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
    if not feat_cols or ratio <= 0:
        return df
    n_rows = int(len(df) * ratio)
    if n_rows <= 0:
        return df
    # Vectorised: pick rows and add noise to a random column per selected row
    rows = rng.choice(len(df), size=n_rows, replace=False)
    cols = rng.choice(len(feat_cols), size=n_rows, replace=True)
    vals = df[feat_cols].to_numpy(dtype=np.float64, copy=True)
    stds = np.std(vals, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    vals[rows, cols] += rng.normal(0.0, stds[cols] * 2.0)
    out = df.copy()
    out[feat_cols] = vals
    return out


def _apply_quality_tier(
    df: pd.DataFrame,
    tier: str,
    label_noise: float,
    feat_noise: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = _flip_labels(df, label_noise, rng)
    df = _corrupt_features(df, feat_noise, rng)
    return df


def _build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if "Label" not in raw.columns:
        raise ValueError("WUSTL-EHMS CSV must contain a binary 'Label' column")
    df = raw.copy()
    df["label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    attack_cat = (
        df["Attack Category"].astype(str)
        if "Attack Category" in df.columns
        else pd.Series(["unknown"] * len(df), index=df.index)
    )

    drop = [c for c in df.columns if c in LEAK_COLS or c in ("Label", "Attack Category")]
    df = df.drop(columns=drop, errors="ignore")

    # Encode sparse categoricals (Flgs); Dport is numeric but nearly constant — keep as number
    if "Flgs" in df.columns:
        dummies = pd.get_dummies(df["Flgs"].astype(str).str.strip(), prefix="Flgs")
        df = pd.concat([df.drop(columns=["Flgs"]), dummies], axis=1)

    # Force numeric feature matrix
    feat_cols = [c for c in df.columns if c != "label"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    df["_attack_cat"] = attack_cat.values
    return df


def convert_wustl_ehms_to_clients(
    source: Path,
    output_dir: Path,
    test_output: Path,
    num_clients: int = 12,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"WUSTL-EHMS source not found: {source}")

    print(f"Loading {source} ...")
    raw = pd.read_csv(source)
    print(f"  raw shape={raw.shape}")
    combined = _build_feature_frame(raw)
    n_feat = len([c for c in combined.columns if c not in ("label", "_attack_cat")])
    print(
        f"  after safe drops: {len(combined):,} rows, {n_feat} features, "
        f"attack_rate={combined['label'].mean():.3f}"
    )

    stratify = combined["label"] if combined["label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        combined, test_size=test_size, random_state=seed, stratify=stratify
    )

    # Assign clients: stratify by label (+ attack cat when possible) then round-robin into 12 silos
    rng = np.random.default_rng(seed)
    train_df = train_df.copy()
    train_df["_row"] = np.arange(len(train_df))
    # Shuffle within (label, attack_cat) then assign hospital by position % N
    parts = []
    for _, g in train_df.groupby(["label", "_attack_cat"], sort=False):
        parts.append(g.sample(frac=1.0, random_state=seed))
    shuffled = pd.concat(parts, ignore_index=True)
    shuffled["hospital_id"] = [
        f"hospital_{(i % num_clients) + 1:02d}" for i in range(len(shuffled))
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    hospital_ids = [f"hospital_{i + 1:02d}" for i in range(num_clients)]
    tier_cycle = (QUALITY_TIERS * ((num_clients // len(QUALITY_TIERS)) + 1))[:num_clients]

    for hid, (tier, label_noise, feat_noise) in zip(hospital_ids, tier_cycle):
        chunk = shuffled[shuffled["hospital_id"] == hid].drop(
            columns=["hospital_id", "_attack_cat", "_row"], errors="ignore"
        )
        chunk = _apply_quality_tier(chunk, tier, label_noise, feat_noise, rng)
        if tier == "compromised":
            fname = f"{hid}_compromised_wustl_ehms.csv"
        else:
            fname = f"{hid}_{tier}_quality_wustl_ehms.csv"
        out_path = output_dir / fname
        chunk.to_csv(out_path, index=False)
        print(
            f"  wrote {out_path.name}: {len(chunk):,} rows, tier={tier}, "
            f"attacks={int(chunk['label'].sum()):,}"
        )

    test_clean = test_df.drop(columns=["_attack_cat"], errors="ignore")
    test_clean.to_csv(test_output, index=False)
    print(f"\nTest set: {test_output} ({len(test_clean):,} rows)")
    print(f"Clients: {output_dir} ({len(list(output_dir.glob('*.csv')))} files)")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Convert WUSTL-EHMS-2020 to federated clients")
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "wustl-ehms-2020_with_attacks_categories.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "CSVs" / "wustl_ehms_clients",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=root / "data" / "CSVs" / "wustl_ehms_test_set.csv",
    )
    parser.add_argument("--num-clients", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_wustl_ehms_to_clients(
        source=args.source,
        output_dir=args.output_dir,
        test_output=args.test_output,
        num_clients=args.num_clients,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
