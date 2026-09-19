#!/usr/bin/env python3
"""
Aggregate TrustFed-Agent experiment metrics and print summary tables.
Usage: python scripts/analyze_trustfed_agent_results.py --input results/trustfed_agent/metrics/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_metrics(metrics_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("run_*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def mean_std(values: List[float]) -> str:
    if not values:
        return "N/A"
    import statistics
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:.2f} ± {s:.2f}"


def summarize(records: List[Dict[str, Any]], run_id: str) -> Dict[str, str]:
    subset = [
        r for r in records
        if r.get("run_id", r.get("approach")) == run_id
    ]
    if not subset:
        return {"run_id": run_id, "n": "0", "f1": "N/A", "response_precision": "N/A",
                "compliance_rate": "N/A", "defense_utility": "N/A"}

    def col(key_path: str) -> List[float]:
        out = []
        for r in subset:
            node = r
            for k in key_path.split("."):
                node = node.get(k, {}) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, (int, float)):
                out.append(float(node))
        return out

    return {
        "run_id": run_id,
        "approach": subset[0].get("approach", run_id),
        "dataset": subset[0].get("dataset", "iomt"),
        "n": str(len(subset)),
        "f1": mean_std(col("detection.f1")),
        "response_precision": mean_std(col("response.precision")),
        "compliance_rate": mean_std(col("governance.compliance_rate")),
        "defense_utility": mean_std(col("response.cumulative_defense_utility")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TrustFed-Agent metrics")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/trustfed_agent/metrics"),
        help="Directory containing run_*.json metric files",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"No metrics directory: {args.input}")
        print("Run experiments first; metric JSON files are written per seed.")
        return

    records = load_metrics(args.input)
    if not records:
        print(f"No run_*.json files in {args.input}")
        return

    run_ids = sorted({r.get("run_id", r.get("approach", "unknown")) for r in records})
    print("\n=== TrustFed-Agent Results Summary ===\n")
    print(f"{'Run ID':<32} {'n':>4} {'F1':>18} {'Resp Prec':>18} {'Compliance':>18} {'Defense Util':>18}")
    print("-" * 112)
    for rid in run_ids:
        row = summarize(records, rid)
        print(
            f"{row['run_id']:<32} {row['n']:>4} {row['f1']:>18} "
            f"{row['response_precision']:>18} {row['compliance_rate']:>18} {row['defense_utility']:>18}"
        )
    print()

    if args.export:
        summary = {rid: summarize(records, rid) for rid in run_ids}
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary exported: {args.export}")


if __name__ == "__main__":
    main()
