#!/usr/bin/env python
"""
scripts/aggregate_results.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Aggregate metrics from multiple experiment runs across seeds.

Computes mean ± std for each metric and produces a summary table.

Usage::

    python scripts/aggregate_results.py \\
        --runs-dir runs \\
        --configs B1 B2 B3 B4 full \\
        --output analysis/aggregated_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment results.")
    p.add_argument("--runs-dir", type=str, default="runs", help="Base runs directory.")
    p.add_argument("--configs", nargs="+", default=["B1", "B2", "B3", "B4", "full"],
                   help="Config names to aggregate.")
    p.add_argument("--output", type=str, default="analysis/aggregated_results.json")
    return p.parse_args()


def load_run_metrics(run_dir: Path) -> Dict[str, float]:
    """Load metrics.json from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict of metric name → value.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load metadata.json from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Metadata dict.
    """
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    runs_base = Path(args.runs_dir)

    if not runs_base.exists():
        logger.warning("Runs directory not found: %s", runs_base)
        return

    # Discover all run directories
    all_runs = sorted(d for d in runs_base.iterdir() if d.is_dir())
    logger.info("Found %d run directories in %s", len(all_runs), runs_base)

    # Group runs by config
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for run_dir in all_runs:
        meta = load_run_metadata(run_dir)
        config_path = meta.get("config_path", "")

        # Match config name
        config_name = "unknown"
        for name in args.configs:
            if name.lower() in config_path.lower():
                config_name = name
                break

        if config_name not in grouped:
            grouped[config_name] = []

        metrics = load_run_metrics(run_dir)
        if metrics:
            grouped[config_name].append(metrics)

    # Aggregate
    summary: Dict[str, Dict[str, Any]] = {}
    metric_keys = {"exact_match", "f1", "factscore", "n_predictions"}

    for config_name, runs in grouped.items():
        if not runs:
            continue

        agg: Dict[str, Any] = {"n_runs": len(runs)}
        all_keys = set()
        for r in runs:
            all_keys.update(r.keys())

        for key in sorted(all_keys & metric_keys):
            values = [r[key] for r in runs if key in r]
            if values:
                agg[key] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                    "values": [round(v, 4) for v in values],
                }

        summary[config_name] = agg

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Config':<12} {'EM':>12} {'F1':>12} {'FactScore':>12} {'Runs':>6}")
    print("-" * 80)
    for name in args.configs:
        if name in summary:
            s = summary[name]
            em = s.get("exact_match", {})
            f1 = s.get("f1", {})
            fs = s.get("factscore", {})
            print(
                f"{name:<12} "
                f"{em.get('mean', 0):.4f}±{em.get('std', 0):.4f}  "
                f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}  "
                f"{fs.get('mean', 0):.4f}±{fs.get('std', 0):.4f}  "
                f"{s['n_runs']:>6}"
            )
    print("=" * 80 + "\n")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Aggregated results saved → %s", out_path)


if __name__ == "__main__":
    main()
