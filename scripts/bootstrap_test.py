#!/usr/bin/env python
"""
scripts/bootstrap_test.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Paired bootstrap significance test between two systems.

Compares per-query metric scores from two different experiment
runs and computes a p-value via paired bootstrap resampling.

Usage::

    python scripts/bootstrap_test.py \\
        --system-a runs/<run-a>/predictions.jsonl \\
        --system-b runs/<run-b>/predictions.jsonl \\
        --metric exact_match \\
        --n-bootstrap 10000 \\
        --seed 42 \\
        --output analysis/bootstrap_test.json

Reference: Berg-Kirkpatrick et al. (2012), "An Empirical Investigation
of Statistical Significance in NLP"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap significance test.")
    p.add_argument("--system-a", type=str, required=True, help="predictions.jsonl for system A")
    p.add_argument("--system-b", type=str, required=True, help="predictions.jsonl for system B")
    p.add_argument("--metric", type=str, default="exact_match",
                   choices=["exact_match", "f1", "factscore"],
                   help="Metric to compare.")
    p.add_argument("--n-bootstrap", type=int, default=10_000, help="Number of bootstrap samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/bootstrap_test.json")
    return p.parse_args()


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file.

    Args:
        path: Path to predictions.jsonl.

    Returns:
        List of prediction dicts.
    """
    predictions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def compute_per_query_metric(
    predictions: List[Dict[str, Any]],
    metric: str,
    references: List[str] | None = None,
) -> List[float]:
    """Compute per-query metric scores.

    Args:
        predictions: List of prediction dicts.
        metric: Metric name.
        references: Optional gold references.

    Returns:
        List of per-query scores.
    """
    from factuality_rag.eval.metrics import compute_em, compute_f1, compute_factscore

    scores = []
    for i, pred in enumerate(predictions):
        answer = pred.get("answer", "")

        if metric == "exact_match" and references:
            scores.append(compute_em(answer, references[i]))
        elif metric == "f1" and references:
            scores.append(compute_f1(answer, references[i]))
        elif metric == "factscore":
            passages = pred.get("trusted_passages", [])
            if passages and answer:
                result = compute_factscore(answer, passages)
                scores.append(result["factscore"])
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

    return scores


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Run a paired bootstrap significance test.

    Tests whether system B is significantly better than system A.

    Args:
        scores_a: Per-query scores for system A.
        scores_b: Per-query scores for system B.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        (p_value, mean_delta, ci_95) where ci_95 is the 95%
        confidence interval half-width.

    Example::

        >>> import numpy as np
        >>> a = np.array([0.5, 0.6, 0.7])
        >>> b = np.array([0.6, 0.7, 0.8])
        >>> p, delta, ci = paired_bootstrap_test(a, b, n_bootstrap=100)
        >>> delta > 0
        True
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n, "Scores must be parallel."

    observed_delta = float(scores_b.mean() - scores_a.mean())

    wins = 0
    deltas = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_a = scores_a[indices].mean()
        boot_b = scores_b[indices].mean()
        delta = boot_b - boot_a
        deltas[i] = delta
        if delta > 0:
            wins += 1

    p_value = 1.0 - (wins / n_bootstrap)
    ci_95 = float(1.96 * deltas.std())

    return p_value, observed_delta, ci_95


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Load predictions
    preds_a = load_predictions(args.system_a)
    preds_b = load_predictions(args.system_b)

    if len(preds_a) != len(preds_b):
        logger.warning(
            "Different numbers of predictions: A=%d, B=%d. "
            "Using minimum.",
            len(preds_a), len(preds_b),
        )
        min_n = min(len(preds_a), len(preds_b))
        preds_a = preds_a[:min_n]
        preds_b = preds_b[:min_n]

    # Compute per-query scores
    scores_a = np.array(compute_per_query_metric(preds_a, args.metric))
    scores_b = np.array(compute_per_query_metric(preds_b, args.metric))

    # Run bootstrap test
    p_value, delta, ci_95 = paired_bootstrap_test(
        scores_a, scores_b,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    result = {
        "system_a": args.system_a,
        "system_b": args.system_b,
        "metric": args.metric,
        "n_queries": len(scores_a),
        "mean_a": round(float(scores_a.mean()), 4),
        "mean_b": round(float(scores_b.mean()), 4),
        "delta": round(delta, 4),
        "p_value": round(p_value, 4),
        "ci_95": round(ci_95, 4),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_bootstrap": args.n_bootstrap,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Bootstrap test saved → %s", out_path)
    logger.info(
        "Result: Δ=%.4f, p=%.4f %s",
        delta,
        p_value,
        "(significant at p<0.05)" if p_value < 0.05 else "(not significant)",
    )


if __name__ == "__main__":
    main()
