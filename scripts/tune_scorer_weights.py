#!/usr/bin/env python
"""
scripts/tune_scorer_weights.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 5A-3: Grid search over scorer fusion weights.

Searches over (w_nli, w_overlap, w_ret) subject to sum=1 constraint
and evaluates FactScore on a validation set.

Usage::

    python scripts/tune_scorer_weights.py \\
        --config configs/exp_full_pipeline.yaml \\
        --step 0.1 \\
        --sample 200 \\
        --seed 42 \\
        --output analysis/weight_tuning.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search scorer weights.")
    p.add_argument("--config", type=str, default="configs/exp_full_pipeline.yaml")
    p.add_argument("--step", type=float, default=0.1, help="Grid step size.")
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/weight_tuning.json")
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


def generate_weight_grid(step: float = 0.1) -> List[Tuple[float, float, float]]:
    """Generate all (w_nli, w_overlap, w_ret) triples summing to 1.

    Args:
        step: Grid step size.

    Returns:
        List of weight triples.

    Example::

        >>> grid = generate_weight_grid(0.5)
        >>> all(abs(sum(w) - 1.0) < 1e-6 for w in grid)
        True
    """
    grid: List[Tuple[float, float, float]] = []
    values = np.arange(0.0, 1.0 + step / 2, step)
    for w1 in values:
        for w2 in values:
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-9:
                grid.append((round(w1, 2), round(w2, 2), round(max(w3, 0.0), 2)))
    return grid


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    from factuality_rag.scorer.passage import PassageScorer
    from factuality_rag.eval.metrics import compute_factscore
    from factuality_rag.pipeline.orchestrator import _load_config, Pipeline

    cfg = _load_config(args.config)

    # Build pipeline for retrieval
    pipe = Pipeline(config_path=args.config, mock_mode=args.mock, seed=args.seed)

    # Generate test queries
    if args.mock:
        queries = [
            "What is the capital of France?",
            "Who discovered penicillin?",
            "What is DNA?",
        ] * (args.sample // 3 + 1)
        queries = queries[:args.sample]
    else:
        # TODO: Load real queries from dataset
        logger.warning("Real dataset loading not yet implemented. Use --mock.")
        return

    # Collect (query, passages, answer) triples
    logger.info("Running pipeline on %d queries ...", len(queries))
    cache: List[Dict[str, Any]] = []
    for q in queries:
        answer, trusted, provenance, tag = pipe.run(q, gate=False)
        cache.append({"query": q, "answer": answer, "passages": trusted})

    # Grid search
    weight_grid = generate_weight_grid(args.step)
    logger.info("Searching %d weight combinations ...", len(weight_grid))

    results: List[Dict[str, Any]] = []
    best_score = -1.0
    best_weights = (0.5, 0.2, 0.3)

    for w_nli, w_overlap, w_ret in weight_grid:
        scorer = PassageScorer(
            mock_mode=args.mock,
            w_nli=w_nli,
            w_overlap=w_overlap,
            w_ret=w_ret,
        )

        fs_scores = []
        for item in cache:
            passages = [dict(p) for p in item["passages"]]  # copy
            if passages:
                scored = scorer.score_passages(item["query"], passages)
                filtered = [p for p in scored if p.get("final_score", 0) >= 0.4]
            else:
                filtered = []

            if filtered and item["answer"]:
                fs = compute_factscore(
                    item["answer"], filtered,
                    nli_fn=scorer._nli_entailment,
                )
                fs_scores.append(fs["factscore"])

        avg_fs = float(np.mean(fs_scores)) if fs_scores else 0.0

        results.append({
            "w_nli": w_nli,
            "w_overlap": w_overlap,
            "w_ret": w_ret,
            "factscore": round(avg_fs, 4),
            "n_scored": len(fs_scores),
        })

        if avg_fs > best_score:
            best_score = avg_fs
            best_weights = (w_nli, w_overlap, w_ret)

    results.sort(key=lambda x: x["factscore"], reverse=True)

    output = {
        "best_weights": {
            "w_nli": best_weights[0],
            "w_overlap": best_weights[1],
            "w_ret": best_weights[2],
        },
        "best_factscore": round(best_score, 4),
        "grid_step": args.step,
        "n_combinations": len(weight_grid),
        "top_10": results[:10],
        "all_results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Weight tuning saved → %s", out_path)
    logger.info("Best weights: w_nli=%.2f, w_overlap=%.2f, w_ret=%.2f → FS=%.4f",
                best_weights[0], best_weights[1], best_weights[2], best_score)


if __name__ == "__main__":
    main()
