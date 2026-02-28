#!/usr/bin/env python
"""
scripts/analyze_scorer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 4B: Scorer AUC analysis.

Evaluates PassageScorer's ability to distinguish gold (relevant)
passages from distractors.  Reports ROC-AUC, PR-AUC, and optimal
threshold.

Usage::

    python scripts/analyze_scorer.py \\
        --config configs/exp_full_pipeline.yaml \\
        --sample 500 \\
        --seed 42 \\
        --output analysis/scorer_auc.json
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
    p = argparse.ArgumentParser(description="Scorer AUC analysis.")
    p.add_argument("--config", type=str, default="configs/exp_full_pipeline.yaml")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/scorer_auc.json")
    p.add_argument("--mock", action="store_true", help="Run in mock mode for testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    from factuality_rag.scorer.passage import PassageScorer
    from factuality_rag.pipeline.orchestrator import _load_config

    cfg = _load_config(args.config)
    scorer_cfg = cfg.get("scorer", {})
    weights = scorer_cfg.get("weights", {})

    scorer = PassageScorer(
        nli_model_hf=cfg.get("models", {}).get(
            "nli_verifier",
            "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        ),
        overlap_metric=scorer_cfg.get("overlap_metric", "token"),
        mock_mode=args.mock,
        w_nli=weights.get("w_nli", 0.5),
        w_overlap=weights.get("w_overlap", 0.2),
        w_ret=weights.get("w_ret", 0.3),
        nli_mode=scorer_cfg.get("nli_mode", "passage"),
    )

    # In mock mode, generate synthetic passages with labels
    if args.mock:
        queries = ["What is the capital of France?"] * args.sample
        gold_passages = [
            {"id": f"gold_{i}", "text": f"Paris is the capital of France {i}",
             "combined_score": 0.8, "label": 1}
            for i in range(args.sample)
        ]
        distractor_passages = [
            {"id": f"dist_{i}", "text": f"Berlin is a large city in Europe {i}",
             "combined_score": 0.3, "label": 0}
            for i in range(args.sample)
        ]
    else:
        # TODO: Load real gold/distractor passages from dataset
        logger.warning("Real dataset loading not yet implemented. Use --mock for testing.")
        return

    # Score all passages
    all_scores: List[float] = []
    all_labels: List[int] = []

    for i in range(args.sample):
        passages = [gold_passages[i], distractor_passages[i]]
        scored = scorer.score_passages(queries[i], passages)
        for p in scored:
            all_scores.append(p["final_score"])
            all_labels.append(p.get("label", 0))

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    # Compute ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore[import-untyped]
        roc_auc = float(roc_auc_score(labels_arr, scores_arr))
        pr_auc = float(average_precision_score(labels_arr, scores_arr))
    except Exception:
        roc_auc = -1.0
        pr_auc = -1.0

    # Find optimal threshold (Youden's J)
    best_thresh = 0.5
    best_j = -1.0
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (scores_arr >= thresh).astype(int)
        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()
        tn = ((preds == 0) & (labels_arr == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "optimal_threshold": round(best_thresh, 3),
        "youden_j": round(best_j, 4),
        "n_samples": len(all_scores),
        "n_positive": int(labels_arr.sum()),
        "n_negative": int((1 - labels_arr).sum()),
        "mean_score_positive": round(float(scores_arr[labels_arr == 1].mean()), 4),
        "mean_score_negative": round(float(scores_arr[labels_arr == 0].mean()), 4),
    }

    output = {"metrics": metrics}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Scorer AUC analysis saved → %s", out_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
