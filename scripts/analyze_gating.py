#!/usr/bin/env python
"""
scripts/analyze_gating.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 4A: Gating oracle analysis.

Compares gating decisions against an "oracle" that retrieves only
when the closed-book answer is wrong.  Reports hit-rate, precision,
recall, and calibration metrics.

Usage::

    python scripts/analyze_gating.py \\
        --config configs/exp_full_pipeline.yaml \\
        --dataset natural_questions \\
        --split validation \\
        --sample 500 \\
        --seed 42 \\
        --output analysis/gating_oracle.json
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
    p = argparse.ArgumentParser(description="Gating oracle analysis.")
    p.add_argument("--config", type=str, default="configs/exp_full_pipeline.yaml")
    p.add_argument("--full-run", type=str, default=None,
                   help="Path to full pipeline run directory (loads predictions).")
    p.add_argument("--closedbook-run", type=str, default=None,
                   help="Path to closed-book run directory (loads predictions).")
    p.add_argument("--dataset", type=str, default="natural_questions")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/gating_oracle.json")
    p.add_argument("--mock", action="store_true", help="Run in mock mode for testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    from factuality_rag.gating.probe import GatingProbe, compute_ece
    from factuality_rag.eval.metrics import compute_em

    # Mode 1: Load from existing run directories
    if args.full_run and args.closedbook_run:
        full_dir = Path(args.full_run)
        cb_dir = Path(args.closedbook_run)

        # Load full pipeline predictions
        full_preds: list = []
        with open(full_dir / "predictions.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    full_preds.append(json.loads(line))

        # Load closed-book predictions
        cb_preds: list = []
        with open(cb_dir / "predictions.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cb_preds.append(json.loads(line))

        # Load references
        refs: Dict[str, str] = {}
        for d in [full_dir, cb_dir]:
            ref_path = d / "references.json"
            if ref_path.exists():
                with open(ref_path, encoding="utf-8") as f:
                    refs.update(json.load(f))

        # Build CB answer and gate decision maps
        cb_map = {p["input"]: p["answer"] for p in cb_preds}

        results: List[Dict[str, Any]] = []
        gate_decisions: List[bool] = []
        oracle_decisions: List[bool] = []

        for pred in full_preds:
            q = pred.get("input", "")
            ref = refs.get(q, "")
            cb_answer = cb_map.get(q, "")
            cb_correct = compute_em(cb_answer, ref) > 0.5 if ref else False

            # Oracle: retrieve only when closed-book is wrong
            oracle_retrieve = not cb_correct

            # Actual decision: infer from prediction record
            retrieval_triggered = pred.get("retrieval_triggered", True)

            gate_decisions.append(retrieval_triggered)
            oracle_decisions.append(oracle_retrieve)

            results.append({
                "query": q,
                "reference": ref,
                "closed_book_answer": cb_answer,
                "closed_book_correct": cb_correct,
                "gate_decision": retrieval_triggered,
                "oracle_decision": oracle_retrieve,
                "match": retrieval_triggered == oracle_retrieve,
            })
    else:
        # Mode 2: Run pipelines live (original behavior)
        from factuality_rag.pipeline.orchestrator import Pipeline

        # Build two pipelines: closed-book (B1) and full
        pipe_closed = Pipeline(config_path="configs/exp_b1_closed_book.yaml", mock_mode=args.mock, seed=args.seed)
        pipe_full = Pipeline(config_path=args.config, mock_mode=args.mock, seed=args.seed)

        # Load queries (demo set in mock mode)
        if args.mock:
            queries = ["What is the capital of France?", "Who wrote Hamlet?", "What is DNA?"]
            references_list = ["Paris", "Shakespeare", "Deoxyribonucleic acid"]
        else:
            from factuality_rag.experiment_runner import _extract_queries_and_references
            queries, references_list = _extract_queries_and_references(
                args.dataset, args.split, args.sample, args.seed,
            )

        # Analyze each query
        results = []
        gate_decisions = []
        oracle_decisions = []

        for i, (q, ref) in enumerate(zip(queries, references_list)):
            # Closed-book answer
            cb_answer, _, _, _ = pipe_closed.run(q, gate=False)
            cb_correct = compute_em(cb_answer, ref) > 0.5

            # Oracle: retrieve only when closed-book is wrong
            oracle_retrieve = not cb_correct

            # Actual gating decision
            info: Dict[str, Any] = {}
            pipe_full.run(q, info=info)
            gate_retrieve = info.get("retrieval_triggered", True)

            gate_decisions.append(gate_retrieve)
            oracle_decisions.append(oracle_retrieve)

            results.append({
                "query": q,
                "reference": ref,
                "closed_book_answer": cb_answer,
                "closed_book_correct": cb_correct,
                "gate_decision": gate_retrieve,
                "oracle_decision": oracle_retrieve,
                "match": gate_retrieve == oracle_retrieve,
            })

            if (i + 1) % 50 == 0:
                logger.info("Processed %d / %d queries.", i + 1, len(queries))

    # Compute metrics
    gate_arr = np.array(gate_decisions)
    oracle_arr = np.array(oracle_decisions)

    tp = int(((gate_arr == True) & (oracle_arr == True)).sum())
    fp = int(((gate_arr == True) & (oracle_arr == False)).sum())
    fn = int(((gate_arr == False) & (oracle_arr == True)).sum())
    tn = int(((gate_arr == False) & (oracle_arr == False)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / len(gate_decisions)

    metrics = {
        "n_queries": len(queries),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gate_retrieve_rate": round(float(gate_arr.mean()), 4),
        "oracle_retrieve_rate": round(float(oracle_arr.mean()), 4),
    }

    output = {"metrics": metrics, "per_query": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Gating oracle analysis saved → %s", out_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
