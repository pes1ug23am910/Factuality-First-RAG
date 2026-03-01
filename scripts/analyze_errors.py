#!/usr/bin/env python
"""
scripts/analyze_errors.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 4C: Error taxonomy analysis.

Reads a predictions.jsonl file and classifies errors into:
- retrieval_miss: relevant passages not retrieved
- scoring_miss: retrieved but filtered by scorer
- generation_miss: passages trusted but answer wrong
- gating_miss: retrieval skipped when it should have happened

Usage::

    python scripts/analyze_errors.py \\
        --predictions runs/<run-id>/predictions.jsonl \\
        --references data/references.json \\
        --output analysis/error_taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error taxonomy analysis.")
    p.add_argument("--predictions", type=str, default=None,
                   help="Path to predictions.jsonl")
    p.add_argument("--full-run", type=str, default=None,
                   help="Path to full pipeline run directory.")
    p.add_argument("--b2-run", type=str, default=None,
                   help="Path to always-RAG (B2) run directory for comparison.")
    p.add_argument("--references", type=str, default=None,
                   help="Path to references JSON (optional).")
    p.add_argument("--n-sample", type=int, default=50,
                   help="Number of failures to sample for taxonomy.")
    p.add_argument("--output", type=str, default="analysis/error_taxonomy.json")
    return p.parse_args()


def classify_error(record: Dict[str, Any], reference: str | None = None) -> str:
    """Classify a single prediction into an error category.

    Uses Session 4 error codes:
    - GATE_MISS: gated out retrieval, model hallucinated
    - RETRIEVAL_MISS: relevant passages not retrieved
    - SCORER_DROP: retrieved but filtered by scorer
    - GEN_IGNORE: passages trusted but answer wrong
    - ANSWER_FORMAT: correct answer but EM normalisation failed
    - CORPUS_GAP: answer simply not in corpus

    Args:
        record: Prediction dict from predictions.jsonl.
        reference: Optional gold reference answer.

    Returns:
        Error category string or "correct".
    """
    from factuality_rag.eval.metrics import compute_em, compute_f1

    answer = record.get("answer", "")
    trusted = record.get("trusted_passages", [])
    confidence = record.get("confidence_tag", "")
    retrieval_triggered = record.get("retrieval_triggered", True)

    # If no reference, classify based on structural signals
    if reference is None:
        if not trusted and confidence == "low":
            return "no_evidence"
        return "unknown"

    is_correct = compute_em(answer, reference) > 0.5

    if is_correct:
        return "correct"

    # Check if F1 is high (possible ANSWER_FORMAT issue)
    f1 = compute_f1(answer, reference)
    if f1 > 0.6:
        return "ANSWER_FORMAT"

    # Wrong answer — classify why
    if not retrieval_triggered:
        # Gating skipped retrieval — model hallucinated
        return "GATE_MISS"

    if not trusted:
        # Retrieval happened but no passages passed scoring
        return "SCORER_DROP"

    # Had trusted passages — check if they contain relevant info
    passage_text = " ".join(p.get("text", "").lower() for p in trusted)
    ref_tokens = set(reference.lower().split())
    overlap = sum(1 for t in ref_tokens if t in passage_text)
    if ref_tokens and overlap / len(ref_tokens) < 0.3:
        return "RETRIEVAL_MISS"

    # Had relevant passages but still wrong
    return "GEN_IGNORE"


def main() -> None:
    args = parse_args()

    # Determine predictions path
    pred_path: Path
    if args.full_run:
        pred_path = Path(args.full_run) / "predictions.jsonl"
    elif args.predictions:
        pred_path = Path(args.predictions)
    else:
        logger.error("Must provide either --full-run or --predictions.")
        return

    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        return

    predictions: List[Dict[str, Any]] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    logger.info("Loaded %d predictions from %s", len(predictions), pred_path)

    # Load references: from predictions (inline), references.json, or --references flag
    references: Dict[str, str] = {}

    # Check inline references in predictions
    for pred in predictions:
        ref = pred.get("reference")
        if ref:
            references[pred.get("input", "")] = ref

    # Check references.json in run directory
    if args.full_run:
        ref_path = Path(args.full_run) / "references.json"
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                references.update(json.load(f))

    # Override with explicit --references flag
    if args.references:
        ref_path = Path(args.references)
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                references.update(json.load(f))

    # Classify each prediction
    taxonomy: Counter = Counter()
    details: List[Dict[str, Any]] = []

    for record in predictions:
        query = record.get("input", "")
        ref = references.get(query)
        category = classify_error(record, ref)
        taxonomy[category] += 1
        details.append({
            "query": query,
            "answer": record.get("answer", ""),
            "reference": ref or "",
            "category": category,
            "confidence_tag": record.get("confidence_tag", ""),
            "retrieval_triggered": record.get("retrieval_triggered", True),
            "n_trusted": len(record.get("trusted_passages", [])),
        })

    # Sample failures for detailed taxonomy
    failures = [d for d in details if d["category"] not in ("correct", "unknown", "no_evidence")]
    if args.n_sample and len(failures) > args.n_sample:
        import random
        random.seed(42)
        failures_sample = random.sample(failures, args.n_sample)
    else:
        failures_sample = failures

    failure_taxonomy: Counter = Counter()
    for f in failures_sample:
        failure_taxonomy[f["category"]] += 1

    # Summary
    total = len(predictions)
    summary: Dict[str, Any] = {
        "total_predictions": total,
        "category_counts": dict(taxonomy),
        "category_rates": {
            k: round(v / total, 4) for k, v in taxonomy.items()
        } if total > 0 else {},
        "failure_sample_size": len(failures_sample),
        "failure_taxonomy": dict(failure_taxonomy),
    }

    output = {"summary": summary, "per_query": details}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Error taxonomy saved → %s", out_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
