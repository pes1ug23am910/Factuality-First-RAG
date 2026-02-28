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
    p.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    p.add_argument("--references", type=str, default=None, help="Path to references JSON (optional).")
    p.add_argument("--output", type=str, default="analysis/error_taxonomy.json")
    return p.parse_args()


def classify_error(record: Dict[str, Any], reference: str | None = None) -> str:
    """Classify a single prediction into an error category.

    Args:
        record: Prediction dict from predictions.jsonl.
        reference: Optional gold reference answer.

    Returns:
        Error category string or "correct".
    """
    from factuality_rag.eval.metrics import compute_em

    answer = record.get("answer", "")
    trusted = record.get("trusted_passages", [])
    confidence = record.get("confidence_tag", "")

    # If no reference, classify based on structural signals
    if reference is None:
        if not trusted and confidence == "low":
            return "no_evidence"
        return "unknown"

    is_correct = compute_em(answer, reference) > 0.5

    if is_correct:
        return "correct"

    # Wrong answer — classify why
    if confidence == "medium" and not trusted:
        # Gating skipped retrieval
        return "gating_miss"

    if not trusted:
        # Retrieval happened but no passages passed scoring
        return "scoring_miss"

    # Had trusted passages but still wrong
    return "generation_miss"


def main() -> None:
    args = parse_args()

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        return

    predictions: List[Dict[str, Any]] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    logger.info("Loaded %d predictions from %s", len(predictions), pred_path)

    # Load references if provided
    references: Dict[str, str] = {}
    if args.references:
        ref_path = Path(args.references)
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                references = json.load(f)

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
            "category": category,
            "confidence_tag": record.get("confidence_tag", ""),
            "n_trusted": len(record.get("trusted_passages", [])),
        })

    # Summary
    total = len(predictions)
    summary: Dict[str, Any] = {
        "total_predictions": total,
        "category_counts": dict(taxonomy),
        "category_rates": {
            k: round(v / total, 4) for k, v in taxonomy.items()
        } if total > 0 else {},
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
