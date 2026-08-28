#!/usr/bin/env python
"""Fail-closed placeholder for scorer-weight tuning.

The earlier implementation optimized an answer-level claim-support score from
the same NLI model used by the scorer, allowed random mock NLI, and did not bind
the evaluator or labels to immutable artifacts.  Those outputs were neither an
independent passage-ranking objective nor publication-safe evidence.

Weight tuning remains disabled until a frozen development split, complete
component-score artifact, independently revision-bound relevance judgments,
and a sealed output schema are implemented.  Use ``scripts/analyze_scorer.py``
for the existing fail-closed, independently labeled scorer analysis.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def generate_weight_grid(step: float = 0.1) -> List[Tuple[float, float, float]]:
    """Generate ``(w_nli, w_overlap, w_ret)`` triples that sum to one.

    This pure helper is retained for future artifact-bound tuning and tests; it
    does not score data or select weights.
    """
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise TypeError("step must be numeric")
    step_value = float(step)
    if not np.isfinite(step_value) or not 0.0 < step_value <= 1.0:
        raise ValueError("step must be finite and in (0, 1]")

    grid: List[Tuple[float, float, float]] = []
    values = np.arange(0.0, 1.0 + step_value / 2, step_value)
    for w_nli in values:
        for w_overlap in values:
            w_ret = 1.0 - w_nli - w_overlap
            if w_ret >= -1e-9:
                grid.append(
                    (
                        round(float(w_nli), 12),
                        round(float(w_overlap), 12),
                        round(max(float(w_ret), 0.0), 12),
                    )
                )
    return grid


def main() -> None:
    """Refuse to manufacture an unsealed tuning result."""
    raise RuntimeError(
        "scorer-weight tuning is disabled until it consumes a frozen dev split, "
        "complete component scores, independent revision-bound judgments, and "
        "sealed metric identities; use scripts/analyze_scorer.py for current "
        "independently labeled scorer analysis"
    )


if __name__ == "__main__":
    main()
