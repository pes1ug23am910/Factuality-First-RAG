#!/usr/bin/env python
"""Fail-closed boundary for the unfinished error-analysis protocol.

The project does not yet have the frozen evaluator-only audit artifact required
to assign labels such as retrieval, scorer, gating, or generation failures.
Prediction-side heuristics can identify whether an answer matches a frozen
reference, but they cannot establish why a wrong answer occurred. The CLI is
therefore disabled until the codebook, blinded LLM-assisted prelabels,
single-author review, corrections, and sealed artifact contract are frozen.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def classify_error(record: Dict[str, Any], reference: Optional[Any] = None) -> str:
    """Return only reference correctness or an explicitly unadjudicated state.

    This compatibility helper deliberately makes no causal diagnosis.  A wrong
    answer remains ``"unadjudicated"`` until an eligible audited error label exists.
    """

    if reference is None:
        return "unknown"

    from factuality_rag.eval.metrics import compute_em_aliases

    answer = record.get("answer", "")
    if type(answer) is not str:
        raise TypeError("record.answer must be a string")
    return "correct" if compute_em_aliases(answer, reference) > 0.5 else "unadjudicated"


def main() -> None:
    """Refuse to emit a taxonomy without the frozen author-audited artifact."""

    raise RuntimeError(
        "error-taxonomy analysis is disabled: freeze the evaluator-only audit "
        "codebook and bind a blinded, single-author-reviewed artifact before use"
    )


if __name__ == "__main__":
    main()
