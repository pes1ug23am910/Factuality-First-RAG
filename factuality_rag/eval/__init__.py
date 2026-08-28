"""Evaluation metrics with explicitly separated lexical and NLI support."""

from factuality_rag.eval.metrics import (  # noqa: F401
    compute_em,
    compute_em_aliases,
    compute_f1,
    compute_f1_aliases,
    compute_lexical_support,
    compute_nli_claim_support,
    evaluate_predictions,
    reference_aliases,
)
