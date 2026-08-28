"""Pure compatibility checks for alias-aware analysis helpers."""

from __future__ import annotations

import pytest

from scripts.analyze_errors import classify_error, main


def test_error_classifier_accepts_reference_aliases() -> None:
    record = {
        "answer": "NYC",
        "trusted_passages": [{"text": "New York City is also called NYC."}],
        "confidence_tag": "high",
        "retrieval_triggered": True,
    }

    assert classify_error(record, ["New York City", "NYC"]) == "correct"


def test_error_classifier_does_not_infer_a_cause_from_prediction_signals() -> None:
    record = {
        "answer": "wrong",
        "trusted_passages": [],
        "confidence_tag": "low",
        "retrieval_triggered": False,
    }

    assert classify_error(record, "correct") == "unadjudicated"


def test_error_taxonomy_cli_is_disabled_until_audited_artifact_exists() -> None:
    with pytest.raises(RuntimeError, match="evaluator-only audit codebook"):
        main()
