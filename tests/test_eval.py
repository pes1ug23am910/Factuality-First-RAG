"""
tests.test_eval
~~~~~~~~~~~~~~~~
Unit tests for reference metrics, explicitly labelled lexical support,
and the fail-closed NLI primitive.
"""

from __future__ import annotations

import pytest

from factuality_rag.abstention import CANONICAL_ABSTENTION, is_canonical_abstention
from factuality_rag.eval.metrics import (
    compute_em,
    compute_f1,
    compute_lexical_support,
    compute_nli_claim_support,
    decompose_claims,
    evaluate_predictions,
)
from factuality_rag.generator.wrapper import Generator


class TestMetrics:
    @pytest.mark.parametrize(
        "answer",
        [
            CANONICAL_ABSTENTION,
            "  i CANNOT answer BASED on the provided CONTEXT.  ",
            "I cannot\nanswer based on\tthe provided context.",
        ],
    )
    def test_canonical_abstention_matching_is_case_and_whitespace_robust(self, answer: str) -> None:
        assert is_canonical_abstention(answer) is True

    @pytest.mark.parametrize(
        "answer",
        [
            "I cannot answer based on the provided context",
            "I cannot answer based on the provided context. But Paris might be correct.",
            "Preface: I cannot answer based on the provided context.",
            "I do not have enough evidence to answer.",
        ],
    )
    def test_canonical_abstention_matching_is_exact_not_substring_based(self, answer: str) -> None:
        assert is_canonical_abstention(answer) is False

    def test_canonical_abstention_rejects_non_string(self) -> None:
        with pytest.raises(TypeError, match="answer must be a string"):
            is_canonical_abstention(None)  # type: ignore[arg-type]

    def test_generator_prompt_uses_shared_canonical_abstention(self) -> None:
        prompt = Generator._format_prompt("question", "context")
        assert f'"{CANONICAL_ABSTENTION}"' in prompt

    def test_exact_match_positive(self) -> None:
        assert compute_em("Paris", "paris") == 1.0

    def test_exact_match_negative(self) -> None:
        assert compute_em("London", "Paris") == 0.0

    def test_exact_match_uses_standard_article_punctuation_normalization(self) -> None:
        assert compute_em("The Paris!", "paris") == 1.0

    def test_token_f1_uses_standard_article_punctuation_normalization(self) -> None:
        assert compute_f1("The Paris!", "paris") == 1.0

    def test_f1_identical(self) -> None:
        assert compute_f1("the cat sat", "the cat sat") == 1.0

    def test_f1_partial(self) -> None:
        score = compute_f1("the cat", "the cat sat on mat")
        assert 0.0 < score < 1.0

    def test_lexical_support_supported(self) -> None:
        ps = [{"text": "Paris is the capital of France"}]
        assert compute_lexical_support(["Paris is a capital"], ps) == 1.0

    def test_lexical_support_unsupported(self) -> None:
        ps = [{"text": "Paris is the capital of France"}]
        assert compute_lexical_support(["Tokyo is in Japan"], ps) == 0.0

    def test_lexical_support_uses_tokens_not_substrings(self) -> None:
        ps = [{"text": "the theme"}]
        assert compute_lexical_support(["he me"], ps) == 0.0

    def test_evaluate_predictions(self) -> None:
        preds = [{"answer": "Paris"}, {"answer": "London"}]
        refs = ["Paris", "Berlin"]
        metrics = evaluate_predictions(preds, refs)
        assert metrics["exact_match"] == 0.5
        assert metrics["support_metric"] == "none"
        assert not any("factscore" in key for key in metrics)

    def test_evaluate_predictions_rejects_reference_length_mismatch(self) -> None:
        predictions = [{"answer": "Paris"}, {"answer": "London"}]
        with pytest.raises(ValueError, match="same length"):
            evaluate_predictions(predictions, ["Paris"])

    def test_evaluate_predictions_accepts_reference_aliases(self) -> None:
        predictions = [{"answer": "NYC"}]
        metrics = evaluate_predictions(predictions, [["New York City", "NYC"]])
        assert metrics["exact_match"] == 1.0
        assert metrics["f1"] == 1.0

    def test_evaluate_predictions_accepts_tuple_aliases(self) -> None:
        metrics = evaluate_predictions([{"answer": "NYC"}], [("New York City", "NYC")])
        assert metrics["exact_match"] == 1.0

    def test_evaluate_predictions_rejects_empty_aliases(self) -> None:
        with pytest.raises(ValueError, match="at least one non-blank string reference"):
            evaluate_predictions([{"answer": "NYC"}], [[]])

    def test_evaluate_predictions_rejects_blank_aliases(self) -> None:
        with pytest.raises(ValueError, match="non-blank"):
            evaluate_predictions([{"answer": "NYC"}], [["NYC", " "]])

    def test_answer_without_trusted_evidence_counts_as_unsupported(self) -> None:
        predictions = [
            {
                "answer": "Paris is the capital of France.",
                "trusted_passages": [{"text": "Paris is the capital of France."}],
            },
            {
                "answer": "Berlin is the capital of Germany.",
                "trusted_passages": [],
            },
        ]
        metrics = evaluate_predictions(predictions, support_metric="lexical")
        assert metrics["lexical_support_answered_only"] == 0.5
        assert metrics["lexical_support_answered_count"] == 2.0
        assert metrics["answer_coverage"] == 1.0
        assert not any("factscore" in key for key in metrics)

    def test_lexical_scope_excludes_abstentions_and_reports_coverage(self) -> None:
        predictions = [
            {
                "answer": "Paris is the capital of France.",
                "trusted_passages": [{"text": "Paris is the capital of France."}],
            },
            {"answer": "   ", "trusted_passages": []},
            {
                "answer": "  i CANNOT answer\n based on the provided CONTEXT.  ",
                "trusted_passages": [
                    {"text": "The abstention words appear in this irrelevant passage."}
                ],
            },
        ]
        metrics = evaluate_predictions(predictions, support_metric="lexical")
        assert metrics["lexical_support_answered_only"] == 1.0
        assert metrics["lexical_support_answered_count"] == 1.0
        assert metrics["answered_count"] == 1.0
        assert metrics["answer_coverage"] == pytest.approx(1 / 3)

    def test_abstention_substring_with_additional_answer_counts_as_answered(self) -> None:
        prediction = {
            "answer": CANONICAL_ABSTENTION + " Paris is the capital of France.",
            "trusted_passages": [{"text": "Paris is the capital of France."}],
        }
        metrics = evaluate_predictions([prediction], support_metric="lexical")
        assert metrics["answered_count"] == 1.0
        assert metrics["answer_coverage"] == 1.0
        assert metrics["lexical_support_answered_count"] == 1.0

    @pytest.mark.parametrize(
        ("prediction", "error", "message"),
        [
            ({}, ValueError, "contain an answer"),
            ({"answer": None}, TypeError, "answer must be a string"),
            ({"answer": "Paris", "trusted_passages": {}}, TypeError, "must be a list"),
        ],
    )
    def test_evaluate_predictions_rejects_invalid_prediction_schema(
        self,
        prediction: dict[str, object],
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            evaluate_predictions([prediction])

    def test_evaluate_rejects_unpinned_nli_callable(self) -> None:
        with pytest.raises(ValueError, match="immutable scorer"):
            evaluate_predictions([], nli_fn=lambda premise, claim: 1.0)

    def test_evaluate_rejects_unknown_support_metric(self) -> None:
        with pytest.raises(ValueError, match="none.*lexical"):
            evaluate_predictions([], support_metric="nli")


class TestClaimDecomposition:
    def test_simple_split(self) -> None:
        claims = decompose_claims("Paris is the capital. It has 2M people.")
        assert len(claims) == 2
        assert "Paris" in claims[0]

    def test_empty_string(self) -> None:
        assert decompose_claims("") == []

    def test_single_sentence(self) -> None:
        claims = decompose_claims("DNA is a molecule.")
        assert len(claims) == 1

    def test_question_mark_split(self) -> None:
        claims = decompose_claims("What is DNA? It is a molecule.")
        assert len(claims) == 2


class TestNLIClaimSupport:
    def test_nli_claim_support_requires_nli(self) -> None:
        """The NLI primitive never falls back to lexical overlap."""
        ps = [{"id": "0", "text": "Paris is the capital of France"}]
        with pytest.raises(TypeError, match="nli_fn"):
            compute_nli_claim_support(  # type: ignore[call-arg]
                "Paris is the capital of France.", ps
            )

    def test_zero_threshold_without_evidence_is_not_supported(self) -> None:
        result = compute_nli_claim_support(
            "A supported-looking sentence.",
            [],
            nli_fn=lambda premise, claim: 0.0,
            entailment_threshold=0.0,
        )

        assert result["n_supported"] == 0
        assert result["nli_claim_support"] == 0.0
        assert result["details"][0]["supported"] is False
        assert result["details"][0]["best_passage_id"] is None

    def test_nli_claim_support_with_mock_nli(self) -> None:
        """With a mock nli_fn, should use it."""

        def always_entail(premise: str, hypothesis: str) -> float:
            return 0.9

        ps = [{"id": "0", "text": "anything"}]
        result = compute_nli_claim_support("Claim one. Claim two.", ps, nli_fn=always_entail)
        assert result["nli_claim_support"] == 1.0
        assert result["n_supported"] == 2

    def test_nli_claim_support_unsupported(self) -> None:
        def never_entail(premise: str, hypothesis: str) -> float:
            return 0.1

        ps = [{"id": "0", "text": "anything"}]
        result = compute_nli_claim_support("Claim one. Claim two.", ps, nli_fn=never_entail)
        assert result["nli_claim_support"] == 0.0
        assert result["n_supported"] == 0

    def test_nli_claim_support_empty_answer(self) -> None:
        result = compute_nli_claim_support("", [{"text": "passage"}], nli_fn=lambda p, h: 1.0)
        assert result["n_claims"] == 0

    @pytest.mark.parametrize("invalid", [True, float("nan"), -0.1, 1.1])
    def test_nli_claim_support_rejects_invalid_nli_output(self, invalid: object) -> None:
        with pytest.raises((TypeError, ValueError), match="nli_fn"):
            compute_nli_claim_support(
                "A supported claim.",
                [{"text": "passage"}],
                nli_fn=lambda premise, claim: invalid,  # type: ignore[return-value]
            )
