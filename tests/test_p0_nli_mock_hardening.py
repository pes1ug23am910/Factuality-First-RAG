"""Focused regressions for fail-closed NLI labels and deterministic mock reranking."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from factuality_rag.scorer.passage import PassageScorer


class FakeNliPipeline:
    """Minimal callable matching the Transformers pipeline surface used by the scorer."""

    def __init__(self, output: Any, config: Optional[Any] = None) -> None:
        self.output = output
        self.calls: List[Dict[str, Any]] = []
        self.model = SimpleNamespace(config=config)

    def __call__(
        self,
        payload: Dict[str, str],
        *,
        top_k: Any,
        truncation: str,
        max_length: int,
    ) -> Any:
        self.calls.append(
            {
                "payload": payload,
                "top_k": top_k,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        return self.output


def _scorer_with_pipeline(output: Any, config: Optional[Any] = None) -> PassageScorer:
    scorer = PassageScorer("fake-nli", mock_mode=False)
    scorer._nli_pipeline = FakeNliPipeline(output, config)
    return scorer


@pytest.mark.parametrize(
    "output",
    [
        {"label": " ENTAILMENT ", "score": 0.81},
        [{"label": "ENTAILMENT", "score": 0.81}],
        [[{"label": "entailment", "score": 0.81}]],
    ],
)
def test_nli_accepts_only_documented_single_input_transformers_shapes(output: Any) -> None:
    scorer = _scorer_with_pipeline(output)

    assert scorer._nli_entailment("evidence", "claim") == pytest.approx(0.81)


@pytest.mark.parametrize(
    "output",
    [
        [],
        [[[{"label": "entailment", "score": 0.8}]]],
        [[{"label": "entailment", "score": 0.8}], [{"label": "neutral", "score": 0.2}]],
        "entailment",
    ],
)
def test_nli_rejects_empty_or_ambiguous_output_shapes(output: Any) -> None:
    scorer = _scorer_with_pipeline(output)

    with pytest.raises(RuntimeError, match="unsupported"):
        scorer._nli_entailment("evidence", "claim")


@pytest.mark.parametrize(
    "label",
    [
        "not_entailment",
        "non_entailment",
        "NOT-ENTAILMENT",
        "probably_entailment",
    ],
)
def test_nli_does_not_substring_match_negative_or_unknown_labels(label: str) -> None:
    scorer = _scorer_with_pipeline([{"label": label, "score": 0.99}])

    with pytest.raises(RuntimeError, match="exactly one unambiguous entailment"):
        scorer._nli_entailment("evidence", "claim")


@pytest.mark.parametrize(
    "config",
    [
        None,
        SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1"}),
        SimpleNamespace(id2label={0: "not_entailment", 1: "entailment"}),
    ],
)
def test_nli_rejects_unresolved_or_non_entailing_generic_label(config: Any) -> None:
    scorer = _scorer_with_pipeline([{"label": "LABEL_0", "score": 0.99}], config)

    with pytest.raises(RuntimeError, match="unresolved 'LABEL_n'"):
        scorer._nli_entailment("evidence", "claim")


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(id2label={"0": "ENTAILMENT", "1": "neutral"}),
        SimpleNamespace(label2id={"entailment": 0, "neutral": 1}),
        SimpleNamespace(
            id2label={0: "LABEL_0", 1: "LABEL_1"},
            label2id={"entailment": 0, "neutral": 1},
        ),
    ],
)
def test_nli_resolves_generic_label_only_from_explicit_model_config(config: Any) -> None:
    scorer = _scorer_with_pipeline([{"label": "LABEL_0", "score": 0.74}], config)

    assert scorer._nli_entailment("evidence", "claim") == pytest.approx(0.74)


def test_nli_rejects_conflicting_model_config_mapping() -> None:
    config = SimpleNamespace(
        id2label={0: "contradiction", 1: "entailment"},
        label2id={"entailment": 0, "neutral": 1},
    )
    scorer = _scorer_with_pipeline([{"label": "LABEL_0", "score": 0.99}], config)

    with pytest.raises(RuntimeError, match="conflicting labels"):
        scorer._nli_entailment("evidence", "claim")


@pytest.mark.parametrize("output_label", ["LABEL_0", "entailment"])
def test_nli_rejects_multiple_entailment_class_ids(output_label: str) -> None:
    config = SimpleNamespace(id2label={0: "entailment", 1: "ENTAILMENT", 2: "neutral"})
    scorer = _scorer_with_pipeline([{"label": output_label, "score": 0.99}], config)

    with pytest.raises(RuntimeError, match="multiple entailment class IDs"):
        scorer._nli_entailment("evidence", "claim")


def test_direct_nli_call_keeps_lazy_loading_and_structured_text_pair_input() -> None:
    scorer = PassageScorer("fake-nli", mock_mode=False)
    pipeline = FakeNliPipeline([{"label": "entailment", "score": 0.66}])
    load_calls = 0

    def fake_load() -> None:
        nonlocal load_calls
        load_calls += 1
        scorer._nli_pipeline = pipeline

    scorer._load_nli = fake_load  # type: ignore[method-assign]

    assert scorer._nli_entailment("evidence", "claim") == pytest.approx(0.66)
    assert load_calls == 1
    assert pipeline.calls == [
        {
            "payload": {"text": "evidence", "text_pair": "claim"},
            "top_k": None,
            "truncation": "only_first",
            "max_length": 512,
        }
    ]


def _mock_passages() -> List[Dict[str, Any]]:
    return [
        {"id": "passage-a", "text": "alpha", "combined_score": 0.1},
        {"id": "passage-b", "text": "beta", "combined_score": 0.5},
        {"id": 3, "text": "gamma", "combined_score": 0.9},
    ]


def test_mock_cross_encoder_is_non_mutating_and_permutation_invariant() -> None:
    scorer = PassageScorer("mock", mock_mode=True, cross_encoder_model="mock")
    passages = _mock_passages()
    original = deepcopy(passages)

    first = scorer._cross_encoder_rerank("same query", passages)
    permuted = scorer._cross_encoder_rerank("same query", list(reversed(passages)))
    repeated = scorer._cross_encoder_rerank("same query", first)

    assert passages == original
    assert first is not passages
    assert all(output is not source for output in first for source in passages)
    assert first == permuted == repeated


def test_mock_cross_encoder_seed_depends_on_query_and_stable_passage_id() -> None:
    scorer = PassageScorer("mock", mock_mode=True, cross_encoder_model="mock")
    first = scorer._cross_encoder_rerank("query-a", [{"id": "stable-id", "text": "first text"}])[0]
    changed_text = scorer._cross_encoder_rerank(
        "query-a", [{"id": "stable-id", "text": "different text"}]
    )[0]
    changed_query = scorer._cross_encoder_rerank(
        "query-b", [{"id": "stable-id", "text": "first text"}]
    )[0]

    assert first["cross_encoder_score"] == changed_text["cross_encoder_score"]
    assert first["cross_encoder_score"] != changed_query["cross_encoder_score"]


@pytest.mark.parametrize(
    "passages,match",
    [
        ([{"text": "missing id"}], "stable string or integer 'id'"),
        ([{"id": "  ", "text": "blank id"}], "non-empty and trimmed"),
        ([{"id": "duplicate"}, {"id": "duplicate"}], "duplicate passage id"),
    ],
)
def test_mock_cross_encoder_rejects_missing_invalid_or_duplicate_ids(
    passages: List[Dict[str, Any]], match: str
) -> None:
    scorer = PassageScorer("mock", mock_mode=True, cross_encoder_model="mock")

    with pytest.raises(ValueError, match=match):
        scorer._cross_encoder_rerank("query", passages)


def test_public_mock_reranking_does_not_mutate_input_passages() -> None:
    scorer = PassageScorer("mock", mock_mode=True, cross_encoder_model="mock")
    passages = _mock_passages()
    original = deepcopy(passages)

    scored = scorer.score_passages("query", passages)

    assert passages == original
    assert all("final_score" in passage for passage in scored)
