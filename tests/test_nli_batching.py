"""Focused tests for safe full-passage NLI batching and scorer config wiring."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from factuality_rag.pipeline.orchestrator import Pipeline, run_pipeline
from factuality_rag.scorer.passage import PassageScorer


def _class_scores(entailment_score: float) -> List[Dict[str, Any]]:
    return [
        {"label": "contradiction", "score": 0.1},
        {"label": "entailment", "score": entailment_score},
        {"label": "neutral", "score": 0.2},
    ]


class SpyNliPipeline:
    """Record the exact inference surface used by ``PassageScorer``."""

    def __init__(
        self,
        output: Any,
        *,
        tokenizer: Any = None,
        max_position_embeddings: Any = None,
    ) -> None:
        self.output = output
        self.calls: List[Dict[str, Any]] = []
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.model = (
            SimpleNamespace(config=SimpleNamespace(max_position_embeddings=max_position_embeddings))
            if max_position_embeddings is not None
            else None
        )

    def __call__(self, payload: Any, **kwargs: Any) -> Any:
        self.calls.append({"payload": payload, "kwargs": kwargs})
        return self.output


class WhitespaceTokenizer:
    """Small tokenizer double that exposes the real length-policy surface."""

    def __init__(self, model_max_length: Any, pair_special_tokens: Any = 3) -> None:
        self.model_max_length = model_max_length
        self.pair_special_tokens = pair_special_tokens
        self.encode_calls: List[Dict[str, Any]] = []

    def encode(self, text: str, **kwargs: Any) -> List[int]:
        self.encode_calls.append({"text": text, "kwargs": kwargs})
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, *, pair: bool) -> Any:
        assert pair is True
        return self.pair_special_tokens


def _passages() -> List[Dict[str, Any]]:
    return [
        {"id": "p-1", "text": "first evidence", "combined_score": 0.2},
        {"id": "p-2", "text": "second evidence", "combined_score": 0.8},
        {"id": "p-3", "text": "third evidence", "combined_score": 0.5},
    ]


def test_full_passage_nli_uses_one_pipeline_call_and_configured_batch_size() -> None:
    pipeline = SpyNliPipeline([_class_scores(0.71), _class_scores(0.82), _class_scores(0.93)])
    scorer = PassageScorer("fake-nli", nli_batch_size=2)
    scorer._nli_pipeline = pipeline

    scored = scorer.score_passages("claim", _passages())

    assert [passage["nli_score"] for passage in scored] == pytest.approx([0.71, 0.82, 0.93])
    assert pipeline.calls == [
        {
            "payload": [
                {"text": "first evidence", "text_pair": "claim"},
                {"text": "second evidence", "text_pair": "claim"},
                {"text": "third evidence", "text_pair": "claim"},
            ],
            "kwargs": {
                "top_k": None,
                "batch_size": 2,
                "truncation": "only_first",
                "max_length": 512,
            },
        }
    ]


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ({"label": "entailment", "score": 0.8}, "unsupported batched result type"),
        ([[_class_scores(0.8)[1]]], "exactly one result per input"),
        ([[], _class_scores(0.8)], "unsupported result shape"),
        (
            [
                [{"label": "not_entailment", "score": 0.99}],
                _class_scores(0.8),
            ],
            "exactly one unambiguous entailment",
        ),
    ],
)
def test_full_passage_nli_rejects_bad_batch_shapes_and_labels(
    output: Any,
    match: str,
) -> None:
    pipeline = SpyNliPipeline(output)
    scorer = PassageScorer("fake-nli")
    scorer._nli_pipeline = pipeline

    with pytest.raises(RuntimeError, match=match):
        scorer.score_passages("claim", _passages()[:2])

    assert len(pipeline.calls) == 1


def test_single_pair_helper_keeps_legacy_call_shape_and_validation() -> None:
    pipeline = SpyNliPipeline(_class_scores(0.66))
    scorer = PassageScorer("fake-nli", nli_batch_size=3)
    scorer._nli_pipeline = pipeline
    long_premise = "evidence " * 600

    assert scorer._nli_entailment(long_premise, "claim") == pytest.approx(0.66)
    assert pipeline.calls == [
        {
            "payload": {"text": long_premise, "text_pair": "claim"},
            "kwargs": {
                "top_k": None,
                "truncation": "only_first",
                "max_length": 512,
            },
        }
    ]


def test_dynamic_test_mock_without_declared_tokenizer_keeps_legacy_cap() -> None:
    pipeline = MagicMock(return_value=_class_scores(0.66))
    scorer = PassageScorer("fake-nli")
    scorer._nli_pipeline = pipeline

    assert scorer._nli_entailment("evidence", "claim") == pytest.approx(0.66)
    pipeline.assert_called_once_with(
        {"text": "evidence", "text_pair": "claim"},
        top_k=None,
        truncation="only_first",
        max_length=512,
    )


def test_overlong_hypothesis_fails_before_pipeline_without_truncating_claim() -> None:
    tokenizer = WhitespaceTokenizer(model_max_length=8, pair_special_tokens=3)
    pipeline = SpyNliPipeline(_class_scores(0.66), tokenizer=tokenizer)
    scorer = PassageScorer("fake-nli")
    scorer._nli_pipeline = pipeline

    with pytest.raises(ValueError, match=r"5 hypothesis tokens.*limit of 8"):
        scorer._nli_entailment("evidence", "one two three four five")

    assert pipeline.calls == []
    assert tokenizer.encode_calls == [
        {
            "text": "one two three four five",
            "kwargs": {"add_special_tokens": False, "truncation": False},
        }
    ]


@pytest.mark.parametrize(
    ("tokenizer_limit", "model_limit", "expected_limit"),
    [
        (64, 96, 64),
        (96, 48, 48),
    ],
)
def test_custom_smaller_tokenizer_or_model_limit_controls_premise_only_truncation(
    tokenizer_limit: int,
    model_limit: int,
    expected_limit: int,
) -> None:
    tokenizer = WhitespaceTokenizer(model_max_length=tokenizer_limit)
    pipeline = SpyNliPipeline(
        [_class_scores(0.71), _class_scores(0.82)],
        tokenizer=tokenizer,
        max_position_embeddings=model_limit,
    )
    scorer = PassageScorer("fake-nli", nli_batch_size=2)
    scorer._nli_pipeline = pipeline
    long_premise = "evidence " * 600

    scores = scorer._batch_nli_entailment(
        [(long_premise, "intact claim"), (long_premise, "intact claim")]
    )

    assert scores == pytest.approx([0.71, 0.82])
    assert pipeline.calls[0]["kwargs"] == {
        "top_k": None,
        "batch_size": 2,
        "truncation": "only_first",
        "max_length": expected_limit,
    }
    assert tokenizer.encode_calls == [
        {
            "text": "intact claim",
            "kwargs": {"add_special_tokens": False, "truncation": False},
        }
    ]


@pytest.mark.parametrize(
    ("tokenizer_limit", "model_limit", "match"),
    [
        (0, 64, "tokenizer.model_max_length"),
        (64, 0, "model.config.max_position_embeddings"),
    ],
)
def test_invalid_advertised_nli_limit_fails_closed(
    tokenizer_limit: int,
    model_limit: int,
    match: str,
) -> None:
    tokenizer = WhitespaceTokenizer(model_max_length=tokenizer_limit)
    pipeline = SpyNliPipeline(
        _class_scores(0.66),
        tokenizer=tokenizer,
        max_position_embeddings=model_limit,
    )
    scorer = PassageScorer("fake-nli")
    scorer._nli_pipeline = pipeline

    with pytest.raises(RuntimeError, match=match):
        scorer._nli_entailment("evidence", "claim")

    assert pipeline.calls == []


def test_singleton_batch_accepts_flat_all_labels_output() -> None:
    pipeline = SpyNliPipeline(_class_scores(0.64))
    scorer = PassageScorer("fake-nli")
    scorer._nli_pipeline = pipeline

    scored = scorer.score_passages("claim", _passages()[:1])

    assert scored[0]["nli_score"] == pytest.approx(0.64)
    assert pipeline.calls[0]["kwargs"] == {
        "top_k": None,
        "batch_size": 8,
        "truncation": "only_first",
        "max_length": 512,
    }


def test_sentence_mode_remains_unbatched() -> None:
    pipeline = SpyNliPipeline(_class_scores(0.75))
    scorer = PassageScorer("fake-nli", nli_mode="sentence", nli_batch_size=2)
    scorer._nli_pipeline = pipeline

    scorer.score_passages(
        "claim",
        [{"id": "p-1", "text": "First sentence. Second sentence.", "combined_score": 0.5}],
    )

    assert [call["payload"] for call in pipeline.calls] == [
        {"text": "First sentence", "text_pair": "claim"},
        {"text": "Second sentence", "text_pair": "claim"},
    ]
    assert all(
        call["kwargs"]
        == {
            "top_k": None,
            "truncation": "only_first",
            "max_length": 512,
        }
        for call in pipeline.calls
    )


@pytest.mark.parametrize("value", [True, 1.5, "8", None])
def test_nli_batch_size_requires_an_integer(value: Any) -> None:
    with pytest.raises(TypeError, match="nli_batch_size"):
        PassageScorer(mock_mode=True, nli_batch_size=value)


@pytest.mark.parametrize("value", [0, -1])
def test_nli_batch_size_requires_a_positive_value(value: int) -> None:
    with pytest.raises(ValueError, match="nli_batch_size"):
        PassageScorer(mock_mode=True, nli_batch_size=value)


def test_nli_loader_propagates_configured_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    loaded_pipeline = object()

    def fake_pipeline(task: str, **kwargs: Any) -> object:
        calls.append({"task": task, "kwargs": kwargs})
        return loaded_pipeline

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    scorer = PassageScorer("fake-nli", device="cuda:1")

    scorer._load_nli()

    assert scorer._nli_pipeline is loaded_pipeline
    assert calls == [
        {
            "task": "text-classification",
            "kwargs": {"model": "fake-nli", "device": "cuda:1"},
        }
    ]


@pytest.mark.parametrize(
    ("scorer_config", "expected_device", "expected_batch_size"),
    [
        ({}, "cpu", 8),
        ({"device": "cuda:1", "nli_batch_size": 3}, "cuda:1", 3),
    ],
)
def test_both_pipeline_construction_paths_wire_scorer_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    scorer_config: Dict[str, Any],
    expected_device: str,
    expected_batch_size: int,
) -> None:
    import factuality_rag.scorer.passage as passage_module

    construction_calls: List[Dict[str, Any]] = []

    class ScorerConstructionSpy:
        def __init__(self, **kwargs: Any) -> None:
            construction_calls.append(kwargs)

    monkeypatch.setattr(passage_module, "PassageScorer", ScorerConstructionSpy)
    config: Dict[str, Any] = {
        "retriever": {"top_k": 0},
        "scorer": scorer_config,
    }

    run_pipeline("query", k=0, gate=False, config=config, mock_mode=True)
    Pipeline(config=config, mock_mode=True)

    assert len(construction_calls) == 2
    for call in construction_calls:
        assert call["device"] == expected_device
        assert call["nli_batch_size"] == expected_batch_size
