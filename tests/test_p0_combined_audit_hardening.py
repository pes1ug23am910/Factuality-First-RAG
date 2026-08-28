"""Adversarial regressions for gating inputs and the retrieval/scoring boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pytest

from factuality_rag.gating.probe import GatingProbe
from factuality_rag.pipeline.orchestrator import (
    _canonicalize_retrieved_passages,
    run_pipeline,
)


def _fail_if_probed(*args: Any, **kwargs: Any) -> np.ndarray:
    raise AssertionError("invalid gating configuration reached model probing")


@pytest.mark.parametrize(
    "kwargs,error_type,match",
    [
        ({"entropy_thresh": True}, TypeError, "entropy_thresh must be a real number"),
        ({"entropy_thresh": "1.2"}, TypeError, "entropy_thresh must be a real number"),
        ({"entropy_thresh": float("nan")}, ValueError, "entropy_thresh must be finite"),
        ({"entropy_thresh": float("inf")}, ValueError, "entropy_thresh must be finite"),
        ({"logit_gap_thresh": False}, TypeError, "logit_gap_thresh must be a real number"),
        ({"logit_gap_thresh": "2.0"}, TypeError, "logit_gap_thresh must be a real number"),
        ({"logit_gap_thresh": float("nan")}, ValueError, "logit_gap_thresh must be finite"),
        ({"logit_gap_thresh": float("-inf")}, ValueError, "logit_gap_thresh must be finite"),
    ],
)
def test_should_retrieve_rejects_non_numeric_or_nonfinite_thresholds_before_probe(
    kwargs: Dict[str, Any],
    error_type: type[Exception],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = GatingProbe("mock", mock_mode=True)
    monkeypatch.setattr(probe, "_get_next_token_logits", _fail_if_probed)
    monkeypatch.setattr(probe, "_get_multi_token_logits", _fail_if_probed)

    with pytest.raises(error_type, match=match):
        probe.should_retrieve("prompt", **kwargs)


@pytest.mark.parametrize(
    "probe_tokens,error_type",
    [
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_should_retrieve_requires_positive_integer_probe_tokens_before_probe(
    probe_tokens: Any,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = GatingProbe("mock", mock_mode=True)
    monkeypatch.setattr(probe, "_get_next_token_logits", _fail_if_probed)
    monkeypatch.setattr(probe, "_get_multi_token_logits", _fail_if_probed)

    with pytest.raises(error_type, match="probe_tokens must be a positive integer"):
        probe.should_retrieve("prompt", probe_tokens=probe_tokens)


@pytest.mark.parametrize(
    "temperature,error_type,match",
    [
        (True, TypeError, "temperature must be a real number"),
        ("1.0", TypeError, "temperature must be a real number"),
        (float("nan"), ValueError, "temperature must be finite"),
        (float("inf"), ValueError, "temperature must be finite"),
        (0.0, ValueError, "temperature must be greater than zero"),
        (-1.0, ValueError, "temperature must be greater than zero"),
    ],
)
def test_should_retrieve_requires_finite_positive_temperature_before_probe(
    temperature: Any,
    error_type: type[Exception],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = GatingProbe("mock", mock_mode=True, temp=temperature)
    monkeypatch.setattr(probe, "_get_next_token_logits", _fail_if_probed)
    monkeypatch.setattr(probe, "_get_multi_token_logits", _fail_if_probed)

    with pytest.raises(error_type, match=match):
        probe.should_retrieve("prompt")


def test_finite_numeric_gating_values_remain_supported() -> None:
    probe = GatingProbe("mock", mock_mode=True, temp=np.float64(1.0))

    decision = probe.should_retrieve(
        "prompt",
        probe_tokens=np.int64(1),
        entropy_thresh=np.float64(1.2),
        logit_gap_thresh=2,
    )

    assert isinstance(decision, bool)


@pytest.mark.parametrize(
    "raw_id,match",
    [
        (True, "invalid boolean id"),
        (False, "invalid boolean id"),
        ("", "non-empty and trimmed"),
        (" passage ", "non-empty and trimmed"),
        (1.5, "trimmed string or non-boolean integer"),
        (None, "trimmed string or non-boolean integer"),
    ],
)
def test_retrieval_boundary_rejects_invalid_passage_ids(raw_id: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _canonicalize_retrieved_passages([{"id": raw_id, "text": "evidence"}])


def test_retrieval_boundary_canonicalizes_without_mutating_retriever_records() -> None:
    retrieved = [
        {"id": 7, "text": "integer id"},
        {"id": "passage-8", "text": "string id"},
    ]
    original = deepcopy(retrieved)

    canonical = _canonicalize_retrieved_passages(retrieved)

    assert retrieved == original
    assert [passage["id"] for passage in canonical] == ["7", "passage-8"]
    assert all(output is not source for output, source in zip(canonical, retrieved))


class StaticRetriever:
    def __init__(self, passages: List[Dict[str, Any]]) -> None:
        self.passages = passages
        self.calls = 0

    def retrieve(self, query: str, *, k: int, rerank: bool) -> List[Dict[str, Any]]:
        self.calls += 1
        return self.passages


class RecordingScorer:
    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls = 0
        self.seen_ids: List[str] = []

    def score_passages(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.calls += 1
        self.seen_ids = [passage["id"] for passage in passages]
        return [{**passage, "final_score": self.score} for passage in passages]


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, query: str, *, context: str) -> str:
        self.calls += 1
        return ""


def test_pipeline_rejects_canonical_id_collisions_before_scoring_or_generation() -> None:
    retrieved = [
        {"id": 1, "text": "integer spelling"},
        {"id": "1", "text": "string spelling"},
    ]
    retriever = StaticRetriever(retrieved)
    scorer = RecordingScorer()
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="duplicate retrieved passage id after canonicalization"):
        run_pipeline(
            "query",
            k=2,
            gate=False,
            config={},
            retriever=retriever,
            scorer=scorer,
            generator=generator,
        )

    assert scorer.calls == 0
    assert generator.calls == 0
    assert [passage["id"] for passage in retrieved] == [1, "1"]


def test_pipeline_scores_and_records_only_canonical_passage_ids() -> None:
    retriever = StaticRetriever([{"id": 7, "text": "evidence"}])
    scorer = RecordingScorer()
    generator = RecordingGenerator()
    info: Dict[str, Any] = {}

    _, trusted, _, _ = run_pipeline(
        "query",
        k=1,
        gate=False,
        config={},
        retriever=retriever,
        scorer=scorer,
        generator=generator,
        info=info,
    )

    assert scorer.seen_ids == ["7"]
    assert [passage["id"] for passage in trusted] == ["7"]
    assert info["scored_passages"] == [{"id": "7", "final_score": 0.8}]
    assert retriever.passages[0]["id"] == 7


def test_invalid_scored_artifact_fails_before_generation_side_effect() -> None:
    retriever = StaticRetriever([{"id": "passage", "text": "evidence"}])
    scorer = RecordingScorer(score=float("nan"))
    generator = RecordingGenerator()
    info: Dict[str, Any] = {}

    with pytest.raises(ValueError, match="final_score must be finite"):
        run_pipeline(
            "query",
            k=1,
            gate=False,
            config={},
            retriever=retriever,
            scorer=scorer,
            generator=generator,
            info=info,
        )

    assert scorer.calls == 1
    assert generator.calls == 0
    assert info == {}


def test_explicit_scorer_bypass_passes_all_retrieved_passages_without_nli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.scorer.passage as passage_module

    class ConstructionForbidden:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("scorer bypass constructed an NLI scorer")

    monkeypatch.setattr(passage_module, "PassageScorer", ConstructionForbidden)
    retriever = StaticRetriever(
        [
            {"id": "p-1", "text": "first", "combined_score": 0.1},
            {"id": "p-2", "text": "second", "combined_score": 0.2},
        ]
    )
    generator = RecordingGenerator()
    info: Dict[str, Any] = {}

    _, trusted, provenance, confidence = run_pipeline(
        "query",
        k=2,
        gate=False,
        score_threshold=0.99,
        config={"scorer": {"enabled": False}},
        retriever=retriever,
        generator=generator,
        info=info,
    )

    assert [passage["id"] for passage in trusted] == ["p-1", "p-2"]
    assert provenance == {}
    assert confidence == "low"
    assert generator.calls == 1
    assert info["scorer_enabled"] is False
    assert "scored_passages" not in info


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_scorer_enabled_requires_exact_boolean(value: Any) -> None:
    with pytest.raises(TypeError, match="scorer.enabled must be exactly bool"):
        run_pipeline("query", config={"scorer": {"enabled": value}}, mock_mode=True)


def test_scorer_bypass_rejects_learned_scorer_mode() -> None:
    with pytest.raises(ValueError, match="use_learned cannot be enabled"):
        run_pipeline(
            "query",
            config={"scorer": {"enabled": False, "use_learned": True}},
            mock_mode=True,
        )


@pytest.mark.parametrize("value", [True, "0.4", None])
def test_score_threshold_rejects_non_numeric_values_before_pipeline_work(value: Any) -> None:
    with pytest.raises(TypeError, match="score_threshold must be a real number"):
        run_pipeline("query", score_threshold=value, mock_mode=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_score_threshold_rejects_out_of_range_or_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        run_pipeline("query", score_threshold=value, mock_mode=True)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_retriever_rerank_requires_exact_boolean(value: Any) -> None:
    with pytest.raises(TypeError, match="retriever.rerank must be exactly bool"):
        run_pipeline(
            "query",
            config={"retriever": {"rerank": value}},
            mock_mode=True,
        )


def test_gate_probes_the_generator_no_context_prompt_not_the_raw_query() -> None:
    class RecordingProbe:
        def __init__(self) -> None:
            self.prompt = ""

        def should_retrieve(self, prompt: str, **kwargs: Any) -> bool:
            self.prompt = prompt
            return False

    probe = RecordingProbe()
    run_pipeline(
        "What is DNA?",
        gate=True,
        config={"scorer": {"enabled": False}},
        mock_mode=True,
        probe=probe,
    )

    assert probe.prompt != "What is DNA?"
    assert "[INST]" in probe.prompt
    assert "Question: What is DNA?" in probe.prompt
