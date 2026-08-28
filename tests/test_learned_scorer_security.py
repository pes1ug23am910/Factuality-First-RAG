"""Security and integrity boundaries for executable learned-scorer artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import factuality_rag.scorer.learned_scorer as learned_module
import factuality_rag.pipeline.orchestrator as orchestrator
from factuality_rag.pipeline.orchestrator import (
    Pipeline,
    _load_configured_learned_scorer,
    run_pipeline,
)
from factuality_rag.scorer.learned_scorer import LearnedScorer


def _trained_scorer() -> LearnedScorer:
    scorer = LearnedScorer("logreg")
    scorer.fit(
        [[0.9, 0.6, 0.8], [0.8, 0.5, 0.7], [0.1, 0.1, 0.2], [0.2, 0.2, 0.1]],
        [1, 1, 0, 0],
    )
    return scorer


def _deserialization_forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("pickle.loads must not run before trust and integrity checks pass")


def test_pickle_load_is_denied_by_default_before_read_or_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pickle, "loads", _deserialization_forbidden)

    with pytest.raises(ValueError, match="pickle deserialization is disabled"):
        LearnedScorer.load(
            tmp_path / "missing",
            expected_metadata_sha256="a" * 64,
        )


@pytest.mark.parametrize("unsafe_value", [False, 0, 1, "true", None])
def test_pickle_opt_in_requires_literal_true(
    tmp_path: Path,
    unsafe_value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pickle, "loads", _deserialization_forbidden)

    with pytest.raises(ValueError, match="pickle deserialization is disabled"):
        LearnedScorer.load(
            tmp_path,
            expected_metadata_sha256="a" * 64,
            allow_unsafe_pickle=unsafe_value,  # type: ignore[arg-type]
        )


def test_tampered_metadata_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = _trained_scorer().save(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
    monkeypatch.setattr(pickle, "loads", _deserialization_forbidden)

    with pytest.raises(ValueError, match="metadata.json SHA-256"):
        LearnedScorer.load(
            tmp_path,
            expected_metadata_sha256=digest,
            allow_unsafe_pickle=True,
        )


def test_tampered_model_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = _trained_scorer().save(tmp_path)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    monkeypatch.setattr(pickle, "loads", _deserialization_forbidden)

    with pytest.raises(ValueError, match="byte length"):
        LearnedScorer.load(
            tmp_path,
            expected_metadata_sha256=digest,
            allow_unsafe_pickle=True,
        )


def test_authenticated_metadata_binds_model_type(
    tmp_path: Path,
) -> None:
    _trained_scorer().save(tmp_path)
    wrong_model = pickle.dumps({"not": "a classifier"}, protocol=pickle.HIGHEST_PROTOCOL)
    (tmp_path / "model.pkl").write_bytes(wrong_model)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_sha256"] = hashlib.sha256(wrong_model).hexdigest()
    metadata["model_size_bytes"] = len(wrong_model)
    metadata_bytes = learned_module._canonical_json_bytes(metadata)
    (tmp_path / "metadata.json").write_bytes(metadata_bytes)

    with pytest.raises(ValueError, match="deserialized model type"):
        LearnedScorer.load(
            tmp_path,
            expected_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            allow_unsafe_pickle=True,
        )


def test_incompatible_runtime_metadata_fails_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trained_scorer().save(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["scikit_learn_version"] = "0.0-incompatible"
    metadata_bytes = learned_module._canonical_json_bytes(metadata)
    (tmp_path / "metadata.json").write_bytes(metadata_bytes)
    monkeypatch.setattr(pickle, "loads", _deserialization_forbidden)

    with pytest.raises(ValueError, match="scikit_learn_version"):
        LearnedScorer.load(
            tmp_path,
            expected_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            allow_unsafe_pickle=True,
        )


def test_configured_loader_does_not_silently_fallback(tmp_path: Path) -> None:
    config = {
        "learned_model_path": str(tmp_path / "missing"),
        "learned_model_metadata_sha256": "a" * 64,
        "allow_unsafe_pickle": True,
    }

    with pytest.raises(FileNotFoundError):
        _load_configured_learned_scorer(config)


def test_pipeline_rejects_untrusted_learned_configuration_before_use(tmp_path: Path) -> None:
    config = {
        "seed": 42,
        "models": {},
        "retriever": {"top_k": 1},
        "gating": {"enabled": False},
        "scorer": {
            "use_learned": True,
            "learned_model_path": str(tmp_path / "untrusted"),
            "learned_model_metadata_sha256": "a" * 64,
        },
    }

    with pytest.raises(ValueError, match="pickle deserialization is disabled"):
        Pipeline(config=config, mock_mode=True)


def test_stateless_pipeline_rejects_untrusted_model_even_without_passages(
    tmp_path: Path,
) -> None:
    config = {
        "scorer": {
            "use_learned": True,
            "learned_model_path": str(tmp_path / "untrusted"),
            "learned_model_metadata_sha256": "a" * 64,
        },
    }

    with pytest.raises(ValueError, match="pickle deserialization is disabled"):
        run_pipeline(
            "query",
            k=0,
            gate=False,
            mock_mode=True,
            config=config,
        )


def test_trusted_round_trip_produces_finite_probabilities(tmp_path: Path) -> None:
    scorer = _trained_scorer()
    digest = scorer.save(tmp_path)

    loaded = LearnedScorer.load(
        tmp_path,
        expected_metadata_sha256=digest,
        allow_unsafe_pickle=True,
    )
    probabilities = loaded.predict_proba([[0.7, 0.4, 0.8], [0.1, 0.2, 0.1]])

    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()


def test_stateful_pipeline_reuses_one_authenticated_learned_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_count = 0
    score_count = 0

    class FakeLearnedScorer:
        def score_passages(self, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal score_count
            score_count += 1
            return [{**passage, "learned_score": 0.9} for passage in passages]

    learned = FakeLearnedScorer()

    def load_once(config: dict[str, Any]) -> FakeLearnedScorer:
        nonlocal load_count
        load_count += 1
        return learned

    monkeypatch.setattr(orchestrator, "_load_configured_learned_scorer", load_once)
    config = {
        "seed": 42,
        "models": {},
        "retriever": {"top_k": 1},
        "gating": {"enabled": False},
        "scorer": {"use_learned": True, "score_threshold": 0.4},
    }

    pipeline = Pipeline(config=config, mock_mode=True)
    pipeline.run("first", gate=False)
    pipeline.run("second", gate=False)

    assert load_count == 1
    assert score_count == 2
