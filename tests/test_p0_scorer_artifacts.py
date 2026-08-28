"""Artifact-level regressions for independent scorer evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from factuality_rag.experiment_runner import run
from factuality_rag.pipeline.orchestrator import _scored_passage_artifact
from scripts import analyze_scorer


class _ArtifactPipeline:
    """Model-free producer fixture with scored candidates above and below threshold."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self, query: str, *, info: Dict[str, Any], **kwargs: Any) -> tuple:
        info.update(
            {
                "retrieval_triggered": True,
                "gating_enabled": False,
                "scored_passages": [
                    {"id": "passage-relevant", "final_score": 0.9},
                    {"id": "passage-negative", "final_score": 0.2},
                    {"id": "passage-unjudged", "final_score": 0.1},
                ],
            }
        )
        trusted = [
            {
                "id": "passage-relevant",
                "text": "Paris is the capital of France.",
                "final_score": 0.9,
                "nli_score": 0.8,
            }
        ]
        return "Paris.", trusted, {"0": ["passage-relevant"]}, "high"


def _write_predictions(path: Path, records: List[Dict[str, Any]]) -> str:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bound_judgments(
    path: Path,
    predictions_sha256: str,
    judgments: List[Dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": analyze_scorer.JUDGMENTS_SCHEMA,
                "independent_of_scorer": True,
                "predictions_sha256": predictions_sha256,
                "label_source": "fixture/manual-relevance-annotations",
                "label_source_revision": hashlib.sha256(
                    b"fixture/manual-relevance-annotations/v1"
                ).hexdigest(),
                "judgments": judgments,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_pipeline_score_artifact_is_json_safe_and_minimal() -> None:
    artifact = _scored_passage_artifact(
        [
            {
                "id": "passage-1",
                "text": "annotation-visible evidence text",
                "final_score": 1,
                "internal_prompt": "must never leave the pipeline",
                "model_state": object(),
            }
        ]
    )

    assert artifact == [{"id": "passage-1", "final_score": 1.0}]
    assert json.loads(json.dumps(artifact, allow_nan=False)) == artifact


def test_runner_artifact_flows_to_independent_scorer_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.pipeline.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "Pipeline", _ArtifactPipeline)
    result = run(
        {"seed": 17, "gating": {"enabled": False}, "retriever": {"top_k": 3}},
        queries=["What is the capital of France?"],
        runs_dir=str(tmp_path / "runs"),
        mock_mode=False,
    )

    prediction = result["predictions"][0]
    assert prediction["scored_passages"] == [
        {"id": "passage-relevant", "final_score": 0.9},
        {"id": "passage-negative", "final_score": 0.2},
        {"id": "passage-unjudged", "final_score": 0.1},
    ]
    assert prediction["trusted_passages"][0]["text"] == "Paris is the capital of France."
    assert prediction["trusted_passages"][0]["nli_score"] == 0.8
    assert all(
        set(candidate) == {"id", "final_score"} for candidate in prediction["scored_passages"]
    )

    predictions_path = Path(result["run_dir"]) / "predictions.jsonl"
    predictions_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    judgments_path = tmp_path / "judgments.json"
    _write_bound_judgments(
        judgments_path,
        predictions_sha256,
        [
            {
                "example_id": prediction["example_id"],
                "judged_passage_ids": ["passage-relevant", "passage-negative"],
                "relevant_passage_ids": ["passage-relevant"],
            }
        ],
    )
    output_path = tmp_path / "scorer-analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_scorer.py",
            "--predictions",
            str(predictions_path),
            "--judgments",
            str(judgments_path),
            "--sample",
            "1",
            "--seed",
            "17",
            "--output",
            str(output_path),
        ],
    )

    analyze_scorer.main()

    analysis = json.loads(output_path.read_text(encoding="utf-8"))
    assert analysis["status"] == "valid_independent_judgments"
    assert analysis["provenance"]["predictions_sha256"] == predictions_sha256
    assert analysis["provenance"]["predictions_binding_verified"] is True
    assert analysis["metrics"]["n_passages"] == 2
    assert analysis["candidate_accounting"]["n_excluded_unjudged_candidates_selected_examples"] == 1
    assert analysis["provenance"]["selection"] == {
        "method": "sha256_rank_v1",
        "seed": 17,
        "requested_sample_limit": 1,
        "available_examples": 1,
        "selected_example_ids": [prediction["example_id"]],
    }


def test_cross_run_predictions_hash_mismatch_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_predictions = tmp_path / "run-a.jsonl"
    second_predictions = tmp_path / "run-b.jsonl"
    first_sha256 = _write_predictions(
        first_predictions,
        [
            {
                "example_id": "example-1",
                "scored_passages": [
                    {"id": "relevant", "final_score": 0.8},
                    {"id": "negative", "final_score": 0.2},
                ],
            }
        ],
    )
    _write_predictions(
        second_predictions,
        [
            {
                "example_id": "example-1",
                "scored_passages": [
                    {"id": "relevant", "final_score": 0.7},
                    {"id": "negative", "final_score": 0.3},
                ],
            }
        ],
    )
    judgments_path = tmp_path / "judgments.json"
    _write_bound_judgments(
        judgments_path,
        first_sha256,
        [
            {
                "example_id": "example-1",
                "judged_passage_ids": ["relevant", "negative"],
                "relevant_passage_ids": ["relevant"],
            }
        ],
    )
    output_path = tmp_path / "should-not-exist.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_scorer.py",
            "--predictions",
            str(second_predictions),
            "--judgments",
            str(judgments_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="does not match"):
        analyze_scorer.main()
    assert not output_path.exists()


def test_relevant_ids_must_be_subset_of_judged_ids(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_sha256 = _write_predictions(
        predictions_path,
        [
            {
                "example_id": "example-1",
                "scored_passages": [
                    {"id": "relevant", "final_score": 0.8},
                    {"id": "negative", "final_score": 0.2},
                ],
            }
        ],
    )
    judgments_path = tmp_path / "judgments.json"
    _write_bound_judgments(
        judgments_path,
        predictions_sha256,
        [
            {
                "example_id": "example-1",
                "judged_passage_ids": ["negative"],
                "relevant_passage_ids": ["relevant"],
            }
        ],
    )

    with pytest.raises(ValueError, match="subset"):
        analyze_scorer._load_independent_judgments(
            judgments_path, expected_predictions_sha256=predictions_sha256
        )


def test_unknown_judged_passage_id_is_rejected(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_sha256 = _write_predictions(
        predictions_path,
        [
            {
                "example_id": "example-1",
                "scored_passages": [
                    {"id": "relevant", "final_score": 0.8},
                    {"id": "negative", "final_score": 0.2},
                ],
            }
        ],
    )
    judgments_path = tmp_path / "judgments.json"
    _write_bound_judgments(
        judgments_path,
        predictions_sha256,
        [
            {
                "example_id": "example-1",
                "judged_passage_ids": ["relevant", "absent"],
                "relevant_passage_ids": ["relevant"],
            }
        ],
    )
    predictions, actual_sha256 = analyze_scorer._read_predictions(predictions_path)
    judgments, _ = analyze_scorer._load_independent_judgments(
        judgments_path, expected_predictions_sha256=actual_sha256
    )

    with pytest.raises(ValueError, match="absent from scored_passages"):
        analyze_scorer._collect_independently_labeled_scores(predictions, judgments, 1)


def test_duplicate_candidate_and_judged_ids_are_rejected(tmp_path: Path) -> None:
    duplicate_candidates = [
        {
            "example_id": "example-1",
            "scored_passages": [
                {"id": "same", "final_score": 0.8},
                {"id": "same", "final_score": 0.2},
            ],
        }
    ]
    judgment = analyze_scorer.IndependentJudgment({"same", "negative"}, {"same"})
    with pytest.raises(ValueError, match="duplicate passage ID"):
        analyze_scorer._collect_independently_labeled_scores(
            duplicate_candidates, {"example-1": judgment}, 1
        )

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_sha256 = _write_predictions(
        predictions_path,
        [
            {
                "example_id": "example-1",
                "scored_passages": [
                    {"id": "relevant", "final_score": 0.8},
                    {"id": "negative", "final_score": 0.2},
                ],
            }
        ],
    )
    judgments_path = tmp_path / "judgments.json"
    _write_bound_judgments(
        judgments_path,
        predictions_sha256,
        [
            {
                "example_id": "example-1",
                "judged_passage_ids": ["relevant", "relevant"],
                "relevant_passage_ids": ["relevant"],
            }
        ],
    )
    with pytest.raises(ValueError, match="duplicate judged passage IDs"):
        analyze_scorer._load_independent_judgments(
            judgments_path, expected_predictions_sha256=predictions_sha256
        )


@pytest.mark.parametrize(
    "artifact_name,raw",
    [
        (
            "predictions",
            '{"example_id":"first","example_id":"second","scored_passages":[]}\n',
        ),
        ("judgments", '{"schema":"first","schema":"second"}'),
    ],
)
def test_strict_parsers_reject_duplicate_object_keys(
    tmp_path: Path, artifact_name: str, raw: str
) -> None:
    path = tmp_path / f"{artifact_name}.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        if artifact_name == "predictions":
            analyze_scorer._read_predictions(path)
        else:
            analyze_scorer._read_json(path, artifact_name)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.parametrize("artifact_name", ["predictions", "judgments"])
def test_strict_parsers_reject_non_finite_json_numbers(
    tmp_path: Path, artifact_name: str, token: str
) -> None:
    path = tmp_path / f"{artifact_name}.json"
    if artifact_name == "predictions":
        raw = (
            f'{{"example_id":"example-1","scored_passages":[{{"id":"p","final_score":{token}}}]}}\n'
        )
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text(f'{{"poison":{token}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON number"):
        if artifact_name == "predictions":
            analyze_scorer._read_predictions(path)
        else:
            analyze_scorer._read_json(path, artifact_name)


def test_predictions_jsonl_uses_lf_only_record_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"example_id":"first"}\r{"example_id":"second"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        analyze_scorer._read_predictions(path)


def test_consumer_rejects_extra_scored_candidate_fields() -> None:
    predictions = [
        {
            "example_id": "example-1",
            "scored_passages": [
                {"id": "relevant", "final_score": 0.8, "text": "must not leak"},
                {"id": "negative", "final_score": 0.2},
            ],
        }
    ]
    judgments = {
        "example-1": analyze_scorer.IndependentJudgment({"relevant", "negative"}, {"relevant"})
    }

    with pytest.raises(ValueError, match="exactly id and final_score"):
        analyze_scorer._collect_independently_labeled_scores(predictions, judgments, 1)


def test_seeded_sampling_is_order_invariant_and_accounts_for_all_candidates() -> None:
    predictions: List[Dict[str, Any]] = []
    judgments: Dict[str, analyze_scorer.IndependentJudgment] = {}
    for index in range(5):
        example_id = f"example-{index}"
        relevant_id = f"relevant-{index}"
        negative_id = f"negative-{index}"
        predictions.append(
            {
                "example_id": example_id,
                "scored_passages": [
                    {"id": relevant_id, "final_score": 0.9 - index * 0.01},
                    {"id": negative_id, "final_score": 0.1 + index * 0.01},
                    {"id": f"unjudged-{index}", "final_score": 0.5},
                ],
            }
        )
        judgments[example_id] = analyze_scorer.IndependentJudgment(
            {relevant_id, negative_id}, {relevant_id}
        )

    first_accounting: Dict[str, Any] = {}
    first_selection: Dict[str, Any] = {}
    first = analyze_scorer._collect_independently_labeled_scores(
        predictions,
        judgments,
        2,
        seed=101,
        accounting=first_accounting,
        selection=first_selection,
    )
    second_accounting: Dict[str, Any] = {}
    second_selection: Dict[str, Any] = {}
    second = analyze_scorer._collect_independently_labeled_scores(
        list(reversed(predictions)),
        judgments,
        2,
        seed=101,
        accounting=second_accounting,
        selection=second_selection,
    )

    assert first == second
    assert first_selection == second_selection
    assert first_selection["method"] == "sha256_rank_v1"
    assert len(first_selection["selected_example_ids"]) == 2
    assert (
        first_accounting
        == second_accounting
        == {
            "n_candidates_all_examples": 15,
            "n_judged_candidates_all_examples": 10,
            "n_excluded_unjudged_candidates_all_examples": 5,
            "n_candidates_selected_examples": 6,
            "n_judged_candidates_selected_examples": 4,
            "n_excluded_unjudged_candidates_selected_examples": 2,
        }
    )


def test_existing_output_is_never_overwritten_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "existing.json"
    original = b'{"status":"stale-but-user-owned"}\n'
    output_path.write_bytes(original)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_scorer.py", "--mock", "--sample", "1", "--output", str(output_path)],
    )

    with pytest.raises(FileExistsError, match="refusing to replace"):
        analyze_scorer.main()
    assert output_path.read_bytes() == original


def test_mock_analysis_is_unambiguously_non_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "mock.json"
    config_path = tmp_path / "mock.yaml"
    config_path.write_text("models: {}\nscorer: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_scorer.py",
            "--mock",
            "--sample",
            "1",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    analyze_scorer.main()

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["status"] == "synthetic_mock_smoke_test_only_non_claim"
    assert output["claim_status"] == "NOT_ELIGIBLE_FOR_EMPIRICAL_OR_PERFORMANCE_CLAIMS"
    assert output["provenance"]["contains_empirical_evidence"] is False
    assert output["threshold_use"] == "synthetic_smoke_test_only_not_a_threshold_estimate"
