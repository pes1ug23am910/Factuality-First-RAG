"""Publication-boundary tests for aggregation and paired bootstrap."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, cast

import numpy as np
import pytest

from factuality_rag.data.splits import build_group_disjoint_split
from factuality_rag.reproducibility import (
    build_run_manifest,
    seal_manifest,
    sha256_file,
    sha256_json,
    sha256_ordered_ids,
    verify_manifest,
    write_immutable_json,
)
from scripts.aggregate_results import (
    _load_publication_manifest,
    load_run_bundle,
    load_run_metrics,
    main as aggregate_main,
)
from scripts.bootstrap_test import (
    _resolve_references,
    _validate_publication_predictions,
    compute_paired_scores,
    main as bootstrap_main,
    pair_predictions,
    paired_bootstrap_test,
)

_SCORER_ID = "factuality-rag.production-scorer"
_SCORER_REVISION = "c" * 40
_MODEL_REVISION = "d" * 40


def _prediction(example_id: str, query: str, answer: str, reference: object) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "input": query,
        "answer": answer,
        "reference": reference,
    }


def _registered_fixture_path() -> Path:
    path = Path(__file__).resolve().parent / "data" / "evaluator_sanity_v1.json"
    if not path.is_file():
        raise FileNotFoundError("registered evaluator fixture is unavailable")
    return path


def _registered_fixture() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(_registered_fixture_path().read_text(encoding="utf-8")),
    )


def _registered_results() -> list[dict[str, Any]]:
    return [
        {"id": case["id"], **deepcopy(case["expected"])} for case in _registered_fixture()["cases"]
    ]


def _fixture_split() -> dict[str, Any]:
    examples = [
        {
            "example_id": f"fixture:{index:02d}",
            "family_ids": [f"family:{index:02d}"],
            "source": "fixture",
        }
        for index in range(15)
    ]
    return cast(
        dict[str, Any],
        build_group_disjoint_split(
            examples,
            ratios={"train": 0.6, "tuning": 0.2, "sealed_final": 0.2},
            seed=42,
        ),
    )


def _selected_final_ids(count: int = 1) -> list[str]:
    return list(_fixture_split()["partitions"]["sealed_final"]["example_ids"][:count])


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _build_bound_run_manifest(run_dir: Path, example_ids: list[str]) -> dict[str, Any]:
    """Create the real files that the publication validator hashes and parses."""
    split = _fixture_split()
    final_ids = set(split["partitions"]["sealed_final"]["example_ids"])
    if not set(example_ids) <= final_ids:
        raise ValueError("test prediction IDs must come from the real sealed-final fixture")

    config = {"seed": 42, "route": "R1"}
    config_path = run_dir / "configs" / "final.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    dataset_manifest = seal_manifest(
        {
            "schema": "factuality-rag.dataset-manifest.v1",
            "dataset_id": "fixture-dataset",
            "source_snapshot_sha256": split["source"]["source_snapshot_sha256"],
            "example_count": split["source"]["example_count"],
        }
    )
    dataset_path = run_dir / "manifests" / "dataset.json"
    split_path = run_dir / "manifests" / "split.json"
    _write_json(dataset_path, dataset_manifest)
    _write_json(split_path, split)

    corpus_manifest = seal_manifest(
        {
            "schema": "factuality-rag.corpus-manifest.v1",
            "corpus_id": "fixture-corpus",
            "corpus_snapshot_sha256": "e" * 64,
            "passage_count": 10_000,
        }
    )
    corpus_path = run_dir / "manifests" / "corpus.json"
    _write_json(corpus_path, corpus_manifest)
    index_manifest = seal_manifest(
        {
            "schema": "factuality-rag.index-manifest.v1",
            "index_id": "fixture-index",
            "corpus_manifest_sha256": sha256_file(corpus_path),
            "corpus_snapshot_sha256": corpus_manifest["corpus_snapshot_sha256"],
            "exact_indexed_passage_count": corpus_manifest["passage_count"],
        }
    )
    index_path = run_dir / "manifests" / "index.json"
    _write_json(index_path, index_manifest)

    fixture = _registered_fixture()
    fixture_path = run_dir / "fixtures" / "evaluator_sanity_v1.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_bytes(_registered_fixture_path().read_bytes())
    fixture_report = {
        "passed": True,
        "production_gate_passed": True,
        "atol": 1e-12,
        "scorer_id": _SCORER_ID,
        "scorer_revision": _SCORER_REVISION,
        "schema_version": "evaluator-sanity-v1",
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_content_sha256": sha256_json(fixture),
        "case_count": len(fixture["cases"]),
        "results": _registered_results(),
    }
    return cast(
        dict[str, Any],
        build_run_manifest(
            run_id=run_dir.name,
            run_kind="final",
            git_commit="b" * 40,
            git_dirty=False,
            artifact_root=run_dir,
            config=config,
            config_path="configs/final.yaml",
            dataset_manifest_path="manifests/dataset.json",
            split_manifest_path="manifests/split.json",
            split_partition="sealed_final",
            selected_example_ids=example_ids,
            corpus_manifest_path="manifests/corpus.json",
            exact_indexed_passage_count=10_000,
            index_manifest_path="manifests/index.json",
            evaluator_fixture_path="fixtures/evaluator_sanity_v1.json",
            evaluator_fixture_report=fixture_report,
            model_revisions={
                "example/model": _MODEL_REVISION,
                _SCORER_ID: _SCORER_REVISION,
            },
            seed=42,
            hardware={"gpu": "fixture"},
            software={"python": "3.10"},
            resource_ceilings={"gpu_hours": 1, "token_count": 1000},
            output_paths=["predictions.jsonl", "metrics.json"],
            mock=False,
            created_at_utc="2026-08-19T00:00:00+00:00",
        ),
    )


def _write_publication_prediction_bundle(
    run_dir: Path,
    predictions: list[dict[str, Any]],
    *,
    selected_count_override: int | None = None,
) -> Path:
    run_dir.mkdir(parents=True)
    path = run_dir / "predictions.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in predictions),
        encoding="utf-8",
    )
    ids = [str(record["example_id"]) for record in predictions]
    manifest = _build_bound_run_manifest(run_dir, ids)
    if selected_count_override is not None:
        plain = dict(manifest)
        plain.pop("manifest_sha256")
        plain["data"] = dict(plain["data"])
        plain["data"]["selected_example_count"] = selected_count_override
        manifest = seal_manifest(plain)
    write_immutable_json(run_dir / "run_manifest.json", manifest)
    sidecar = seal_manifest(
        {
            "schema": "factuality-rag.predictions.v1",
            "run_id": run_dir.name,
            "run_manifest_sha256": manifest["manifest_sha256"],
            "predictions_sha256": sha256_file(path),
            "example_ids_sha256": sha256_ordered_ids(ids),
            "example_count": len(ids),
            "mock": False,
        }
    )
    write_immutable_json(path.with_suffix(path.suffix + ".manifest.json"), sidecar)
    return path


def _sealed_metrics(
    run_id: str,
    run_manifest_sha256: str,
    *,
    support_metric: str = "none",
    n_predictions: int = 1,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "factuality-rag.metrics.v2",
        "run_id": run_id,
        "run_manifest_sha256": run_manifest_sha256,
        "support_metric": support_metric,
        "exact_match": 1.0,
        "f1": 1.0,
        "answered_count": float(n_predictions),
        "answer_coverage": 1.0,
        "n_predictions": float(n_predictions),
        "retrieval_count": 0.0,
        "retrieval_rate": 0.0,
    }
    if support_metric == "lexical":
        payload.update(
            {
                "lexical_support_answered_only": 1.0,
                "lexical_support_answered_count": float(n_predictions),
            }
        )
    return cast(dict[str, Any], seal_manifest(payload))


def _write_publication_analysis_bundle(
    run_dir: Path, predictions: list[dict[str, Any]], *, support_metric: str = "none"
) -> Path:
    path = _write_publication_prediction_bundle(run_dir, predictions)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = _sealed_metrics(
        run_dir.name,
        manifest["manifest_sha256"],
        support_metric=support_metric,
        n_predictions=len(predictions),
    )
    write_immutable_json(run_dir / "metrics.json", metrics)
    config_identity = "package://factuality_rag/configs/final.yaml"
    metadata = seal_manifest(
        {
            "schema": "factuality-rag.run-metadata.v2",
            "run_id": run_dir.name,
            "git_commit": manifest["git"]["commit"],
            "git_dirty": False,
            "config_path": config_identity,
            "config_identity": config_identity,
            "config_source_sha256": manifest["config_file_sha256"],
            "config_sha256": manifest["config_sha256"],
            "mock_mode": False,
            "support_metric": support_metric,
            "publication_artifact": True,
        }
    )
    write_immutable_json(run_dir / "metadata.json", metadata)
    return path


def _rewrite_sealed(path: Path, updates: Mapping[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload.update(updates)
    path.write_text(
        json.dumps(seal_manifest(payload), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_bootstrap_pairs_by_exact_id_not_file_position(tmp_path: Path) -> None:
    predictions_a = [
        _prediction("id-b", "q2", "wrong", ["right", "alias"]),
        _prediction("id-a", "q1", "alpha", "alpha"),
    ]
    predictions_b = [
        _prediction("id-a", "q1", "wrong", "alpha"),
        _prediction("id-b", "q2", "alias", ["right", "alias"]),
    ]

    pairs = pair_predictions(predictions_a, predictions_b)
    references = _resolve_references(pairs, tmp_path, tmp_path)
    scores_a, scores_b = compute_paired_scores(pairs, references, "exact_match")

    assert [pair[0] for pair in pairs] == ["id-a", "id-b"]
    assert scores_a.tolist() == [1.0, 0.0]
    assert scores_b.tolist() == [0.0, 1.0]


def test_bootstrap_rejects_id_set_mismatch() -> None:
    with pytest.raises(ValueError, match="example_id sets differ"):
        pair_predictions(
            [_prediction("id-a", "q", "a", "a")],
            [_prediction("id-b", "q", "a", "a")],
        )


def test_bootstrap_rejects_duplicate_ids() -> None:
    repeated = [_prediction("id-a", "q", "a", "a")] * 2
    with pytest.raises(ValueError, match="duplicate example_id"):
        pair_predictions(repeated, [_prediction("id-a", "q", "a", "a")])


def test_bootstrap_rejects_reference_mismatch(tmp_path: Path) -> None:
    pairs = pair_predictions(
        [_prediction("id-a", "q", "a", ["one", "alias"])],
        [_prediction("id-a", "q", "a", ["one", "different"])],
    )
    with pytest.raises(ValueError, match="reference mismatch"):
        _resolve_references(pairs, tmp_path, tmp_path)


def test_bootstrap_publication_mode_rejects_unsealed_predictions(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps(_prediction("id-a", "q", "a", "a")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run manifest not found"):
        _validate_publication_predictions(path, [_prediction("id-a", "q", "a", "a")])


def test_bootstrap_publication_mode_accepts_real_bound_final_artifacts(tmp_path: Path) -> None:
    example_id = _selected_final_ids()[0]
    predictions = [_prediction(example_id, "q", "a", "a")]
    path = _write_publication_prediction_bundle(tmp_path / "run-a", predictions)

    manifest = _validate_publication_predictions(path, predictions)

    assert manifest["run_kind"] == "final"
    assert manifest["data"]["selected_example_count"] == 1


def test_bootstrap_rejects_tampered_bound_config_bytes(tmp_path: Path) -> None:
    example_id = _selected_final_ids()[0]
    predictions = [_prediction(example_id, "q", "a", "a")]
    path = _write_publication_prediction_bundle(tmp_path / "run-a", predictions)
    with (path.parent / "configs" / "final.yaml").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="config bytes do not match the sealed digest"):
        _validate_publication_predictions(path, predictions)


def test_bootstrap_rejects_selected_count_mismatch_even_when_resealed(tmp_path: Path) -> None:
    example_id = _selected_final_ids()[0]
    predictions = [_prediction(example_id, "q", "a", "a")]
    path = _write_publication_prediction_bundle(
        tmp_path / "run-a", predictions, selected_count_override=2
    )

    with pytest.raises(ValueError, match="selected_example_count"):
        _validate_publication_predictions(path, predictions)


def test_bootstrap_rejects_boolean_sidecar_count_even_when_resealed(tmp_path: Path) -> None:
    example_id = _selected_final_ids()[0]
    predictions = [_prediction(example_id, "q", "a", "a")]
    path = _write_publication_prediction_bundle(tmp_path / "run-a", predictions)
    _rewrite_sealed(path.with_suffix(path.suffix + ".manifest.json"), {"example_count": True})

    with pytest.raises(ValueError, match="example_count must be a positive integer"):
        _validate_publication_predictions(path, predictions)


def test_bootstrap_rejects_unpaired_or_invalid_score_arrays() -> None:
    with pytest.raises(ValueError, match="exactly paired"):
        paired_bootstrap_test(np.array([1.0]), np.array([1.0, 0.0]), n_bootstrap=10)
    with pytest.raises(ValueError, match="positive integer"):
        paired_bootstrap_test(np.array([1.0]), np.array([1.0]), n_bootstrap=0)


def test_bootstrap_output_is_path_free_hash_bound_deterministic_and_not_claim_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_ids = _selected_final_ids(2)
    predictions_a = [
        _prediction(example_ids[0], "q1", "wrong", "right"),
        _prediction(example_ids[1], "q2", "two", "two"),
    ]
    predictions_b = [
        _prediction(example_ids[0], "q1", "right", "right"),
        _prediction(example_ids[1], "q2", "two", "two"),
    ]
    path_a = _write_publication_prediction_bundle(tmp_path / "run-a", predictions_a)
    path_b = _write_publication_prediction_bundle(tmp_path / "run-b", predictions_b)
    output_path = tmp_path / "analysis" / "bootstrap.json"
    argv = [
        "bootstrap_test.py",
        "--system-a",
        str(path_a.resolve()),
        "--system-b",
        str(path_b.resolve()),
        "--n-bootstrap",
        "100",
        "--seed",
        "7",
        "--output",
        str(output_path.resolve()),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    bootstrap_main()
    first_bytes = output_path.read_bytes()
    bootstrap_main()

    result = json.loads(first_bytes)
    assert output_path.read_bytes() == first_bytes
    assert verify_manifest(result)
    assert result["schema"] == "factuality-rag.bootstrap.v3"
    assert result["publication_safe"] is False
    assert result["publication_status"] == "structurally_validated_but_not_claim_safe"
    assert (
        result["publication_safety_reason"]
        == "runtime_immutable_model_revision_binding_is_unestablished"
    )
    assert str(tmp_path) not in first_bytes.decode("utf-8")
    assert result["system_a"] == "run-a"
    assert result["system_b"] == "run-b"
    assert result["paired_example_ids_sha256"] == sha256_ordered_ids(example_ids)
    binding_a = result["input_bindings"]["system_a"]
    assert binding_a["predictions_sha256"] == sha256_file(path_a)
    assert binding_a["run_manifest_file_sha256"] == sha256_file(path_a.parent / "run_manifest.json")
    assert binding_a["predictions_manifest_file_sha256"] == sha256_file(
        path_a.with_suffix(path_a.suffix + ".manifest.json")
    )


def test_aggregate_publication_mode_rejects_unsealed_metrics(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"support_metric": "none", "exact_match": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a valid sealed"):
        load_run_metrics(tmp_path)


def test_aggregate_accepts_exact_supported_metric_schema(tmp_path: Path) -> None:
    metrics = _sealed_metrics("run-a", "a" * 64, support_metric="lexical", n_predictions=2)
    write_immutable_json(tmp_path / "metrics.json", metrics)

    assert load_run_metrics(tmp_path) == metrics


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"unexpected_metric": 0.0}, "exact 'none' schema"),
        ({"exact_match": True}, "must be numeric"),
        ({"f1": 1.01}, r"closed interval \[0, 1\]"),
        ({"answered_count": 0.5}, "integer-valued"),
        ({"answered_count": 2.0}, "cannot exceed n_predictions"),
        ({"answer_coverage": 0.5}, "does not equal answered_count/n_predictions"),
        ({"retrieval_count": 1.0}, "does not equal retrieval_count/n_predictions"),
        ({"n_predictions": 0.0}, "positive integer-valued"),
    ],
)
def test_aggregate_rejects_invalid_metric_fields_types_ranges_and_identities(
    tmp_path: Path, updates: dict[str, Any], message: str
) -> None:
    payload = _sealed_metrics("run-a", "a" * 64)
    payload.pop("manifest_sha256")
    payload.update(updates)
    write_immutable_json(tmp_path / "metrics.json", seal_manifest(payload))

    with pytest.raises(ValueError, match=message):
        load_run_metrics(tmp_path)


def test_aggregate_rejects_lexical_count_identity_mismatch(tmp_path: Path) -> None:
    payload = _sealed_metrics("run-a", "a" * 64, support_metric="lexical", n_predictions=2)
    payload.pop("manifest_sha256")
    payload["lexical_support_answered_count"] = 1.0
    write_immutable_json(tmp_path / "metrics.json", seal_manifest(payload))

    with pytest.raises(ValueError, match="does not equal answered_count"):
        load_run_metrics(tmp_path)


def test_aggregate_manifest_loader_rejects_resealed_pilot_with_real_inputs(
    tmp_path: Path,
) -> None:
    example_id = _selected_final_ids()[0]
    run_dir = tmp_path / "run-a"
    _write_publication_prediction_bundle(run_dir, [_prediction(example_id, "q", "a", "a")])
    _rewrite_sealed(run_dir / "run_manifest.json", {"run_kind": "pilot"})

    with pytest.raises(ValueError, match="run_kind='final'"):
        _load_publication_manifest(run_dir)


def test_aggregate_bundle_binds_metric_count_to_selected_population(tmp_path: Path) -> None:
    example_id = _selected_final_ids()[0]
    run_dir = tmp_path / "run-a"
    _write_publication_analysis_bundle(run_dir, [_prediction(example_id, "q", "a", "a")])
    metrics_path = run_dir / "metrics.json"
    _rewrite_sealed(
        metrics_path,
        {
            "n_predictions": 2.0,
            "answered_count": 2.0,
            "answer_coverage": 1.0,
        },
    )

    with pytest.raises(ValueError, match="selected result population"):
        load_run_bundle(run_dir, exploratory=False)


def test_aggregate_output_is_hash_bound_sealed_and_not_claim_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_id = _selected_final_ids()[0]
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-a"
    _write_publication_analysis_bundle(run_dir, [_prediction(example_id, "q", "a", "a")])
    output_path = tmp_path / "analysis" / "aggregate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results.py",
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(output_path),
        ],
    )

    aggregate_main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert verify_manifest(result)
    assert result["schema"] == "factuality-rag.aggregate.v3"
    assert result["publication_safe"] is False
    assert result["publication_status"] == "structurally_validated_but_not_claim_safe"
    assert (
        result["publication_safety_reason"]
        == "runtime_immutable_model_revision_binding_is_unestablished"
    )
    binding = result["input_bindings"][0]
    assert binding["run_id"] == "run-a"
    assert binding["metrics_file_sha256"] == sha256_file(run_dir / "metrics.json")
    assert binding["metadata_file_sha256"] == sha256_file(run_dir / "metadata.json")
    assert binding["run_manifest_file_sha256"] == sha256_file(run_dir / "run_manifest.json")


def test_aggregate_exploratory_mode_strips_legacy_factscore(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "support_metric": "lexical",
                "factscore": 0.9,
                "lexical_support_answered_only": 0.2,
            }
        ),
        encoding="utf-8",
    )
    metrics = load_run_metrics(tmp_path, exploratory=True)

    assert "factscore" not in metrics
    assert metrics["lexical_support_answered_only"] == 0.2


def test_aggregate_exploratory_mode_still_rejects_ambiguous_json(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        '{"support_metric":"none","exact_match":1,"exact_match":0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_run_metrics(tmp_path, exploratory=True)
