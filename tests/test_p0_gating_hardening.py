"""Fail-closed regression tests for exploratory gating proxy analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import pytest

from scripts import analyze_gating


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _make_runs(
    tmp_path: Path,
    *,
    full_records: Sequence[Dict[str, Any]],
    closedbook_records: Sequence[Dict[str, Any]],
) -> tuple[Path, Path]:
    full_dir = tmp_path / "full"
    closedbook_dir = tmp_path / "closedbook"
    full_dir.mkdir()
    closedbook_dir.mkdir()
    _write_jsonl(full_dir / "predictions.jsonl", full_records)
    _write_jsonl(closedbook_dir / "predictions.jsonl", closedbook_records)
    return full_dir, closedbook_dir


def _run_offline(
    monkeypatch: pytest.MonkeyPatch,
    full_dir: Path,
    closedbook_dir: Path,
    output_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_gating.py",
            "--full-run",
            str(full_dir),
            "--closedbook-run",
            str(closedbook_dir),
            "--output",
            str(output_path),
        ],
    )
    analyze_gating.main()


def _valid_full_record(**overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "example_id": "row-00000000",
        "input": "Question?",
        "reference": ["Correct answer", "alias"],
        "answer": "Correct answer",
        "retrieval_triggered": True,
    }
    record.update(overrides)
    return record


def _valid_closedbook_record(**overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "example_id": "row-00000000",
        "input": "Question?",
        "answer": "alias",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize("option", ["--full-run", "--closedbook-run"])
def test_offline_arguments_must_be_supplied_as_a_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    output_path = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_gating.py", option, str(tmp_path / "run"), "--output", str(output_path)],
    )

    with pytest.raises(ValueError, match="must be provided together"):
        analyze_gating.main()
    assert not output_path.exists()


@pytest.mark.parametrize("invalid_value", [1, "true", None])
def test_offline_retrieval_decision_requires_exact_bool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: Any,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record(retrieval_triggered=invalid_value)],
        closedbook_records=[_valid_closedbook_record()],
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(TypeError, match="retrieval_triggered must be exactly bool"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_offline_unresolved_reference_fails_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_record = _valid_full_record()
    full_record.pop("reference")
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[full_record],
        closedbook_records=[_valid_closedbook_record()],
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="unresolved reference"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_shared_example_id_with_different_queries_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record(input="Different question?")],
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="query mismatch for shared example_id"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_reference_artifact_must_bind_example_id_to_prediction_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_record = _valid_full_record()
    full_record.pop("reference")
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[full_record],
        closedbook_records=[_valid_closedbook_record()],
    )
    (full_dir / "references_by_example_id.json").write_text(
        json.dumps(
            {
                "row-00000000": {
                    "input": "A different stored query?",
                    "reference": "Correct answer",
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="stored query mismatch"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_unmatched_closedbook_record_fails_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = _valid_closedbook_record(
        example_id="row-00000001",
        input="Extra question?",
    )
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record(), extra],
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="unmatched prediction record"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_valid_offline_artifact_binds_inputs_and_reports_complete_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_record = _valid_full_record(reference=["Correct answer", "alias"])
    closedbook_record = _valid_closedbook_record()
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[full_record],
        closedbook_records=[closedbook_record],
    )
    (full_dir / "references_by_example_id.json").write_text(
        json.dumps(
            {
                "row-00000000": {
                    "input": "Question?",
                    "reference": ["alias", "Correct answer"],
                }
            }
        ),
        encoding="utf-8",
    )
    (closedbook_dir / "references_by_example_id.json").write_text(
        json.dumps(
            {
                "row-00000000": {
                    "input": "Question?",
                    "reference": ["Correct answer", "alias"],
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "analysis.json"

    _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["schema"] == analyze_gating.GATING_ANALYSIS_SCHEMA
    assert artifact["status"] == "exploratory_proxy_not_retrieval_utility"
    assert artifact["publication_safe"] is False
    assert artifact["mode"] == "offline"
    assert len(artifact["input_sha256"]) == 64
    int(artifact["input_sha256"], 16)
    full_predictions_path = full_dir / "predictions.jsonl"
    assert (
        artifact["inputs"]["full_run"]["predictions"]["sha256"]
        == hashlib.sha256(full_predictions_path.read_bytes()).hexdigest()
    )
    assert artifact["coverage"] == {
        "n_full_records": 1,
        "n_closedbook_records": 1,
        "n_references_resolved": 1,
        "n_closedbook_matched": 1,
        "n_analyzed": 1,
        "n_unresolved_references": 0,
        "n_unmatched_full_records": 0,
        "n_unused_closedbook_records": 0,
        "analysis_fraction": 1.0,
    }
    assert artifact["metrics"]["n_analyzed"] == 1
    assert artifact["per_query"][0]["closed_book_correct"] is True
    assert artifact["per_query"][0]["closed_book_error_proxy"] is False
    assert "oracle_decision" not in artifact["per_query"][0]
    assert artifact["per_query"][0]["closed_book_match_method"] == "example_id"


def test_fallback_pairing_rejects_different_non_null_example_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record(example_id="full-id")],
        closedbook_records=[_valid_closedbook_record(example_id="closedbook-id")],
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="example_id values differ"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_offline_analysis_rejects_the_same_run_for_both_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "predictions.jsonl", [_valid_full_record()])
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="different run directories"):
        _run_offline(monkeypatch, run_dir, run_dir, output_path)
    assert not output_path.exists()


@pytest.mark.parametrize("target_name", ["predictions.jsonl", "references_by_example_id.json"])
@pytest.mark.parametrize("alias_suffix", [".", ". "])
def test_output_rejects_windows_alias_of_consumed_offline_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    alias_suffix: str,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    reference_path = full_dir / "references_by_example_id.json"
    reference_path.write_text(
        json.dumps(
            {
                "row-00000000": {
                    "input": "Question?",
                    "reference": ["Correct answer", "alias"],
                }
            }
        ),
        encoding="utf-8",
    )
    target_path = full_dir / target_name
    original_bytes = target_path.read_bytes()
    output_alias = Path(str(target_path) + alias_suffix)

    with pytest.raises((ValueError, FileExistsError), match="alias|refusing to replace"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_alias)
    assert target_path.read_bytes() == original_bytes


def test_jsonl_duplicate_keys_are_rejected_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    (full_dir / "predictions.jsonl").write_text(
        '{"example_id":"first","example_id":"second","input":"Question?",'
        '"reference":"Correct answer","retrieval_triggered":true}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_json_reference_non_finite_value_is_rejected_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_record = _valid_full_record()
    full_record.pop("reference")
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[full_record],
        closedbook_records=[_valid_closedbook_record()],
    )
    (full_dir / "references_by_example_id.json").write_text(
        '{"row-00000000":{"input":"Question?","reference":NaN}}',
        encoding="utf-8",
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_live_mode_snapshots_and_passes_both_exact_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    full_config_path = configs_dir / "full.yaml"
    closedbook_config_path = configs_dir / "exp_b1_closed_book.yaml"
    full_bytes = b"full_marker: original\n"
    closedbook_bytes = b"closedbook_marker: original\n"
    full_config_path.write_bytes(full_bytes)
    closedbook_config_path.write_bytes(closedbook_bytes)
    output_path = tmp_path / "analysis.json"
    constructor_calls: list[Dict[str, Any]] = []

    class FakePipeline:
        def __init__(self, *, config_path: str, config: Dict[str, Any], **kwargs: Any) -> None:
            self.closedbook = config_path.endswith("exp_b1_closed_book.yaml")
            constructor_calls.append(
                {
                    "config_path": config_path,
                    "config": config,
                    "kwargs": kwargs,
                }
            )
            if self.closedbook:
                full_config_path.write_bytes(b"full_marker: changed_after_snapshot\n")

        def run(self, query: str, **kwargs: Any) -> tuple[str, list, list, str]:
            if self.closedbook:
                answers = {
                    "What is the capital of France?": "Paris",
                    "Who wrote Hamlet?": "Shakespeare",
                    "What is DNA?": "Deoxyribonucleic acid",
                }
                return answers[query], [], [], "high"
            info = kwargs["info"]
            info["retrieval_triggered"] = False
            return "answer", [], [], "high"

    import factuality_rag.pipeline.orchestrator as orchestrator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "Pipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_gating.py",
            "--mock",
            "--config",
            "configs/full.yaml",
            "--output",
            str(output_path),
        ],
    )

    analyze_gating.main()

    assert [call["config"] for call in constructor_calls] == [
        {"closedbook_marker": "original"},
        {"full_marker": "original"},
    ]
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["inputs"]["full_config"]["sha256"] == hashlib.sha256(full_bytes).hexdigest()
    assert (
        artifact["inputs"]["closedbook_config"]["sha256"]
        == hashlib.sha256(closedbook_bytes).hexdigest()
    )


@pytest.mark.parametrize("alias_suffix", [".", ". "])
def test_live_output_rejects_windows_config_alias_before_pipeline_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_suffix: str,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    full_config_path = configs_dir / "full.yaml"
    full_config_path.write_text("full_marker: original\n", encoding="utf-8")
    (configs_dir / "exp_b1_closed_book.yaml").write_text(
        "closedbook_marker: original\n",
        encoding="utf-8",
    )
    construction_count = 0

    class ConstructionForbidden:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal construction_count
            construction_count += 1

    import factuality_rag.pipeline.orchestrator as orchestrator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "Pipeline", ConstructionForbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_gating.py",
            "--mock",
            "--config",
            "configs/full.yaml",
            "--output",
            str(full_config_path) + alias_suffix,
        ],
    )

    with pytest.raises((ValueError, FileExistsError), match="alias|refusing to replace"):
        analyze_gating.main()
    assert construction_count == 0


def test_preexisting_output_is_refused_and_preserved_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    output_path = tmp_path / "analysis.json"
    sentinel = b'{"owner":"preexisting"}\n'
    output_path.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="refusing to replace existing analysis output"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert output_path.read_bytes() == sentinel


def test_publication_race_preserves_competing_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    output_path = tmp_path / "analysis.json"
    competing_bytes = b'{"owner":"competing-writer"}\n'

    def competing_link(source: object, destination: object) -> None:
        assert Path(destination) == output_path
        assert Path(source).is_file()
        output_path.write_bytes(competing_bytes)
        raise FileExistsError("simulated publication race")

    monkeypatch.setattr(analyze_gating.os, "link", competing_link)

    with pytest.raises(FileExistsError, match="refusing to replace existing analysis output"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert output_path.read_bytes() == competing_bytes
    assert list(tmp_path.glob(".analysis.json.*.tmp")) == []


def test_existing_unconsumed_run_artifact_cannot_be_used_as_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    metrics_path = full_dir / "metrics.json"
    sentinel = b'{"existing_metric":1}\n'
    metrics_path.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="refusing to replace existing analysis output"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, metrics_path)
    assert metrics_path.read_bytes() == sentinel


def test_output_serialization_rejects_nan_without_publishing(tmp_path: Path) -> None:
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="not strict JSON serializable"):
        analyze_gating._write_new_json(output_path, {"metric": float("nan")})
    assert not output_path.exists()


def test_json_float_overflow_is_rejected_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    (full_dir / "predictions.jsonl").write_text(
        '{"example_id":"row-00000000","input":"Question?",'
        '"reference":"Correct answer","retrieval_triggered":true,"weight":1e400}\n',
        encoding="utf-8",
        newline="\n",
    )
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="non-finite JSON number"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()


def test_jsonl_uses_only_lf_as_a_record_delimiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dir, closedbook_dir = _make_runs(
        tmp_path,
        full_records=[_valid_full_record()],
        closedbook_records=[_valid_closedbook_record()],
    )
    first = json.dumps(_valid_full_record(), ensure_ascii=False)
    second = json.dumps(
        _valid_full_record(example_id="row-00000001", input="Second question?"),
        ensure_ascii=False,
    )
    (full_dir / "predictions.jsonl").write_bytes((first + "\r" + second).encode("utf-8"))
    output_path = tmp_path / "analysis.json"

    with pytest.raises(ValueError, match="not strict JSON"):
        _run_offline(monkeypatch, full_dir, closedbook_dir, output_path)
    assert not output_path.exists()
