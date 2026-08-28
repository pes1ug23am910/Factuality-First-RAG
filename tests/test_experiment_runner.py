"""Boundary tests for reference extraction and experiment persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from factuality_rag.experiment_runner import (
    _extract_reference,
    _get_git_state,
    _parse_args,
    _resolve_dataset_selection,
    build_metadata,
    load_reference_artifacts,
    resolve_record_reference,
    run,
)


class _FakePipeline:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self, query: str, **kwargs: Any) -> tuple:
        return "NYC", [], [], "low"


@pytest.mark.parametrize(
    "config,error_type,match",
    [
        ({"gating": {"enabled": "false"}}, TypeError, "gating.enabled"),
        ({"retriever": {"rerank": 1}}, TypeError, "retriever.rerank"),
        ({"retriever": {"top_k": True}}, ValueError, "retriever.top_k"),
        ({"scorer": {"score_threshold": float("nan")}}, ValueError, "score_threshold"),
        ({"scorer": {"score_threshold": 1.1}}, ValueError, "score_threshold"),
    ],
)
def test_invalid_execution_modes_fail_before_run_directory_creation(
    tmp_path: Path,
    config: Dict[str, Any],
    error_type: type[Exception],
    match: str,
) -> None:
    runs_dir = tmp_path / "runs"

    with pytest.raises(error_type, match=match):
        run(config, queries=["query"], runs_dir=str(runs_dir), mock_mode=True)

    assert not runs_dir.exists()


def test_run_never_substitutes_demo_queries(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    with pytest.raises(ValueError, match="demo fallback is disabled"):
        run({"seed": 7}, runs_dir=str(runs_dir), mock_mode=True)
    assert not runs_dir.exists()


def test_config_only_b2_resolves_its_declared_dataset_split_and_sample() -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "exp_b2_always_rag.yaml").read_text(
            encoding="utf-8"
        )
    )

    selection = _resolve_dataset_selection(
        config,
        cli_dataset=None,
        cli_split=None,
        cli_sample=None,
    )

    assert selection == ("natural_questions", "validation", 500)


def test_multiple_config_datasets_require_explicit_cli_selection() -> None:
    config = {
        "data": {
            "datasets": [
                {"name": "natural_questions", "split": "validation"},
                {"name": "hotpot_qa", "split": "validation"},
            ]
        }
    }

    with pytest.raises(ValueError, match="exactly one dataset"):
        _resolve_dataset_selection(
            config,
            cli_dataset=None,
            cli_split=None,
            cli_sample=None,
        )


def test_installed_source_without_direct_git_marker_never_walks_to_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import factuality_rag.experiment_runner as runner

    installed_root = tmp_path / "outer-repository" / "site-packages" / "factuality_rag_source"
    installed_root.mkdir(parents=True)
    (tmp_path / "outer-repository" / ".git").mkdir()
    monkeypatch.setattr(runner, "_SOURCE_ROOT", installed_root)

    def forbidden(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("Git must not run without a direct source-root .git marker")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    assert _get_git_state() == ("git-not-available", None)


def test_git_state_rejects_upward_or_mismatched_toplevel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import factuality_rag.experiment_runner as runner

    source_root = tmp_path / "source"
    unrelated_root = tmp_path / "unrelated"
    (source_root / ".git").mkdir(parents=True)
    unrelated_root.mkdir()
    monkeypatch.setattr(runner, "_SOURCE_ROOT", source_root)

    calls = 0

    def mismatched(command: list[str], **kwargs: Any) -> bytes:
        nonlocal calls
        calls += 1
        assert command[-1] == "--show-toplevel"
        return str(unrelated_root).encode()

    monkeypatch.setattr(subprocess, "check_output", mismatched)
    assert _get_git_state() == ("git-not-available", None)
    assert calls == 1


def test_git_state_records_full_head_and_dirty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import factuality_rag.experiment_runner as runner

    source_root = tmp_path / "source"
    (source_root / ".git").mkdir(parents=True)
    monkeypatch.setattr(runner, "_SOURCE_ROOT", source_root)
    commit = "a" * 40

    def source_git(command: list[str], **kwargs: Any) -> bytes:
        if command[-1] == "--show-toplevel":
            return str(source_root).encode()
        if command[-2:] == ["rev-parse", "HEAD"]:
            return commit.encode()
        if "status" in command:
            return b" M factuality_rag/eval/metrics.py\n"
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "check_output", source_git)
    assert _get_git_state() == (commit, True)


def test_metadata_uses_source_root_config_bytes_not_hostile_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.experiment_runner as runner
    from factuality_rag.reproducibility import sha256_bytes, sha256_json

    source_root = tmp_path / "source"
    hostile_cwd = tmp_path / "hostile"
    source_config = source_root / "configs" / "exp.yaml"
    hostile_config = hostile_cwd / "configs" / "exp.yaml"
    source_config.parent.mkdir(parents=True)
    hostile_config.parent.mkdir(parents=True)
    source_bytes = b"seed: 7\n"
    source_config.write_bytes(source_bytes)
    hostile_config.write_bytes(b"seed: 999\n")
    monkeypatch.setattr(runner, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(runner, "_get_git_state", lambda: ("git-not-available", None))
    monkeypatch.setattr(runner, "_get_lib_versions", lambda: {})
    monkeypatch.chdir(hostile_cwd)

    metadata = build_metadata({"seed": 7}, "configs/exp.yaml")
    source_sha256 = sha256_bytes(source_bytes)

    assert metadata["config_path"] == f"external-config://exp.yaml?sha256={source_sha256}"
    assert metadata["config_identity"] == metadata["config_path"]
    assert metadata["config_source_sha256"] == source_sha256
    assert metadata["config_sha256"] == sha256_json({"seed": 7})
    assert str(tmp_path) not in json.dumps(metadata)


def test_metadata_external_identity_never_persists_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.experiment_runner as runner

    config_path = tmp_path / "Private Config.yaml"
    config_path.write_bytes(b"seed: 7\n")
    monkeypatch.setattr(runner, "_get_git_state", lambda: ("git-not-available", None))
    monkeypatch.setattr(runner, "_get_lib_versions", lambda: {})

    metadata = build_metadata({"seed": 7}, str(config_path))

    assert metadata["config_path"].startswith("external-config://Private%20Config.yaml?sha256=")
    assert str(tmp_path) not in json.dumps(metadata)


def test_metadata_package_identity_is_logical_and_content_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.experiment_runner as runner
    from factuality_rag.resources import DEFAULT_EXPERIMENT_CONFIG, experiment_config_identity

    monkeypatch.setattr(runner, "_get_git_state", lambda: ("git-not-available", None))
    monkeypatch.setattr(runner, "_get_lib_versions", lambda: {})
    metadata = build_metadata(
        {"seed": 7},
        experiment_config_identity(DEFAULT_EXPERIMENT_CONFIG),
    )

    assert metadata["config_path"].startswith(
        "package://factuality_rag.resources/configs/exp_full_pipeline.yaml?sha256="
    )
    assert metadata["config_source_sha256"] == metadata["config_path"].rsplit("=", 1)[1]


def test_extra_metadata_cannot_override_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.experiment_runner as runner

    monkeypatch.setattr(runner, "_get_git_state", lambda: ("git-not-available", None))
    monkeypatch.setattr(runner, "_get_lib_versions", lambda: {})
    with pytest.raises(ValueError, match="must not override"):
        build_metadata({"seed": 7}, extra={"git_commit": "spoofed"})


def test_nq_and_answer_dict_aliases_are_not_collapsed() -> None:
    assert _extract_reference({"answer": ["New York City", "NYC"]}, "nq_open") == [
        "New York City",
        "NYC",
    ]
    assert _extract_reference(
        {
            "answer": {
                "value": "NYC",
                "text": ["New York City"],
                "normalized_aliases": ["Big Apple", "NYC"],
            }
        },
        "custom",
    ) == ["NYC", "New York City", "Big Apple"]


@pytest.mark.parametrize(
    "dataset,row,match",
    [
        ("fever", {"label": "SUPPORTS"}, "classification"),
        (
            "EleutherAI/truthful_qa_mc",
            {"choices": ["a", "b"], "label": 0},
            "choices/label",
        ),
        ("popqa", {"possible_answers": ["answer"], "s_pop": 1}, "popularity-stratum"),
        (
            "hagrid",
            {"answers": [{"answer": "text", "attributable": False}]},
            "informative/attributable",
        ),
    ],
)
def test_task_specific_reference_extraction_fails_closed(
    dataset: str, row: Dict[str, Any], match: str
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _extract_reference(row, dataset)


def test_run_rejects_reference_length_mismatch_before_side_effects(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="references and queries must have the same length"):
        run(
            {"seed": 7},
            queries=["one", "two"],
            references=["one"],
            runs_dir=str(tmp_path),
            mock_mode=True,
        )
    assert list(tmp_path.iterdir()) == []


def test_run_rejects_bare_reference_string_before_side_effects(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="references must be an ordered sequence"):
        run(
            {"seed": 7},
            queries=["one"],
            references="x",
            runs_dir=str(tmp_path),
            mock_mode=True,
        )
    assert list(tmp_path.iterdir()) == []


def test_run_rejects_legacy_factscore_label_before_side_effects(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="legacy FactScore"):
        run(
            {"seed": 7, "eval": {"metrics": ["exact_match", "factscore"]}},
            queries=["one"],
            runs_dir=str(tmp_path),
            mock_mode=True,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("queries", "error"),
    [
        ([], ValueError),
        ([""], ValueError),
        ([" blank"], ValueError),
        (["line\nbreak"], ValueError),
        ([1], TypeError),
        ("bare string", TypeError),
        (b"bytes", TypeError),
    ],
)
def test_run_rejects_invalid_queries_before_side_effects(
    tmp_path: Path, queries: Any, error: type
) -> None:
    with pytest.raises(error):
        run({"seed": 7}, queries=queries, runs_dir=str(tmp_path), mock_mode=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("prefix", ["../outside", "..\\outside", "/rooted", "bad\nname"])
def test_run_rejects_unsafe_prefix_before_side_effects(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ValueError, match="run_id_prefix"):
        run({"seed": 7}, queries=["one"], runs_dir=str(tmp_path), run_id_prefix=prefix)
    assert list(tmp_path.iterdir()) == []


def test_duplicate_queries_have_lossless_reference_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.pipeline.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "Pipeline", _FakePipeline)
    result = run(
        {"seed": 7},
        queries=["same question", "same question"],
        references=[["New York City", "NYC"], "York"],
        runs_dir=str(tmp_path),
        mock_mode=True,
    )
    run_dir = Path(result["run_dir"])
    with open(run_dir / "references_by_example_id.json", encoding="utf-8") as f:
        by_id: Dict[str, Dict[str, Any]] = json.load(f)
    with open(run_dir / "references.json", encoding="utf-8") as f:
        legacy = json.load(f)

    assert list(by_id) == ["row-00000000", "row-00000001"]
    assert by_id["row-00000000"]["reference"] == ["New York City", "NYC"]
    assert by_id["row-00000001"]["reference"] == "York"
    assert legacy == {}
    assert result["metrics"]["exact_match"] == 0.5
    assert result["metadata"]["ambiguous_legacy_reference_queries"] == 1


def test_explicit_lexical_mode_never_emits_factscore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.pipeline.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "Pipeline", _FakePipeline)
    result = run(
        {"seed": 7, "eval": {"metrics": ["exact_match", "lexical_support"]}},
        queries=["one"],
        references=["NYC"],
        runs_dir=str(tmp_path),
        mock_mode=True,
    )

    assert result["metrics"]["support_metric"] == "lexical"
    assert "lexical_support_answered_only" in result["metrics"]
    assert not any("factscore" in key.lower() for key in result["metrics"])
    assert result["metadata"]["support_metric"] == "lexical"
    assert result["metadata"]["mock_mode"] is True
    assert result["metadata"]["publication_artifact"] is False
    assert len(result["metadata"]["config_sha256"]) == 64
    assert result["metadata"]["config_path"].startswith("in-memory-config://effective?")
    assert result["metadata"]["config_source_sha256"] is None


def test_reference_resolution_is_example_id_first_before_legacy(tmp_path: Path) -> None:
    (tmp_path / "references_by_example_id.json").write_text(
        json.dumps(
            {
                "row-00000000": {"input": "duplicate", "reference": ["first", "one"]},
                "row-00000001": {"input": "duplicate", "reference": ["second", "two"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "references.json").write_text(json.dumps({"duplicate": "legacy"}), encoding="utf-8")
    by_id, by_query = load_reference_artifacts(tmp_path)

    assert resolve_record_reference(
        {"example_id": "row-00000000", "input": "duplicate"}, by_id, by_query
    ) == ["first", "one"]
    assert resolve_record_reference(
        {"example_id": "row-00000001", "input": "duplicate"}, by_id, by_query
    ) == ["second", "two"]
    assert (
        resolve_record_reference(
            {
                "example_id": "row-00000000",
                "input": "duplicate",
                "reference": "inline",
            },
            by_id,
            by_query,
        )
        == "inline"
    )


def test_prefixed_run_ids_are_unique(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import factuality_rag.pipeline.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "Pipeline", _FakePipeline)
    first = run({"seed": 7}, queries=["one"], runs_dir=str(tmp_path), run_id_prefix="trial")
    second = run({"seed": 7}, queries=["one"], runs_dir=str(tmp_path), run_id_prefix="trial")
    assert first["run_id"] != second["run_id"]
    assert Path(first["run_dir"]).is_dir()
    assert Path(second["run_dir"]).is_dir()


def test_runtime_source_identity_changes_with_exact_package_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.experiment_runner as runner

    source_root = tmp_path / "factuality_rag"
    source_root.mkdir()
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "experiment_runner.py").write_text("VERSION = 1\n", encoding="utf-8")
    resource = source_root / "resources" / "policy.json"
    resource.parent.mkdir()
    resource.write_text('{"mode":"first"}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "_RUNTIME_SOURCE_ROOT", source_root)

    first = runner._get_runtime_source_identity()
    resource.write_text('{"mode":"other"}\n', encoding="utf-8")
    second = runner._get_runtime_source_identity()

    assert first["schema"] == "factuality-rag.runtime-source.v1"
    assert first["file_count"] == second["file_count"] == 3
    assert first["sha256"] != second["sha256"]


def test_runtime_package_origin_rejects_shadowing_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag
    import factuality_rag.experiment_runner as runner

    shadow_init = tmp_path / "site-packages" / "factuality_rag" / "__init__.py"
    shadow_init.parent.mkdir(parents=True)
    shadow_init.write_text("", encoding="utf-8")
    monkeypatch.setattr(factuality_rag, "__file__", str(shadow_init))

    with pytest.raises(RuntimeError, match="does not match the experiment runner"):
        runner._validate_runtime_package_origin()


def test_resume_rejects_changed_dirty_source_bytes_at_same_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.experiment_runner as runner
    import factuality_rag.pipeline.orchestrator as orchestrator

    source_root = tmp_path / "factuality_rag"
    source_root.mkdir()
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    runner_source = source_root / "experiment_runner.py"
    runner_source.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_RUNTIME_SOURCE_ROOT", source_root)
    monkeypatch.setattr(runner, "_get_git_state", lambda: ("a" * 40, True))
    monkeypatch.setattr(runner, "_get_lib_versions", lambda: {})
    monkeypatch.setattr(orchestrator, "Pipeline", _FakePipeline)

    created = run(
        {"seed": 7},
        queries=["one"],
        runs_dir=str(tmp_path / "runs"),
        mock_mode=True,
    )
    checkpoint = Path(created["run_dir"]) / "predictions.jsonl"
    before = checkpoint.read_bytes()
    runner_source.write_text("VERSION = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment binding does not match"):
        run(
            {"seed": 7},
            queries=["one"],
            mock_mode=True,
            resume_dir=created["run_dir"],
        )

    manifest = json.loads(
        (Path(created["run_dir"]) / "resume_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "factuality-rag.resume-checkpoint.v2"
    assert manifest["bindings"]["environment"]["git_dirty"] is True
    assert (
        manifest["bindings"]["environment"]["runtime_source"]
        == created["metadata"]["runtime_source"]
    )
    assert checkpoint.read_bytes() == before


def test_non_mock_run_uses_source_binding_when_git_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.experiment_runner as runner
    import factuality_rag.pipeline.orchestrator as orchestrator

    source_root = tmp_path / "factuality_rag"
    source_root.mkdir()
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "experiment_runner.py").write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_RUNTIME_SOURCE_ROOT", source_root)
    monkeypatch.setattr(runner, "_get_git_state", lambda: ("git-not-available", None))
    monkeypatch.setattr(runner, "_get_lib_versions", lambda: {})
    monkeypatch.setattr(orchestrator, "Pipeline", _FakePipeline)

    result = run(
        {"seed": 7},
        queries=["one"],
        runs_dir=str(tmp_path / "runs"),
        mock_mode=False,
    )

    assert result["metadata"]["git_commit"] == "git-not-available"
    assert result["metadata"]["git_dirty"] is None
    assert result["metadata"]["runtime_source"]["file_count"] == 2
    assert len(result["metadata"]["runtime_source"]["sha256"]) == 64


@pytest.mark.parametrize("env_name", [".env", ".env.local"])
def test_runtime_source_identity_refuses_env_without_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    import factuality_rag.experiment_runner as runner

    source_root = tmp_path / "factuality_rag"
    source_root.mkdir()
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "experiment_runner.py").write_text("VERSION = 1\n", encoding="utf-8")
    (source_root / env_name).write_text("must-not-be-read", encoding="utf-8")
    monkeypatch.setattr(runner, "_RUNTIME_SOURCE_ROOT", source_root)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == env_name:
            raise AssertionError(".env file must be rejected before reading")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(RuntimeError, match="must not contain .env"):
        runner._get_runtime_source_identity()


def _create_interrupted_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Dict[str, Any], list[str], list[str]]:
    import factuality_rag.pipeline.orchestrator as orchestrator

    config: Dict[str, Any] = {"seed": 17}
    queries = ["first question", "second question", "third question"]
    references = ["first answer", "second answer", "different answer"]
    calls: list[str] = []

    class InterruptingPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self, query: str, **kwargs: Any) -> tuple:
            calls.append(query)
            if query == queries[1]:
                raise RuntimeError("simulated interruption")
            kwargs["info"]["retrieval_triggered"] = query != queries[1]
            kwargs["info"]["scorer_enabled"] = True
            answer = references[queries.index(query)]
            return answer, [{"id": query, "text": answer}], {}, "high"

    monkeypatch.setattr(orchestrator, "Pipeline", InterruptingPipeline)
    runs_dir = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(
            config,
            queries=queries,
            references=references,
            runs_dir=str(runs_dir),
            mock_mode=True,
        )

    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    assert calls == queries[:2]
    return run_dirs[0], config, queries, references


def test_resume_discards_torn_tail_skips_completed_ids_and_matches_fresh_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.pipeline.orchestrator as orchestrator

    run_dir, config, queries, references = _create_interrupted_checkpoint(
        tmp_path,
        monkeypatch,
    )
    checkpoint = run_dir / "predictions.jsonl"
    with checkpoint.open("ab") as handle:
        handle.write(b'{"example_id":"row-00000001","input":"torn')

    resumed_calls: list[str] = []

    class DeterministicPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self, query: str, **kwargs: Any) -> tuple:
            resumed_calls.append(query)
            kwargs["info"]["retrieval_triggered"] = query != queries[1]
            kwargs["info"]["scorer_enabled"] = True
            answer = references[queries.index(query)]
            return answer, [{"id": query, "text": answer}], {}, "high"

    monkeypatch.setattr(orchestrator, "Pipeline", DeterministicPipeline)
    resumed = run(
        config,
        queries=queries,
        references=references,
        mock_mode=True,
        resume_dir=run_dir,
    )

    assert resumed_calls == queries[1:]
    assert resumed["run_id"] == run_dir.name
    assert len(resumed["predictions"]) == len(queries)
    example_ids = [record["example_id"] for record in resumed["predictions"]]
    assert example_ids == [f"row-{index:08d}" for index in range(len(queries))]
    assert len(example_ids) == len(set(example_ids))
    checkpoint_lines = checkpoint.read_bytes().splitlines(keepends=True)
    assert len(checkpoint_lines) == len(queries)
    assert all(line.endswith(b"\n") for line in checkpoint_lines)
    assert b'"input":"torn' not in checkpoint.read_bytes()

    resumed_calls.clear()
    fresh = run(
        config,
        queries=queries,
        references=references,
        runs_dir=str(tmp_path / "fresh"),
        mock_mode=True,
    )
    assert resumed_calls == queries
    assert resumed["metrics"] == fresh["metrics"]
    assert json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) == fresh["metrics"]


def test_resume_rejects_newline_terminated_malformed_record_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import factuality_rag.pipeline.orchestrator as orchestrator

    run_dir, config, queries, references = _create_interrupted_checkpoint(
        tmp_path,
        monkeypatch,
    )
    checkpoint = run_dir / "predictions.jsonl"
    with checkpoint.open("ab") as handle:
        handle.write(b'{"malformed":\n')
    before = checkpoint.read_bytes()

    class MustNotConstruct:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("checkpoint validation must precede Pipeline construction")

    monkeypatch.setattr(orchestrator, "Pipeline", MustNotConstruct)
    with pytest.raises(ValueError, match="line 2 is not valid strict JSON"):
        run(
            config,
            queries=queries,
            references=references,
            mock_mode=True,
            resume_dir=run_dir,
        )
    assert checkpoint.read_bytes() == before


def test_resume_rejects_config_mismatch_before_truncating_torn_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, config, queries, references = _create_interrupted_checkpoint(
        tmp_path,
        monkeypatch,
    )
    checkpoint = run_dir / "predictions.jsonl"
    with checkpoint.open("ab") as handle:
        handle.write(b'{"uncommitted":')
    before = checkpoint.read_bytes()

    changed_config = {**config, "seed": 18}
    with pytest.raises(ValueError, match="config binding does not match"):
        run(
            changed_config,
            queries=queries,
            references=references,
            mock_mode=True,
            resume_dir=run_dir,
        )
    assert checkpoint.read_bytes() == before


def test_resume_requires_manifest_and_cli_excludes_new_run_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_run = tmp_path / "legacy-run"
    legacy_run.mkdir()
    (legacy_run / "predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="resume_manifest.json"):
        run(
            {"seed": 7},
            queries=["one"],
            mock_mode=True,
            resume_dir=legacy_run,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_runner", "--resume", str(legacy_run)],
    )
    parsed = _parse_args()
    assert parsed.resume == str(legacy_run)
    assert parsed.run_id is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_runner", "--resume", str(legacy_run), "--run-id", "new"],
    )
    with pytest.raises(SystemExit):
        _parse_args()
