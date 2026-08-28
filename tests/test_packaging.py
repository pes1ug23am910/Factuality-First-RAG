"""Packaging, resource-integrity, and arbitrary-working-directory gates."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from factuality_rag.eval.sanity import (
    EVALUATOR_SANITY_V1_CONTENT_SHA256,
    load_sanity_fixture_bytes,
)
from factuality_rag.pipeline.orchestrator import Pipeline, _load_config
from factuality_rag.resources import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    EXPERIMENT_CONFIG_NAMES,
    experiment_config_identity,
    read_evaluator_sanity_bytes,
    read_experiment_config_bytes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_FIXTURE_ARTIFACT_SHA256 = (
    "093b301d216fedfafa8beea95d9d02f6170919dc7549427641aade7d0c2a5aaa"
)


def _lf_text(raw: bytes) -> str:
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


@pytest.mark.parametrize("name", EXPERIMENT_CONFIG_NAMES)
def test_packaged_experiment_config_matches_checkout_copy(name: str) -> None:
    checkout = (REPO_ROOT / "configs" / name).read_bytes()
    packaged = read_experiment_config_bytes(name)

    # Git may materialize tracked text with CRLF on Windows. Compare the exact
    # LF-canonical text and parsed mapping so any substantive drift is fatal.
    assert _lf_text(packaged) == _lf_text(checkout)
    assert yaml.safe_load(packaged) == yaml.safe_load(checkout)


@pytest.mark.parametrize(
    "name",
    [
        "exp_2wiki.yaml",
        "exp_b3_gate_only.yaml",
        "exp_b5_learned_scorer.yaml",
        "exp_fever.yaml",
        "exp_full_pipeline.yaml",
        "exp_hagrid.yaml",
        "exp_popqa.yaml",
        "exp_sample.yaml",
    ],
)
def test_packaged_gating_temperature_is_fixed_not_labelled_calibrated(name: str) -> None:
    config = yaml.safe_load(read_experiment_config_bytes(name))
    gating = config["gating"]

    assert "calibration_temp" not in gating
    assert gating["softmax_temperature"] == 1.0


def test_packaged_learned_scorer_config_defaults_fail_closed() -> None:
    config = yaml.safe_load(read_experiment_config_bytes("exp_b5_learned_scorer.yaml"))
    scorer = config["scorer"]

    assert scorer["use_learned"] is True
    assert scorer["allow_unsafe_pickle"] is False
    assert scorer["learned_model_metadata_sha256"] == ""


def test_packaged_evaluator_fixture_is_the_canonical_checkout_artifact() -> None:
    packaged = read_evaluator_sanity_bytes()
    checkout = (REPO_ROOT / "tests" / "data" / "evaluator_sanity_v1.json").read_bytes()

    assert packaged == checkout
    assert hashlib.sha256(packaged).hexdigest() == PACKAGED_FIXTURE_ARTIFACT_SHA256
    fixture = load_sanity_fixture_bytes(packaged)
    assert fixture["schema_version"] == "evaluator-sanity-v1"
    assert EVALUATOR_SANITY_V1_CONTENT_SHA256 == (
        "4d1a496ab46dd4addc9123615ac3b4b56b96a60ae70f0b5fd8c14a01ec900863"
    )


def test_pipeline_default_ignores_shadow_config_in_arbitrary_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow_dir = tmp_path / "configs"
    shadow_dir.mkdir()
    (shadow_dir / DEFAULT_PIPELINE_CONFIG).write_text("- malicious-shadow\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    pipe = Pipeline(mock_mode=True)

    assert pipe.cfg["seed"] == 42
    assert pipe.cfg["models"]["generator"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert pipe._config_path == experiment_config_identity(DEFAULT_PIPELINE_CONFIG)


def test_explicit_config_path_is_literal_and_in_memory_config_can_bypass_it(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="does not exist or is not a file"):
        Pipeline(config_path=str(missing), mock_mode=True)

    supplied: Dict[str, Any] = {
        "seed": 9,
        "retriever": {"top_k": 1},
        "gating": {"enabled": False},
    }
    pipe = Pipeline(config_path=str(missing), config=supplied, mock_mode=True)
    assert pipe.cfg is supplied


@pytest.mark.parametrize("contents", [b"", b"[]\n", b"{}\n", b"[unterminated\n"])
def test_explicit_empty_nonmapping_or_malformed_config_fails_closed(
    tmp_path: Path, contents: bytes
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        _load_config(str(path))


def test_console_run_parser_passes_package_default_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import factuality_rag.cli.__main__ as cli
    import factuality_rag.pipeline.orchestrator as orchestrator

    seen: Dict[str, Any] = {}

    class FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def run(self, query: str, **kwargs: Any) -> tuple:
            return "answer", [], {}, "low"

    monkeypatch.setattr(orchestrator, "Pipeline", FakePipeline)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["factuality-rag", "run", "--query", "test", "--mock-mode"],
    )

    cli.main()

    assert seen["config_path"] is None
    assert "Answer:      answer" in capsys.readouterr().out


def test_console_explicit_missing_config_fails_before_pipeline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.cli.__main__ as cli

    missing = tmp_path / "missing.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "factuality-rag",
            "run",
            "--query",
            "test",
            "--mock-mode",
            "--config",
            str(missing),
        ],
    )

    with pytest.raises(FileNotFoundError):
        cli.main()


def _experiment_args(tmp_path: Path, config: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        dataset=None,
        split=None,
        sample=None,
        seed=None,
        run_id=None,
        override=[],
        mock=True,
        runs_dir=str(tmp_path / "runs"),
    )


def test_experiment_runner_default_ignores_cwd_and_records_logical_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.experiment_runner as runner

    shadow_dir = tmp_path / "configs"
    shadow_dir.mkdir()
    (shadow_dir / DEFAULT_EXPERIMENT_CONFIG).write_text("- malicious-shadow\n", encoding="utf-8")
    seen: Dict[str, Any] = {}
    extracted: Dict[str, Any] = {}
    sentinel_queries = ["synthetic packaging-test query"]
    sentinel_references = ["synthetic packaging-test reference"]

    def fake_run(**kwargs: Any) -> Dict[str, Any]:
        seen.update(kwargs)
        return {
            "run_id": "test-run",
            "run_dir": str(tmp_path / "unused"),
            "predictions": [],
            "metrics": {},
        }

    def fake_extract(**kwargs: Any) -> tuple[list[str], list[str]]:
        extracted.update(kwargs)
        return sentinel_queries, sentinel_references

    monkeypatch.setattr(runner, "_parse_args", lambda: _experiment_args(tmp_path, None))
    monkeypatch.setattr(runner, "_extract_queries_and_references", fake_extract)
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    runner.main()

    assert seen["config"]["seed"] == 42
    assert seen["config"]["data"] == {
        "datasets": [{"name": "natural_questions", "split": "validation"}],
        "dev_sample_size": 500,
    }
    assert seen["config_path"] == experiment_config_identity(DEFAULT_EXPERIMENT_CONFIG)
    assert extracted == {
        "dataset_name": "natural_questions",
        "split": "validation",
        "sample": 500,
        "seed": 42,
    }
    assert seen["queries"] is sentinel_queries
    assert seen["references"] is sentinel_references


def test_experiment_runner_explicit_missing_config_fails_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factuality_rag.experiment_runner as runner

    called = False

    def fake_run(**kwargs: Any) -> Dict[str, Any]:
        nonlocal called
        called = True
        return {}

    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: _experiment_args(tmp_path, str(missing)),
    )
    monkeypatch.setattr(runner, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        runner.main()
    assert called is False
