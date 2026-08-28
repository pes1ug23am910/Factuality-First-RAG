"""Artifact-contract tests for dense and sparse index builders."""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import factuality_rag.index.builder as builder
from factuality_rag.cli import __main__ as cli


def _write_corpus(path: Path) -> list[dict[str, str]]:
    documents = [
        {"id": "doc-a", "text": "alpha text"},
        {"id": "doc-b", "text": "beta text"},
        {"id": "doc-c", "text": "gamma text"},
    ]
    path.write_text(
        "".join(json.dumps(document) + "\n" for document in documents),
        encoding="utf-8",
    )
    return documents


def test_faiss_builder_persists_exact_id_and_text_sidecars(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    documents = _write_corpus(corpus)
    index_path = tmp_path / "dense.faiss"

    builder.build_faiss_index(
        str(corpus),
        out_path=str(index_path),
        mock_mode=True,
        dim=4,
    )

    assert json.loads(index_path.with_suffix(".ids.json").read_text(encoding="utf-8")) == [
        document["id"] for document in documents
    ]
    sidecar = [
        json.loads(line)
        for line in index_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sidecar == documents


def test_faiss_builder_canonicalizes_missing_and_integer_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"text":"alpha"}\n{"id":7,"text":"beta"}\n',
        encoding="utf-8",
    )
    index_path = tmp_path / "dense.faiss"

    builder.build_faiss_index(
        str(corpus),
        out_path=str(index_path),
        mock_mode=True,
        dim=4,
    )

    sidecar = [
        json.loads(line)
        for line in index_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sidecar == [{"id": "0", "text": "alpha"}, {"id": "7", "text": "beta"}]
    assert json.loads(index_path.with_suffix(".ids.json").read_text(encoding="utf-8")) == [
        "0",
        "7",
    ]


def test_build_index_cli_feeds_faiss_canonical_sidecar_to_lucene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "dense.faiss"
    canonical_sidecar = index_path.with_suffix(".jsonl")
    calls: list[tuple[str, str, Any]] = []

    def fake_faiss(**kwargs: Any) -> str:
        canonical_sidecar.write_text('{"id":"0","text":"alpha"}\n', encoding="utf-8")
        return str(index_path.resolve())

    def fake_lucene(**kwargs: Any) -> str:
        calls.append((kwargs["jsonl_path"], kwargs["out_dir"], kwargs.get("dev_sample_size")))
        return str((tmp_path / "lucene").resolve())

    monkeypatch.setattr(builder, "build_faiss_index", fake_faiss)
    monkeypatch.setattr(builder, "build_pyserini_index", fake_lucene)
    args = Namespace(
        dry_run=False,
        corpus=str(tmp_path / "source.jsonl"),
        embedding_model="mock",
        faiss_out=str(index_path),
        mock_mode=True,
        faiss_type="hnsw_flat",
        dev_sample_size=1,
        pyserini_out=str(tmp_path / "lucene"),
    )

    cli._cmd_build_index(args)

    assert calls == [(str(canonical_sidecar), str(tmp_path / "lucene"), None)]


@pytest.mark.parametrize("count", [1, 50, 255])
def test_ivfpq_fails_early_when_training_corpus_is_too_small(tmp_path: Path, count: int) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps({"id": str(i), "text": f"text {i}"}) + "\n" for i in range(count)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 256 training vectors"):
        builder.build_faiss_index(
            str(corpus),
            out_path=str(tmp_path / "dense.faiss"),
            mock_mode=True,
            faiss_type="ivfpq",
            dim=16,
        )


def test_ivfpq_size_validation_precedes_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"0","text":"alpha"}\n', encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "_get_sentence_transformer",
        lambda: pytest.fail("embedding model loaded before IVFPQ size validation"),
    )

    with pytest.raises(ValueError, match="at least 256 training vectors"):
        builder.build_faiss_index(
            str(corpus),
            out_path=str(tmp_path / "dense.faiss"),
            mock_mode=False,
            faiss_type="ivfpq",
        )


def test_bounded_jsonl_load_stops_after_requested_records(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"0","text":"first"}\n'
        '{"id":"1","text":"second"}\n'
        "this trailing record is invalid JSON\n",
        encoding="utf-8",
    )

    assert builder.load_jsonl(str(corpus), limit=2) == [
        {"id": "0", "text": "first"},
        {"id": "1", "text": "second"},
    ]


def test_lucene_builder_moves_only_verified_completed_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    target = tmp_path / "lucene"
    monkeypatch.setattr(builder.sys, "platform", "linux")
    java_home = tmp_path / "jdk-21"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin" / "java").write_bytes(b"test executable marker")
    (java_home / "release").write_text('JAVA_VERSION="21.0.12"\n', encoding="utf-8")
    monkeypatch.setenv("FACTUALITY_RAG_JAVA_HOME", str(java_home))

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        output = Path(command[command.index("--index") + 1])
        output.mkdir(parents=True)
        (output / "segments_1").write_bytes(b"verified marker")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    result = builder.build_pyserini_index(str(corpus), str(target), threads=2)

    assert Path(result) == target.resolve()
    assert (target / "segments_1").read_bytes() == b"verified marker"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_pyserini_index(str(corpus), str(target))


def test_lucene_builder_preflights_java_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    monkeypatch.setattr(builder.sys, "platform", "linux")
    monkeypatch.delenv("FACTUALITY_RAG_JAVA_HOME", raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(
        builder,
        "prepare_pyserini_collection",
        lambda *args, **kwargs: pytest.fail("collection work must follow Java preflight"),
    )

    with pytest.raises(RuntimeError, match="Java 21 is required"):
        builder.build_pyserini_index(str(corpus), str(tmp_path / "lucene"))


def test_lucene_builder_reports_bounded_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    monkeypatch.setattr(builder.sys, "platform", "linux")
    java_home = tmp_path / "jdk-21"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin" / "java").write_bytes(b"test executable marker")
    (java_home / "release").write_text('JAVA_VERSION="21.0.12"\n', encoding="utf-8")
    monkeypatch.setenv("FACTUALITY_RAG_JAVA_HOME", str(java_home))
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            7,
            stdout=b"",
            stderr=b"diagnostic from Java",
        ),
    )

    with pytest.raises(RuntimeError, match="diagnostic from Java"):
        builder.build_pyserini_index(str(corpus), str(tmp_path / "lucene"))
