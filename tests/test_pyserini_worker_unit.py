"""In-process tests for the Pyserini worker's protocol and validation logic."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, List

import pytest

import factuality_rag.retriever.pyserini_worker as worker


def _request(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "index_path": "C:/index",
        "query": "capital of France",
        "k": 3,
    }
    value.update(overrides)
    return value


def _request_json(**overrides: Any) -> str:
    return json.dumps(_request(**overrides), separators=(",", ":"))


def _java_home(tmp_path: Path, version: str = "21.0.12") -> Path:
    java_home = tmp_path / f"jdk-{version}"
    (java_home / "bin").mkdir(parents=True)
    java_name = "java.exe" if sys.platform == "win32" else "java"
    (java_home / "bin" / java_name).write_bytes(b"test executable marker")
    (java_home / "release").write_text(
        f'IMPLEMENTOR="Test"\nJAVA_VERSION="{version}"\n', encoding="utf-8"
    )
    return java_home


def _lucene_index(tmp_path: Path) -> Path:
    index_path = tmp_path / "lucene"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"marker")
    return index_path


def test_worker_load_request_accepts_exact_json_with_optional_final_lf() -> None:
    expected = _request()
    assert worker._load_request(_request_json()) == expected
    assert worker._load_request(_request_json() + "\n") == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param(" " + _request_json(), id="leading-space"),
        pytest.param(_request_json() + " \n", id="trailing-space"),
        pytest.param(_request_json() + "\n\n", id="extra-line"),
        pytest.param(_request_json() + "\r\n", id="crlf"),
        pytest.param("[]", id="not-object"),
        pytest.param("{}", id="wrong-fields"),
        pytest.param(
            '{"schema_version":1,"schema_version":1,"index_path":"x","query":"q","k":1}',
            id="duplicate-key",
        ),
        pytest.param(
            '{"schema_version":1,"index_path":"x","query":"q","k":NaN}',
            id="nonfinite-constant",
        ),
        pytest.param(
            '{"schema_version":1,"index_path":"x","query":"q","k":1e999}',
            id="overflow-float",
        ),
    ],
)
def test_worker_load_request_rejects_non_strict_json(raw: str) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        worker._load_request(raw)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"schema_version": True}, id="bool-version"),
        pytest.param({"schema_version": 1.0}, id="float-version"),
        pytest.param({"schema_version": 2}, id="wrong-version"),
        pytest.param({"index_path": ""}, id="empty-path"),
        pytest.param({"index_path": 1}, id="numeric-path"),
        pytest.param({"query": 1}, id="numeric-query"),
        pytest.param({"k": True}, id="bool-k"),
        pytest.param({"k": 1.5}, id="float-k"),
        pytest.param({"k": 0}, id="zero-k"),
        pytest.param({"k": worker._MAX_K + 1}, id="excess-k"),
    ],
)
def test_worker_load_request_rejects_invalid_field_types(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        worker._load_request(_request_json(**overrides))


def test_worker_load_request_rejects_oversized_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_MAX_QUERY_BYTES", 3)
    with pytest.raises(ValueError, match="query exceeds"):
        worker._load_request(_request_json(query="four"))


def test_worker_numeric_helpers_accept_finite_and_reject_nonfinite() -> None:
    assert worker._parse_finite_float("1.25") == 1.25
    with pytest.raises(ValueError, match="non-finite JSON number"):
        worker._parse_finite_float("1e999")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        worker._reject_non_finite("Infinity")


def test_worker_duplicate_key_helper() -> None:
    assert worker._reject_duplicate_keys([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="duplicate JSON key"):
        worker._reject_duplicate_keys([("a", 1), ("a", 2)])


def test_worker_java_preflight_requires_java_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAVA_HOME", raising=False)
    with pytest.raises(RuntimeError, match="JAVA_HOME is required"):
        worker._validate_java_environment()


def test_worker_java_preflight_rejects_missing_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        worker._validate_java_environment()


def test_worker_java_preflight_rejects_missing_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_home = tmp_path / "jdk"
    java_home.mkdir()
    monkeypatch.setenv("JAVA_HOME", str(java_home))
    with pytest.raises(RuntimeError, match="does not contain"):
        worker._validate_java_environment()


@pytest.mark.parametrize(
    "release_text",
    [
        pytest.param('JAVA_VERSION="25.0.2"\n', id="unsupported-major"),
        pytest.param('IMPLEMENTOR="Test"\n', id="missing-version"),
    ],
)
def test_worker_java_preflight_rejects_unsupported_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_text: str,
) -> None:
    java_home = _java_home(tmp_path)
    (java_home / "release").write_text(release_text, encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(java_home))
    with pytest.raises(RuntimeError, match="requires Java 21"):
        worker._validate_java_environment()


def test_worker_java_preflight_accepts_java_21(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_home = _java_home(tmp_path)
    monkeypatch.setenv("JAVA_HOME", str(java_home))
    worker._validate_java_environment()


def test_worker_java_preflight_accepts_linux_java_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_home = tmp_path / "jdk-linux"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin" / "java").write_bytes(b"test executable marker")
    (java_home / "release").write_text('JAVA_VERSION="21.0.12"\n', encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(java_home))
    monkeypatch.setattr(worker.sys, "platform", "linux")

    worker._validate_java_environment()


def _install_fake_pyserini(
    monkeypatch: pytest.MonkeyPatch, raw_hits: List[Any]
) -> List[tuple[str, Any]]:
    calls: List[tuple[str, Any]] = []

    class FakeSearcher:
        def __init__(self, index_path: str) -> None:
            calls.append(("init", index_path))

        def search(self, query: str, k: int) -> List[Any]:
            calls.append((query, k))
            return raw_hits

    def autoclass(class_name: str) -> Any:
        calls.append(("autoclass", class_name))
        return FakeSearcher

    package = ModuleType("pyserini")
    package.__path__ = []  # type: ignore[attr-defined]
    pyclass = ModuleType("pyserini.pyclass")
    pyclass.autoclass = autoclass  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyserini", package)
    monkeypatch.setitem(sys.modules, "pyserini.pyclass", pyclass)
    return calls


def test_worker_search_rejects_file_and_collection_directories(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_bytes(b"file")
    with pytest.raises(FileNotFoundError):
        worker._search(_request(index_path=str(file_path)))

    collection_path = tmp_path / "collection"
    collection_path.mkdir()
    (collection_path / "docs.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker not found"):
        worker._search(_request(index_path=str(collection_path)))


def test_worker_search_returns_validated_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = _lucene_index(tmp_path)
    hits = [
        SimpleNamespace(docid="doc-a", score=3.5),
        SimpleNamespace(docid="doc-b", score=2),
    ]
    calls = _install_fake_pyserini(monkeypatch, hits)
    monkeypatch.setattr(worker, "_validate_java_environment", lambda: None)

    assert worker._search(_request(index_path=str(index_path), query="query", k=2)) == [
        {"docid": "doc-a", "score": 3.5},
        {"docid": "doc-b", "score": 2.0},
    ]
    assert calls == [
        ("autoclass", "io.anserini.search.SimpleSearcher"),
        ("init", str(index_path.resolve())),
        ("query", 2),
    ]


@pytest.mark.parametrize(
    "hits",
    [
        pytest.param([SimpleNamespace(docid=1, score=1)], id="numeric-id"),
        pytest.param([SimpleNamespace(docid="", score=1)], id="empty-id"),
        pytest.param([SimpleNamespace(docid=" padded ", score=1)], id="padded-id"),
        pytest.param([SimpleNamespace(docid="zero\u200bwidth", score=1)], id="control-id"),
        pytest.param(
            [
                SimpleNamespace(docid="duplicate", score=1),
                SimpleNamespace(docid="duplicate", score=2),
            ],
            id="duplicate-id",
        ),
        pytest.param([SimpleNamespace(docid="doc", score=math.nan)], id="nonfinite-score"),
    ],
)
def test_worker_search_rejects_invalid_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hits: List[Any],
) -> None:
    index_path = _lucene_index(tmp_path)
    _install_fake_pyserini(monkeypatch, hits)
    monkeypatch.setattr(worker, "_validate_java_environment", lambda: None)
    with pytest.raises(ValueError):
        worker._search(_request(index_path=str(index_path), k=len(hits)))


def test_worker_search_rejects_more_hits_than_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = _lucene_index(tmp_path)
    hits = [
        SimpleNamespace(docid="doc-a", score=2),
        SimpleNamespace(docid="doc-b", score=1),
    ]
    _install_fake_pyserini(monkeypatch, hits)
    monkeypatch.setattr(worker, "_validate_java_environment", lambda: None)
    with pytest.raises(ValueError, match="more hits"):
        worker._search(_request(index_path=str(index_path), k=1))


def _stdio(monkeypatch: pytest.MonkeyPatch, stdin: bytes) -> tuple[io.BytesIO, io.StringIO]:
    stdout = io.BytesIO()
    stderr = io.StringIO()
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(stdin)))
    monkeypatch.setattr(worker.sys, "stdout", SimpleNamespace(buffer=stdout))
    monkeypatch.setattr(worker.sys, "stderr", stderr)
    return stdout, stderr


def test_worker_main_writes_hash_bound_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (_request_json() + "\n").encode("utf-8")
    stdout, stderr = _stdio(monkeypatch, raw)
    monkeypatch.setattr(
        worker,
        "_search",
        lambda request: [{"docid": "doc-a", "score": 1.5}],
    )

    assert worker.main() == 0
    response = json.loads(stdout.getvalue().decode("utf-8"))
    assert response["schema_version"] == 1
    assert response["hits"] == [{"docid": "doc-a", "score": 1.5}]
    assert len(response["request_sha256"]) == 64
    assert stderr.getvalue() == ""


@pytest.mark.parametrize(
    "stdin",
    [
        pytest.param(b"{}\n", id="invalid-schema"),
        pytest.param(b"\xff", id="invalid-utf8"),
        pytest.param(b"x" * (worker._MAX_REQUEST_BYTES + 1), id="oversized"),
    ],
)
def test_worker_main_reports_protocol_errors(monkeypatch: pytest.MonkeyPatch, stdin: bytes) -> None:
    stdout, stderr = _stdio(monkeypatch, stdin)
    assert worker.main() == 1
    assert stdout.getvalue() == b""
    assert stderr.getvalue().startswith("Pyserini worker failed (")


def test_worker_main_reports_search_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, stderr = _stdio(monkeypatch, (_request_json() + "\n").encode("utf-8"))

    def fail_search(request: dict[str, Any]) -> List[dict[str, Any]]:
        raise RuntimeError("search failed")

    monkeypatch.setattr(worker, "_search", fail_search)
    assert worker.main() == 1
    assert stdout.getvalue() == b""
    assert "RuntimeError" in stderr.getvalue()
