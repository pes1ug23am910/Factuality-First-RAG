"""Fail-closed tests for the platform-specific Pyserini boundaries."""

from __future__ import annotations

import hashlib
import builtins
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable

import pytest

import factuality_rag.retriever.hybrid as hybrid
from factuality_rag.retriever.hybrid import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]


def _completed(run_kwargs: Any, *, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> Any:
    run_kwargs["stdout"].write(stdout)
    run_kwargs["stdout"].flush()
    run_kwargs["stderr"].write(stderr)
    run_kwargs["stderr"].flush()
    return subprocess.CompletedProcess([], returncode)


def _response(hits: Any, request_sha256: str) -> bytes:
    return (
        json.dumps(
            {"schema_version": 1, "request_sha256": request_sha256, "hits": hits},
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _request_sha(query: str = "query", k: int = 2, index_path: str = "unused") -> str:
    request = {
        "schema_version": 1,
        "index_path": index_path,
        "query": query,
        "k": k,
    }
    payload = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _windows_retriever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HybridRetriever:
    index_path = tmp_path / "lucene ü & (safe)"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"marker")
    java_home = tmp_path / "jdk-21"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin" / "java.exe").write_bytes(b"test executable marker")
    (java_home / "release").write_text('JAVA_VERSION="21.0.12"\n', encoding="utf-8")
    monkeypatch.setenv("FACTUALITY_RAG_JAVA_HOME", str(java_home))
    monkeypatch.setattr(hybrid.sys, "platform", "win32")
    return HybridRetriever("unused", str(index_path))


def test_windows_worker_success_is_strict_and_does_not_import_pyserini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pyserini" or name.startswith("pyserini.")
    }
    before_environment = dict(os.environ)

    def fake_run(command: Any, **kwargs: Any) -> Any:
        assert command == [sys.executable, "-m", "factuality_rag.retriever.pyserini_worker"]
        assert "capture_output" not in kwargs
        assert hasattr(kwargs["stdout"], "write")
        assert hasattr(kwargs["stderr"], "write")
        assert "text" not in kwargs
        assert "encoding" not in kwargs
        assert "errors" not in kwargs
        assert kwargs["timeout"] == hybrid._PYSERINI_WORKER_TIMEOUT_SECONDS
        assert kwargs["check"] is False
        assert kwargs["shell"] is False
        assert kwargs["env"] is not os.environ
        assert kwargs["env"]["JAVA_HOME"] == str((tmp_path / "jdk-21").resolve())
        assert kwargs["env"]["PATH"].startswith(
            str((tmp_path / "jdk-21" / "bin").resolve()) + os.pathsep
        )
        assert kwargs["env"]["PYTHONUTF8"] == "1"
        assert kwargs["input"].count(b"\n") == 1
        request = json.loads(kwargs["input"].decode("utf-8"))
        assert request == {
            "schema_version": 1,
            "index_path": str(Path(retriever.pyserini_index_path).resolve()),
            "query": "where is paris\n雪 & | $(ignored)",
            "k": 2,
        }
        request_sha256 = hashlib.sha256(kwargs["input"][:-1]).hexdigest()
        return _completed(
            kwargs,
            stdout=_response(
                [{"docid": "doc-a", "score": 4.25}, {"docid": "doc-b", "score": 1}],
                request_sha256,
            ),
            stderr=(
                b"WARNING: Using incubator modules: jdk.incubator.vector\r\n"
                b"Aug 19, 2026 11:41:02 AM "
                b"org.apache.lucene.store.MemorySegmentIndexInputProvider <init>\r\n"
                b"INFO: Using MemorySegmentIndexInput with Java 21; to disable start with "
                b"-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false\r\n"
            ),
        )

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)

    assert retriever._bm25_search("where is paris\n雪 & | $(ignored)", 2) == {
        "doc-a": 4.25,
        "doc-b": 1.0,
    }
    after_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pyserini" or name.startswith("pyserini.")
    }
    assert after_modules == before_modules
    assert dict(os.environ) == before_environment
    assert retriever._lucene_searcher is None


def _valid_empty_response(request_bytes: bytes) -> bytes:
    return _response([], hashlib.sha256(request_bytes[:-1]).hexdigest())


@pytest.mark.parametrize(
    ("response_factory", "stderr", "returncode"),
    [
        (_valid_empty_response, b"worker error", 3),
        (lambda request: b"not json\n", b"", 0),
        (lambda request: _valid_empty_response(request) + b"unexpected log\n", b"", 0),
        (lambda request: b"log prefix " + _valid_empty_response(request), b"", 0),
        (_valid_empty_response, b"Windows fatal exception: access violation", 0),
        (_valid_empty_response, b"Fatal Python error: Segmentation fault", 0),
        (_valid_empty_response, b"unexpected warning", 0),
        (lambda request: _response([], "0" * 64), b"", 0),
        (
            lambda request: (
                b'{"schema_version":1,"request_sha256":"'
                + hashlib.sha256(request[:-1]).hexdigest().encode("ascii")
                + b'","hits":[{"docid":"a","score":NaN}]}\n'
            ),
            b"",
            0,
        ),
        (
            lambda request: (
                b'{"schema_version":1,"request_sha256":"'
                + hashlib.sha256(request[:-1]).hexdigest().encode("ascii")
                + b'","hits":[{"docid":"a","score":1e999}]}\n'
            ),
            b"",
            0,
        ),
        (
            lambda request: _response(
                [{"docid": "same", "score": 1.0}, {"docid": "same", "score": 2.0}],
                hashlib.sha256(request[:-1]).hexdigest(),
            ),
            b"",
            0,
        ),
        (
            lambda request: _response(
                [{"docid": "a", "score": True}],
                hashlib.sha256(request[:-1]).hexdigest(),
            ),
            b"",
            0,
        ),
        (
            lambda request: _response(
                [{"docid": " padded ", "score": 1.0}],
                hashlib.sha256(request[:-1]).hexdigest(),
            ),
            b"",
            0,
        ),
        (
            lambda request: _response(
                [{"docid": "control\nvalue", "score": 1.0}],
                hashlib.sha256(request[:-1]).hexdigest(),
            ),
            b"",
            0,
        ),
        (
            lambda request: _response(
                [{"docid": "zero\u200bwidth", "score": 1.0}],
                hashlib.sha256(request[:-1]).hexdigest(),
            ),
            b"",
            0,
        ),
        (
            lambda request: _response(
                [{"docid": "a", "score": 1.0, "rank": 1}],
                hashlib.sha256(request[:-1]).hexdigest(),
            ),
            b"",
            0,
        ),
        (lambda request: b'{"schema_version":2,"hits":[]}\n', b"", 0),
        (
            lambda request: (
                b'{"schema_version":true,"request_sha256":"'
                + hashlib.sha256(request[:-1]).hexdigest().encode("ascii")
                + b'","hits":[]}\n'
            ),
            b"",
            0,
        ),
        (lambda request: b'{"schema_version":1,"hits":[],"extra":true}\n', b"", 0),
        (
            lambda request: b'{"schema_version":1,"schema_version":1,"hits":[]}\n',
            b"",
            0,
        ),
        (lambda request: b"\xff\xfe", b"", 0),
    ],
)
def test_windows_worker_output_failures_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_factory: Callable[[bytes], bytes],
    stderr: bytes,
    returncode: int,
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _completed(
            kwargs,
            stdout=response_factory(kwargs["input"]),
            stderr=stderr,
            returncode=returncode,
        )

    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(hybrid.PyseriniWorkerError):
        retriever._bm25_search("query", 2)


def test_windows_worker_timeout_returns_no_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    opened_streams = []

    def time_out(*args: Any, **kwargs: Any) -> Any:
        opened_streams.extend([kwargs["stdout"], kwargs["stderr"]])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(hybrid.subprocess, "run", time_out)
    with pytest.raises(hybrid.PyseriniWorkerError):
        retriever._bm25_search("query", 2)
    assert opened_streams and all(stream.closed for stream in opened_streams)


def test_windows_worker_os_error_returns_no_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("could not start worker")),
    )

    with pytest.raises(hybrid.PyseriniWorkerError):
        retriever._bm25_search("query", 2)


def test_windows_worker_rejects_unsupported_java_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    unsupported_home = tmp_path / "jdk-25"
    (unsupported_home / "bin").mkdir(parents=True)
    (unsupported_home / "bin" / "java.exe").write_bytes(b"test executable marker")
    (unsupported_home / "release").write_text('JAVA_VERSION="25.0.2"\n', encoding="utf-8")
    monkeypatch.setenv("FACTUALITY_RAG_JAVA_HOME", str(unsupported_home))
    before_environment = dict(os.environ)
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unsupported Java must be rejected before spawn"),
    )

    with pytest.raises(hybrid.PyseriniWorkerError, match="Java 21"):
        retriever._bm25_search("query", 2)
    assert dict(os.environ) == before_environment


def test_collection_directory_is_rejected_without_spawning_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_path = tmp_path / "collection-only"
    collection_path.mkdir()
    (collection_path / "docs.jsonl").write_text('{"id":"1","contents":"text"}\n')
    retriever = HybridRetriever("unused", str(collection_path))
    monkeypatch.setattr(hybrid.sys, "platform", "win32")
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a collection directory is not a Lucene index"),
    )

    with pytest.raises(hybrid.BM25BackendError, match="No Lucene index marker"):
        retriever._bm25_search("query", 2)


def test_missing_real_index_fails_closed(tmp_path: Path) -> None:
    retriever = HybridRetriever("unused", str(tmp_path / "missing"))

    with pytest.raises(hybrid.BM25BackendError, match="index not found"):
        retriever._bm25_search("query", 2)


def test_worker_rejects_invalid_request_before_loading_pyserini() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "factuality_rag.retriever.pyserini_worker"],
        input=b"{}\n",
        capture_output=True,
        timeout=10,
        check=False,
        shell=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == b""
    stderr = completed.stderr.decode("utf-8", errors="strict")
    assert "invalid request schema" in stderr
    assert "incubator modules" not in stderr


def test_worker_rejects_oversized_stdin_before_loading_pyserini() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "factuality_rag.retriever.pyserini_worker"],
        input=b"x" * (hybrid._PYSERINI_MAX_REQUEST_BYTES + 1),
        capture_output=True,
        timeout=10,
        check=False,
        shell=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == b""
    stderr = completed.stderr.decode("utf-8", errors="strict")
    assert "request exceeds the protocol limit" in stderr
    assert "incubator modules" not in stderr


@pytest.mark.parametrize(
    ("query", "k"),
    [
        pytest.param("query", hybrid._PYSERINI_MAX_K + 1, id="k-too-large"),
        pytest.param("x" * (hybrid._PYSERINI_MAX_QUERY_BYTES + 1), 1, id="query-too-large"),
    ],
)
def test_parent_request_limits_apply_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    k: int,
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("oversized request must be rejected before spawn"),
    )

    with pytest.raises(ValueError):
        hybrid._search_pyserini_in_worker(Path(retriever.pyserini_index_path), query, k)


@pytest.mark.parametrize(
    ("stream_name", "limit_name"),
    [("stdout", "_PYSERINI_MAX_STDOUT_BYTES"), ("stderr", "_PYSERINI_MAX_STDERR_BYTES")],
)
def test_parent_checks_temporary_output_size_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    limit_name: str,
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    monkeypatch.setattr(hybrid, limit_name, 16)

    def write_oversized_output(*args: Any, **kwargs: Any) -> Any:
        kwargs[stream_name].write(b"x" * 17)
        kwargs[stream_name].flush()
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(hybrid.subprocess, "run", write_oversized_output)
    with pytest.raises(hybrid.PyseriniWorkerError, match="exceeded the protocol limit"):
        retriever._bm25_search("query", 2)


def test_windows_worker_rejects_more_hits_than_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = _windows_retriever(tmp_path, monkeypatch)
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            kwargs,
            stdout=_response(
                [{"docid": "a", "score": 1.0}, {"docid": "b", "score": 0.5}],
                _request_sha("query", 1, str(Path(retriever.pyserini_index_path).resolve())),
            ),
        ),
    )

    with pytest.raises(hybrid.PyseriniWorkerError):
        retriever._bm25_search("query", 1)


def test_non_windows_uses_credential_free_direct_binding_and_caches_searcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "lucene"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"marker")
    retriever = HybridRetriever("unused", str(index_path))
    monkeypatch.setattr(hybrid.sys, "platform", "linux")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    class Hit:
        def __init__(self, docid: str, score: float) -> None:
            self.docid = docid
            self.score = score

    autoclass_calls: list[str] = []
    constructor_calls: list[str] = []
    search_calls: list[tuple[str, int]] = []
    document_count_calls = 0

    class Searcher:
        def __init__(self, path: str) -> None:
            constructor_calls.append(path)

        def get_total_num_docs(self) -> int:
            nonlocal document_count_calls
            document_count_calls += 1
            return 2

        def search(self, query: str, k: int) -> Any:
            search_calls.append((query, k))
            return [Hit("doc-a", 2.5), Hit("doc-b", 1.25)][:k]

    def fake_autoclass(name: str) -> type[Searcher]:
        autoclass_calls.append(name)
        return Searcher

    pyserini_package = ModuleType("pyserini")
    pyserini_package.__path__ = []  # type: ignore[attr-defined]
    pyclass_module = ModuleType("pyserini.pyclass")
    pyclass_module.autoclass = fake_autoclass  # type: ignore[attr-defined]
    pyserini_package.pyclass = pyclass_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyserini", pyserini_package)
    monkeypatch.setitem(sys.modules, "pyserini.pyclass", pyclass_module)

    aggregate_imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyserini.search" or name.startswith("pyserini.search."):
            aggregate_imports.append(name)
            raise AssertionError("aggregate Pyserini search modules must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("non-Windows path must not spawn a worker"),
    )

    assert retriever._bm25_search("first query", 2) == {"doc-a": 2.5, "doc-b": 1.25}
    assert retriever._bm25_search("second query", 1) == {"doc-a": 2.5}
    assert autoclass_calls == ["io.anserini.search.SimpleSearcher"]
    assert constructor_calls == [str(index_path.resolve())]
    assert document_count_calls == 1
    assert search_calls == [("first query", 2), ("second query", 1)]
    assert aggregate_imports == []
    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_ADMIN_KEY" not in os.environ


def test_non_windows_direct_binding_initialization_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "lucene"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"marker")
    retriever = HybridRetriever("unused", str(index_path))
    monkeypatch.setattr(hybrid.sys, "platform", "linux")

    def failing_autoclass(name: str) -> None:
        assert name == "io.anserini.search.SimpleSearcher"
        raise RuntimeError("JVM binding failed")

    pyserini_package = ModuleType("pyserini")
    pyserini_package.__path__ = []  # type: ignore[attr-defined]
    pyclass_module = ModuleType("pyserini.pyclass")
    pyclass_module.autoclass = failing_autoclass  # type: ignore[attr-defined]
    pyserini_package.pyclass = pyclass_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyserini", pyserini_package)
    monkeypatch.setitem(sys.modules, "pyserini.pyclass", pyclass_module)

    with pytest.raises(hybrid.BM25BackendError, match="BM25 search failed") as exc_info:
        retriever._bm25_search("query", 1)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "JVM binding failed"
    assert retriever._lucene_searcher is None


def test_non_windows_search_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "lucene"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"marker")
    retriever = HybridRetriever("unused", str(index_path))

    class BrokenSearcher:
        def search(self, query: str, k: int) -> Any:
            raise RuntimeError("backend failed")

    retriever._lucene_searcher = BrokenSearcher()
    monkeypatch.setattr(hybrid.sys, "platform", "linux")

    with pytest.raises(hybrid.BM25BackendError, match="BM25 search failed") as exc_info:
        retriever._bm25_search("query", 1)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "backend failed"


@pytest.mark.integration
def test_real_windows_worker_survives_repeated_outer_process_queries() -> None:
    if sys.platform != "win32":
        pytest.skip("Windows-only JNI isolation gate")
    java_override = os.environ.get("FACTUALITY_RAG_JAVA_HOME")
    if not java_override:
        pytest.skip("FACTUALITY_RAG_JAVA_HOME is not configured")
    index_path = REPO_ROOT / "indexes" / "wiki100k_lucene"
    if not index_path.is_dir() or not any(index_path.glob("segments_*")):
        pytest.skip("local Lucene integration index is unavailable")

    script = (
        "import json\n"
        "from factuality_rag.retriever.hybrid import HybridRetriever\n"
        f"r = HybridRetriever('unused', {str(index_path)!r})\n"
        "for q in ('capital of France', 'who wrote Hamlet'):\n"
        "    print(json.dumps(r._bm25_search(q, 3), sort_keys=True, allow_nan=False))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=180,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stderr == b""
    lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
    assert len(lines) == 2
    for line in lines:
        scores = json.loads(line)
        assert 0 < len(scores) <= 3
        assert all(
            isinstance(doc_id, str)
            and doc_id
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            for doc_id, score in scores.items()
        )
