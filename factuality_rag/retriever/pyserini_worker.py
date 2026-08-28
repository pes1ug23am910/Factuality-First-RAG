"""Single-request Pyserini worker used to isolate JNI on Windows."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List
import unicodedata

_PROTOCOL_VERSION = 1
_MAX_K = 1_000
_MAX_QUERY_BYTES = 100_000
_MAX_REQUEST_BYTES = 200_000


def _reject_duplicate_keys(pairs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _load_request(raw: str) -> Dict[str, Any]:
    payload = raw[:-1] if raw.endswith("\n") else raw
    if not payload or "\n" in payload or "\r" in payload or payload != payload.strip():
        raise ValueError("request must contain exactly one JSON line")
    value = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_keys,
        parse_float=_parse_finite_float,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "index_path",
        "query",
        "k",
    }:
        raise ValueError("invalid request schema")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != _PROTOCOL_VERSION
    ):
        raise ValueError("unsupported protocol version")
    if not isinstance(value["index_path"], str) or not value["index_path"]:
        raise ValueError("index_path must be a non-empty string")
    if not isinstance(value["query"], str):
        raise ValueError("query must be a string")
    if len(value["query"].encode("utf-8")) > _MAX_QUERY_BYTES:
        raise ValueError("query exceeds the protocol limit")
    if (
        isinstance(value["k"], bool)
        or not isinstance(value["k"], int)
        or not 1 <= value["k"] <= _MAX_K
    ):
        raise ValueError(f"k must be an integer in [1, {_MAX_K}]")
    return value


def _validate_java_environment() -> None:
    """Reject unsupported Java runtimes before importing jnius."""
    selected_home = os.environ.get("JAVA_HOME")
    if not selected_home:
        raise RuntimeError("JAVA_HOME is required")
    java_home = Path(selected_home).resolve(strict=True)
    java_name = "java.exe" if sys.platform == "win32" else "java"
    java_executable = java_home / "bin" / java_name
    if not java_home.is_dir() or not java_executable.is_file():
        raise RuntimeError("JAVA_HOME does not contain a Java runtime")
    release = (java_home / "release").read_text(encoding="utf-8")
    match = re.search(r'^JAVA_VERSION="([0-9]+)(?:[._][^"]*)?"$', release, flags=re.MULTILINE)
    if match is None or int(match.group(1)) != 21:
        raise RuntimeError("Pyserini worker requires Java 21")


def _search(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    index_path = Path(request["index_path"]).resolve(strict=True)
    if not index_path.is_dir():
        raise FileNotFoundError(f"Lucene index directory not found: {index_path}")
    if not any(
        child.is_file() and child.name.startswith("segments_") and child.stat().st_size > 0
        for child in index_path.iterdir()
    ):
        raise ValueError(f"Lucene index marker not found: {index_path}")

    _validate_java_environment()
    # Keep Python-level library chatter off protocol stdout. Native writes are
    # still visible to the parent and cause it to reject the response.
    with contextlib.redirect_stdout(sys.stderr):
        from pyserini.pyclass import autoclass  # type: ignore[import-untyped]

        # Import the sparse Java binding directly. Importing the aggregate
        # ``pyserini.search.lucene`` package eagerly initializes unrelated
        # optional encoders before Lucene search is available.
        simple_searcher = autoclass("io.anserini.search.SimpleSearcher")
        searcher = simple_searcher(str(index_path))
        raw_hits = searcher.search(request["query"], request["k"])

    hits: List[Dict[str, Any]] = []
    seen = set()
    for hit in raw_hits:
        doc_id = hit.docid
        if (
            not isinstance(doc_id, str)
            or not doc_id
            or doc_id != doc_id.strip()
            or any(unicodedata.category(character).startswith("C") for character in doc_id)
            or doc_id in seen
        ):
            raise ValueError("Pyserini returned an invalid or duplicate document id")
        score = float(hit.score)
        if not math.isfinite(score):
            raise ValueError("Pyserini returned a non-finite score")
        seen.add(doc_id)
        hits.append({"docid": doc_id, "score": score})
    if len(hits) > request["k"]:
        raise ValueError("Pyserini returned more hits than requested")
    return hits


def main() -> int:
    try:
        raw_request_bytes = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
        if len(raw_request_bytes) > _MAX_REQUEST_BYTES:
            raise ValueError("request exceeds the protocol limit")
        raw_request = raw_request_bytes.decode("utf-8", errors="strict")
        request_payload = raw_request[:-1] if raw_request.endswith("\n") else raw_request
        request = _load_request(raw_request)
        response = {
            "schema_version": _PROTOCOL_VERSION,
            "request_sha256": hashlib.sha256(request_payload.encode("utf-8")).hexdigest(),
            "hits": _search(request),
        }
        sys.stdout.buffer.write(
            (
                json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"Pyserini worker failed ({type(exc).__name__}): {exc}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid triggering JNI/JVM finalizers during ordinary interpreter teardown.
    os._exit(exit_code)
