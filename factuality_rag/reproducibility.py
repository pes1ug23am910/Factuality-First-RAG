"""Deterministic hashing and immutable experiment-manifest utilities."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import AbstractSet, Any, Dict, Mapping, Optional, Sequence, Union, cast
from urllib.parse import parse_qsl, urlsplit

import yaml  # type: ignore[import-untyped]

MANIFEST_HASH_FIELD = "manifest_sha256"
RUN_MANIFEST_SCHEMA = "factuality-rag.run-manifest.v1"
DATASET_MANIFEST_SCHEMA = "factuality-rag.dataset-manifest.v1"
CORPUS_MANIFEST_SCHEMA = "factuality-rag.corpus-manifest.v1"
INDEX_MANIFEST_SCHEMA = "factuality-rag.index-manifest.v1"

_DATASET_MANIFEST_FIELDS = {
    "dataset_id",
    "example_count",
    "manifest_sha256",
    "schema",
    "source_snapshot_sha256",
}
_CORPUS_MANIFEST_FIELDS = {
    "corpus_id",
    "corpus_snapshot_sha256",
    "manifest_sha256",
    "passage_count",
    "schema",
}
_INDEX_MANIFEST_FIELDS = {
    "corpus_manifest_sha256",
    "corpus_snapshot_sha256",
    "exact_indexed_passage_count",
    "index_id",
    "manifest_sha256",
    "schema",
}
_RUN_MANIFEST_FIELDS = {
    "schema",
    "run_id",
    "run_kind",
    "created_at_utc",
    "git",
    "config",
    "config_sha256",
    "config_path",
    "config_file_sha256",
    "data",
    "retrieval",
    "evaluator",
    "model_revisions",
    "seed",
    "hardware",
    "software",
    "resource_ceilings",
    "output_paths",
    "mock",
    "manifest_sha256",
}
_RUN_DATA_FIELDS = {
    "dataset_manifest_path",
    "dataset_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "split_partition",
    "selected_example_ids_sha256",
    "selected_example_count",
}
_RUN_RETRIEVAL_FIELDS = {
    "corpus_manifest_path",
    "corpus_manifest_sha256",
    "index_manifest_path",
    "index_manifest_sha256",
    "exact_indexed_passage_count",
}
_RUN_EVALUATOR_FIELDS = {
    "fixture_path",
    "fixture_sha256",
    "fixture_content_sha256",
    "fixture_report",
    "fixture_report_sha256",
    "scorer_id",
    "scorer_revision",
    "production_gate_passed",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
_WINDOWS_RESERVED_COMPONENT_RE = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?|wss?)://[^\s\"'<>]+", re.IGNORECASE)
_KNOWN_SECRET_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"gh[pousr]_[A-Za-z0-9_]{4,}|"
    r"github_pat_[A-Za-z0-9_]{4,}|"
    r"glpat-[A-Za-z0-9_-]{4,}|"
    r"hf_[A-Za-z0-9]{8,}|"
    r"sk-[A-Za-z0-9_-]{8,}|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|"
    r"AKIA[0-9A-Z]{12,}|"
    r"AIza[0-9A-Za-z_-]{20,}"
    r")"
)
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_SENSITIVE_SINGLE_KEY_WORDS = {
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "passwd",
    "password",
    "secret",
    "token",
}
_SENSITIVE_KEY_SEQUENCES = {
    ("access", "key"),
    ("access", "token"),
    ("api", "key"),
    ("client", "secret"),
    ("private", "key"),
    ("refresh", "token"),
    ("secret", "access", "key"),
    ("security", "token"),
    ("session", "token"),
    ("signing", "key"),
}
_SENSITIVE_COMPACT_KEYS = {
    "accesskey",
    "accesstoken",
    "apikey",
    "clientsecret",
    "privatekey",
    "refreshtoken",
    "secretaccesskey",
    "securitytoken",
    "sessiontoken",
}
_SAFE_NON_SECRET_KEY_WORDS = {
    ("input", "tokens"),
    ("max", "tokens"),
    ("output", "tokens"),
    ("token", "budget"),
    ("token", "count"),
    ("token", "limit"),
    ("tokens", "used"),
    ("total", "tokens"),
}
_SENSITIVE_URL_KEYS = {
    "apikey",
    "awsaccesskeyid",
    "googleaccessid",
    "key",
    "keypairid",
    "policy",
    "sharedaccesssignature",
    "sig",
    "signature",
    "xamzcredential",
    "xamzsecuritytoken",
    "xamzsignature",
    "xgoogcredential",
    "xgoogsignature",
}
_EVALUATOR_REPORT_KEYS = {
    "atol",
    "case_count",
    "fixture_content_sha256",
    "fixture_sha256",
    "passed",
    "production_gate_passed",
    "results",
    "schema_version",
    "scorer_id",
    "scorer_revision",
}
_EVALUATOR_SCHEMA_VERSION = "evaluator-sanity-v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* as canonical UTF-8 JSON.

    NaN and infinities are rejected so hashes cannot depend on a parser's
    non-standard floating-point behavior.
    """

    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON for *value*."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Union[Path, str], chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_artifact_path(path: str) -> str:
    """Validate and normalize a repository/run-relative artifact path."""

    if not isinstance(path, str):
        raise TypeError("Artifact path must be a string")
    if not path or path != path.strip():
        raise ValueError(f"Artifact path must be non-empty and unpadded: {path!r}")
    normalized = path.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        not candidate.parts
        or candidate.as_posix() == "."
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != normalized
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"Artifact path must be non-empty and relative: {path!r}")
    if any(":" in part for part in candidate.parts):
        raise ValueError(f"Artifact path must not contain a drive prefix or colon: {path!r}")
    for part in candidate.parts:
        if part.endswith((".", " ")):
            raise ValueError("Artifact path components must not end with a dot or space")
        if _WINDOWS_RESERVED_COMPONENT_RE.fullmatch(part):
            raise ValueError("Artifact path contains a reserved Windows device component")
    return candidate.as_posix()


def _windows_artifact_alias(path: str) -> str:
    """Return a conservative Windows-equivalence key for a validated path."""

    return "/".join(
        unicodedata.normalize("NFKC", part).casefold() for part in PurePosixPath(path).parts
    )


def sha256_ordered_ids(example_ids: Sequence[str]) -> str:
    """Hash an ordered, duplicate-free list of non-empty example IDs."""

    if isinstance(example_ids, (str, bytes)) or not isinstance(example_ids, Sequence):
        raise TypeError("example_ids must be an ordered sequence")
    normalized = list(example_ids)
    if any(not isinstance(example_id, str) for example_id in normalized):
        raise TypeError("example IDs must be strings")
    if any(not example_id or example_id != example_id.strip() for example_id in normalized):
        raise ValueError("example IDs must be non-empty and unpadded")
    if len(set(normalized)) != len(normalized):
        raise ValueError("example IDs must be unique")
    return sha256_json(normalized)


def _validate_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-'")
    if run_id.endswith((".", " ")):
        raise ValueError("run_id must not end with a dot or space")
    if _WINDOWS_RESERVED_COMPONENT_RE.fullmatch(run_id):
        raise ValueError("run_id must not be a reserved Windows device component")
    return run_id


def create_run_directory(root: Union[Path, str], run_id: str) -> Path:
    """Create a run directory exactly once and fail on collisions."""

    run_id = _validate_run_id(run_id)
    if not isinstance(root, (str, Path)):
        raise TypeError("root must be a path")
    run_root = Path(root)
    run_root.mkdir(parents=True, exist_ok=True)
    destination = run_root / run_id
    destination.mkdir(exist_ok=False)
    return destination


def _key_words(key: str) -> Sequence[str]:
    with_acronyms_split = _ACRONYM_BOUNDARY_RE.sub(r"\1_\2", key)
    with_camel_split = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", with_acronyms_split)
    return tuple(word.lower() for word in _NON_ALNUM_RE.split(with_camel_split) if word)


def _is_sensitive_key(key: str) -> bool:
    words = _key_words(key)
    if tuple(words) in _SAFE_NON_SECRET_KEY_WORDS:
        return False
    if "".join(words) in _SENSITIVE_COMPACT_KEYS:
        return True
    if any(word in _SENSITIVE_SINGLE_KEY_WORDS for word in words):
        return True
    return any(
        tuple(words[index : index + len(sequence)]) == sequence
        for sequence in _SENSITIVE_KEY_SEQUENCES
        for index in range(len(words) - len(sequence) + 1)
    )


def _url_has_credentials(value: str) -> bool:
    for match in _URL_RE.finditer(value):
        url = match.group(0).rstrip(".,);")
        try:
            parsed = urlsplit(url)
            if parsed.username is not None or parsed.password is not None:
                return True
            parameters = parse_qsl(parsed.query.replace(";", "&"), keep_blank_values=True)
            parameters.extend(parse_qsl(parsed.fragment.replace(";", "&"), keep_blank_values=True))
        except ValueError:
            return True
        for raw_key, _ in parameters:
            compact = "".join(_key_words(raw_key))
            if _is_sensitive_key(raw_key) or compact in _SENSITIVE_URL_KEYS:
                return True
    return False


def _assert_no_secrets(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"Manifest object keys must be strings: {path}")
            key = raw_key
            if _url_has_credentials(key) or _KNOWN_SECRET_VALUE_RE.search(key):
                raise ValueError(f"Secret-bearing mapping key is forbidden in manifests: {path}")
            if _is_sensitive_key(key):
                raise ValueError(f"Sensitive field is forbidden in manifests: {path}.{key}")
            _assert_no_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if _url_has_credentials(value):
            raise ValueError(f"Credential-bearing URL is forbidden in manifests: {path}")
        if _KNOWN_SECRET_VALUE_RE.search(value):
            raise ValueError(f"Token-like secret value is forbidden in manifests: {path}")


def seal_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of *payload* carrying a self-verifiable SHA-256 seal."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if MANIFEST_HASH_FIELD in payload:
        raise ValueError(f"Unsealed payload must not contain {MANIFEST_HASH_FIELD!r}")
    plain = dict(payload)
    _assert_no_secrets(plain)
    sealed = dict(plain)
    sealed[MANIFEST_HASH_FIELD] = sha256_json(plain)
    return sealed


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    """Return ``True`` when the manifest seal matches its canonical payload."""

    if not isinstance(manifest, Mapping):
        return False
    supplied = manifest.get(MANIFEST_HASH_FIELD)
    if not isinstance(supplied, str) or not _SHA256_RE.fullmatch(supplied):
        return False
    plain = dict(manifest)
    plain.pop(MANIFEST_HASH_FIELD, None)
    try:
        _assert_no_secrets(plain)
        expected = sha256_json(plain)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(supplied, expected)


def validate_publication_run_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_run_id: Optional[str] = None,
    artifact_root: Optional[Union[Path, str]] = None,
    selected_example_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate the sealed invariants required at a publication boundary.

    A valid hash only proves that a payload is self-consistent.  This validator
    additionally rejects resealed smoke, pilot, tuning, ablation, dirty, mock,
    non-final-split, or evaluator-gate-failing payloads.  It deliberately does
    not turn a manifest into evidence that the underlying experiment is
    scientifically sufficient; callers must still compare exact run identities
    and bound result artifacts. When ``artifact_root`` is supplied, every
    referenced input artifact is read from that contained directory, hashed,
    parsed, and cross-checked. ``selected_example_ids`` additionally binds the
    exact ordered result population to the audited sealed-final split.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("run manifest must be a mapping")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA or not verify_manifest(manifest):
        raise ValueError("run manifest is invalid or unsealed")
    if set(manifest) != _RUN_MANIFEST_FIELDS:
        raise ValueError("run manifest fields do not match the exact schema")

    run_id = _validate_run_id(manifest.get("run_id"))
    if expected_run_id is not None:
        expected = _validate_run_id(expected_run_id)
        if run_id != expected:
            raise ValueError("run ID does not match the publication artifact directory")
    if manifest.get("run_kind") != "final":
        raise ValueError("publication artifacts require run_kind='final'")
    if manifest.get("mock") is not False:
        raise ValueError("publication artifacts require mock=false")

    git = manifest.get("git")
    if not isinstance(git, Mapping) or set(git) != {"commit", "dirty"}:
        raise ValueError("publication run manifest has invalid Git identity")
    _require_immutable_revision(git.get("commit"), "git.commit")
    if git.get("dirty") is not False:
        raise ValueError("publication artifacts require git.dirty=false")

    config = manifest.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ValueError("publication run manifest config must be a non-empty mapping")
    if manifest.get("config_sha256") != sha256_json(config):
        raise ValueError("publication run manifest config hash does not match")
    _require_manifest_digest(manifest.get("config_file_sha256"), "config_file_sha256")

    data = manifest.get("data")
    if not isinstance(data, Mapping) or set(data) != _RUN_DATA_FIELDS:
        raise ValueError("publication run manifest data binding does not match its exact schema")
    if data.get("split_partition") != "sealed_final":
        raise ValueError("publication artifacts require data.split_partition='sealed_final'")
    for field in (
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "selected_example_ids_sha256",
    ):
        _require_manifest_digest(data.get(field), f"data.{field}")
    _require_positive_manifest_count(
        data.get("selected_example_count"), "data.selected_example_count"
    )

    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, Mapping) or set(retrieval) != _RUN_RETRIEVAL_FIELDS:
        raise ValueError(
            "publication run manifest retrieval binding does not match its exact schema"
        )
    for field in ("corpus_manifest_sha256", "index_manifest_sha256"):
        _require_manifest_digest(retrieval.get(field), f"retrieval.{field}")
    _require_positive_manifest_count(
        retrieval.get("exact_indexed_passage_count"),
        "retrieval.exact_indexed_passage_count",
    )

    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, Mapping) or set(evaluator) != _RUN_EVALUATOR_FIELDS:
        raise ValueError(
            "publication run manifest evaluator binding does not match its exact schema"
        )
    if evaluator.get("production_gate_passed") is not True:
        raise ValueError("publication artifacts require evaluator.production_gate_passed=true")
    for field in (
        "fixture_sha256",
        "fixture_content_sha256",
        "fixture_report_sha256",
    ):
        _require_manifest_digest(evaluator.get(field), f"evaluator.{field}")
    report = evaluator.get("fixture_report")
    if not isinstance(report, Mapping) or set(report) != _EVALUATOR_REPORT_KEYS:
        raise ValueError("publication evaluator report does not match its exact schema")
    if report.get("passed") is not True or report.get("production_gate_passed") is not True:
        raise ValueError("publication evaluator report must pass its production gate")
    if evaluator.get("fixture_report_sha256") != sha256_json(report):
        raise ValueError("publication evaluator report hash does not match")
    if report.get("schema_version") != _EVALUATOR_SCHEMA_VERSION:
        raise ValueError("publication evaluator report has an unsupported schema version")
    if report.get("fixture_sha256") != evaluator.get("fixture_sha256") or report.get(
        "fixture_content_sha256"
    ) != evaluator.get("fixture_content_sha256"):
        raise ValueError("publication evaluator report does not bind the registered fixture")
    if report.get("scorer_id") != evaluator.get("scorer_id") or report.get(
        "scorer_revision"
    ) != evaluator.get("scorer_revision"):
        raise ValueError("publication evaluator identity is internally inconsistent")

    models = _require_string_mapping(manifest.get("model_revisions"), "model_revisions")
    for model_id, revision in models.items():
        if not _IDENTITY_RE.fullmatch(model_id):
            raise ValueError("model_revisions keys must be stable component identifiers")
        _require_immutable_revision(revision, f"model_revisions[{model_id!r}]")
    scorer_id = evaluator.get("scorer_id")
    scorer_revision = evaluator.get("scorer_revision")
    if not isinstance(scorer_id, str) or models.get(scorer_id) != scorer_revision:
        raise ValueError("publication evaluator revision is not bound in model_revisions")
    if isinstance(manifest.get("seed"), bool) or not isinstance(manifest.get("seed"), int):
        raise ValueError("publication run manifest seed must be an integer")
    created_at_utc = manifest.get("created_at_utc")
    if not isinstance(created_at_utc, str):
        raise ValueError("publication run manifest created_at_utc must be a string")
    _require_utc_timestamp(created_at_utc)

    selected_ids: Optional[list[str]] = None
    if selected_example_ids is not None:
        if isinstance(selected_example_ids, (str, bytes)) or not isinstance(
            selected_example_ids, Sequence
        ):
            raise TypeError("selected_example_ids must be an ordered sequence")
        selected_ids = list(selected_example_ids)
        if not selected_ids:
            raise ValueError("selected_example_ids must be non-empty")
        if data.get("selected_example_ids_sha256") != sha256_ordered_ids(selected_ids):
            raise ValueError("publication results do not match selected_example_ids_sha256")
        if data.get("selected_example_count") != len(selected_ids):
            raise ValueError("publication results do not match selected_example_count")

    if artifact_root is not None:
        if not isinstance(artifact_root, (str, Path)):
            raise TypeError("artifact_root must be a path")
        root = Path(artifact_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("artifact_root must identify a directory")

        path_bindings = {
            "config": (
                manifest.get("config_path"),
                manifest.get("config_file_sha256"),
            ),
            "dataset manifest": (
                data.get("dataset_manifest_path"),
                data.get("dataset_manifest_sha256"),
            ),
            "split manifest": (
                data.get("split_manifest_path"),
                data.get("split_manifest_sha256"),
            ),
            "corpus manifest": (
                retrieval.get("corpus_manifest_path"),
                retrieval.get("corpus_manifest_sha256"),
            ),
            "index manifest": (
                retrieval.get("index_manifest_path"),
                retrieval.get("index_manifest_sha256"),
            ),
            "evaluator fixture": (
                evaluator.get("fixture_path"),
                evaluator.get("fixture_sha256"),
            ),
        }
        snapshots: Dict[str, bytes] = {}
        for name, (relative_path, expected_sha256) in path_bindings.items():
            if not isinstance(relative_path, str):
                raise ValueError(f"publication {name} path must be a string")
            snapshot = _read_bound_artifact(root, relative_path)
            if sha256_bytes(snapshot) != expected_sha256:
                raise ValueError(f"publication {name} bytes do not match the sealed digest")
            snapshots[name] = snapshot

        config_file = _load_strict_yaml_bytes(snapshots["config"])
        if not isinstance(config_file, Mapping) or canonical_json_bytes(
            config_file
        ) != canonical_json_bytes(config):
            raise ValueError("publication config bytes do not match the sealed effective config")

        dataset_manifest = _load_strict_json_bytes(
            snapshots["dataset manifest"], "dataset manifest"
        )
        _validate_sealed_artifact_manifest(
            dataset_manifest,
            expected_schema=DATASET_MANIFEST_SCHEMA,
            expected_fields=_DATASET_MANIFEST_FIELDS,
            name="dataset manifest",
        )
        split_manifest = _load_strict_json_bytes(snapshots["split manifest"], "split manifest")
        _validate_split_selection(
            split_manifest,
            split_partition="sealed_final",
            selected_ids=selected_ids or [],
        )
        _validate_dataset_split_binding(dataset_manifest, split_manifest)

        corpus_manifest = _load_strict_json_bytes(snapshots["corpus manifest"], "corpus manifest")
        _validate_sealed_artifact_manifest(
            corpus_manifest,
            expected_schema=CORPUS_MANIFEST_SCHEMA,
            expected_fields=_CORPUS_MANIFEST_FIELDS,
            name="corpus manifest",
        )
        index_manifest = _load_strict_json_bytes(snapshots["index manifest"], "index manifest")
        _validate_sealed_artifact_manifest(
            index_manifest,
            expected_schema=INDEX_MANIFEST_SCHEMA,
            expected_fields=_INDEX_MANIFEST_FIELDS,
            name="index manifest",
        )
        _validate_indexed_passage_count(
            index_manifest, cast(int, retrieval.get("exact_indexed_passage_count"))
        )
        _validate_corpus_index_binding(
            corpus_manifest,
            index_manifest,
            corpus_manifest_sha256=cast(str, retrieval.get("corpus_manifest_sha256")),
        )

        from factuality_rag.eval.sanity import (
            EVALUATOR_SANITY_V1_CONTENT_SHA256,
            load_sanity_fixture_bytes,
        )

        fixture = load_sanity_fixture_bytes(snapshots["evaluator fixture"])
        fixture_content_sha256 = sha256_json(fixture)
        if fixture_content_sha256 != EVALUATOR_SANITY_V1_CONTENT_SHA256:
            raise ValueError("publication run requires the registered evaluator fixture")
        if evaluator.get("fixture_content_sha256") != fixture_content_sha256:
            raise ValueError("publication evaluator fixture content digest does not match")
        _validate_evaluator_report(
            report,
            fixture=fixture,
            fixture_sha256=cast(str, evaluator.get("fixture_sha256")),
            fixture_content_sha256=fixture_content_sha256,
            model_revisions=models,
        )

    # Canonical JSON provides a detached strict-JSON copy for downstream code.
    return cast(Dict[str, Any], json.loads(canonical_json_bytes(manifest).decode("utf-8")))


def write_immutable_json(path: Union[Path, str], payload: Mapping[str, Any]) -> str:
    """Write canonical JSON once.

    Repeating the exact write is idempotent. Any attempt to replace existing
    content with different bytes raises ``FileExistsError``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload) + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if destination.read_bytes() == encoded:
            return "unchanged"
        raise FileExistsError(f"Refusing to overwrite immutable JSON: {destination}")
    return "created"


def _require_utc_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("created_at_utc must be a string")
    if not value or value != value.strip():
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp")
    return value


def _read_bound_artifact(root: Path, relative_path: str) -> bytes:
    """Resolve one contained artifact and capture its bytes exactly once."""

    normalized = validate_relative_artifact_path(relative_path)
    candidate = (root / normalized).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes artifact_root: {relative_path!r}") from exc
    if not candidate.is_file():
        raise ValueError(f"Artifact path must identify a file: {relative_path!r}")
    with candidate.open("rb") as handle:
        return handle.read()


def _reject_duplicate_json_keys(pairs: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_pair in pairs:
        key, value = raw_pair
        if not isinstance(key, str):
            raise ValueError("JSON object keys must be strings")
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_strict_json_bytes(data: bytes, name: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from exc


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and merge keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> Dict[Any, Any]:
    if not isinstance(node, yaml.nodes.MappingNode):
        raise yaml.constructor.ConstructorError(
            None, None, "expected a mapping node", node.start_mark
        )
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are forbidden",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _load_strict_yaml_bytes(data: bytes) -> Any:
    try:
        return yaml.load(data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("config must be strict UTF-8 YAML") from exc


def _validate_sealed_artifact_manifest(
    value: Any,
    *,
    expected_schema: str,
    expected_fields: AbstractSet[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object")
    if value.get("schema") != expected_schema:
        raise ValueError(f"{name} must use stable schema {expected_schema!r}")
    if not verify_manifest(value):
        raise ValueError(f"{name} seal is invalid")
    _require_manifest_digest(value.get(MANIFEST_HASH_FIELD), f"{name} seal")
    if set(value) != set(expected_fields):
        raise ValueError(f"{name} fields do not match its exact schema")
    return value


def _require_manifest_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _require_manifest_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if set(value) == {"0"}:
        raise ValueError(f"{name} must not be an all-zero SHA-256 placeholder")
    return value


def _require_positive_manifest_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _validate_dataset_split_binding(
    dataset_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    _require_manifest_id(dataset_manifest["dataset_id"], "dataset manifest dataset_id")
    dataset_count = _require_positive_manifest_count(
        dataset_manifest["example_count"], "dataset manifest example_count"
    )
    dataset_snapshot = _require_manifest_digest(
        dataset_manifest["source_snapshot_sha256"],
        "dataset manifest source_snapshot_sha256",
    )
    source = split_manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("split manifest source is missing")
    if (
        source.get("example_count") != dataset_count
        or source.get("source_snapshot_sha256") != dataset_snapshot
    ):
        raise ValueError("dataset manifest does not match the audited split source snapshot")


def _validate_corpus_index_binding(
    corpus_manifest: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    *,
    corpus_manifest_sha256: str,
) -> None:
    _require_manifest_id(corpus_manifest["corpus_id"], "corpus manifest corpus_id")
    corpus_snapshot = _require_manifest_digest(
        corpus_manifest["corpus_snapshot_sha256"],
        "corpus manifest corpus_snapshot_sha256",
    )
    corpus_count = _require_positive_manifest_count(
        corpus_manifest["passage_count"], "corpus manifest passage_count"
    )
    _require_manifest_id(index_manifest["index_id"], "index manifest index_id")
    bound_corpus_manifest = _require_manifest_digest(
        index_manifest["corpus_manifest_sha256"],
        "index manifest corpus_manifest_sha256",
    )
    bound_corpus_snapshot = _require_manifest_digest(
        index_manifest["corpus_snapshot_sha256"],
        "index manifest corpus_snapshot_sha256",
    )
    indexed_count = _require_positive_manifest_count(
        index_manifest["exact_indexed_passage_count"],
        "index manifest exact_indexed_passage_count",
    )
    if (
        bound_corpus_manifest != corpus_manifest_sha256
        or bound_corpus_snapshot != corpus_snapshot
        or indexed_count != corpus_count
    ):
        raise ValueError("index manifest does not match the bound corpus manifest")


def _require_immutable_revision(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not _IMMUTABLE_REVISION_RE.fullmatch(value):
        raise ValueError(f"{name} must be exactly 40 or 64 lowercase hexadecimal characters")
    if set(value) == {"0"}:
        raise ValueError(f"{name} must not be an all-zero placeholder")
    return value


def _validate_split_selection(
    split_manifest: Any,
    *,
    split_partition: str,
    selected_ids: Sequence[str],
) -> None:
    if not isinstance(split_manifest, Mapping):
        raise ValueError("split manifest must be a JSON object")
    if not verify_manifest(split_manifest):
        raise ValueError("split manifest seal is invalid")

    # Runtime import avoids a module cycle: data.splits itself uses this
    # module's sealing utilities. Its audit recomputes the stored leakage
    # checks rather than trusting a caller-supplied `audit.passed` boolean.
    from factuality_rag.data.splits import audit_split_manifest

    try:
        audit = audit_split_manifest(split_manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("split manifest audit failed") from exc
    if not isinstance(audit, Mapping) or audit.get("passed") is not True:
        raise ValueError("split manifest audit failed")
    _require_manifest_digest(split_manifest.get(MANIFEST_HASH_FIELD), "split manifest seal")
    source = split_manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("split manifest source is missing")
    for field in ("source_snapshot_sha256", "normalized_examples_sha256"):
        _require_manifest_digest(source.get(field), f"split manifest source.{field}")
    components = split_manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("split manifest components are missing")
    for component_id in components:
        _require_manifest_digest(component_id, "split manifest component ID")

    try:
        partition_ids = split_manifest["partitions"][split_partition]["example_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError("split manifest does not contain the requested partition") from exc
    if not isinstance(partition_ids, list) or any(
        not isinstance(example_id, str) for example_id in partition_ids
    ):
        raise ValueError("split partition example_ids must be a string list")
    unknown = sorted(set(selected_ids) - set(partition_ids))
    if unknown:
        raise ValueError(
            "selected_example_ids include IDs outside split_partition: " + ", ".join(unknown[:3])
        )


def _validate_indexed_passage_count(index_manifest: Any, expected: int) -> None:
    if not isinstance(index_manifest, Mapping):
        raise ValueError("index manifest must be a JSON object")
    actual = index_manifest.get("exact_indexed_passage_count")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual <= 0:
        raise ValueError("index manifest exact_indexed_passage_count must be a positive integer")
    if actual != expected:
        raise ValueError("exact_indexed_passage_count does not match the actual index manifest")


def _require_nonempty_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(key, str) or not key or key != key.strip() for key in value):
        raise ValueError(f"{name} keys must be non-empty, unpadded strings")
    return value


def _require_string_mapping(value: Any, name: str) -> Mapping[str, str]:
    mapping = _require_nonempty_mapping(value, name)
    if any(
        not isinstance(item, str) or not item or item != item.strip() for item in mapping.values()
    ):
        raise ValueError(f"{name} values must be non-empty, unpadded strings")
    return cast(Mapping[str, str], mapping)


def _validate_evaluator_report(
    report: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    fixture_content_sha256: str,
    model_revisions: Mapping[str, str],
) -> Dict[str, Any]:
    if set(report) != _EVALUATOR_REPORT_KEYS:
        raise ValueError(
            "evaluator_fixture_report must contain exactly the versioned report schema"
        )
    if report["passed"] is not True or report["production_gate_passed"] is not True:
        raise ValueError("the production evaluator sanity gate must pass")
    atol = report["atol"]
    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        raise TypeError("evaluator atol must be numeric")
    if not math.isfinite(float(atol)) or atol < 0 or atol > 1e-12:
        raise ValueError("production evaluator atol must be finite and between 0 and 1e-12")
    if report["schema_version"] != _EVALUATOR_SCHEMA_VERSION or report[
        "schema_version"
    ] != fixture.get("schema_version"):
        raise ValueError(
            f"evaluator_fixture_report schema_version must be {_EVALUATOR_SCHEMA_VERSION!r}"
        )
    if report["fixture_sha256"] != fixture_sha256:
        raise ValueError("evaluator fixture report does not match the actual fixture bytes")
    content_sha256 = report["fixture_content_sha256"]
    if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(content_sha256):
        raise ValueError("fixture_content_sha256 must be a lowercase SHA-256 digest")
    if content_sha256 != fixture_content_sha256:
        raise ValueError(
            "evaluator fixture report does not match the actual canonical fixture content"
        )

    scorer_id = report["scorer_id"]
    scorer_revision = report["scorer_revision"]
    if not isinstance(scorer_id, str) or not _IDENTITY_RE.fullmatch(scorer_id):
        raise ValueError("scorer_id must be a non-empty, unpadded stable identifier")
    scorer_revision = _require_immutable_revision(scorer_revision, "scorer_revision")
    if model_revisions.get(scorer_id) != scorer_revision:
        raise ValueError("scorer_id and scorer_revision must be bound in model_revisions")

    case_count = report["case_count"]
    results = report["results"]
    if isinstance(case_count, bool) or not isinstance(case_count, int):
        raise TypeError("evaluator case_count must be an integer")
    fixture_cases = fixture.get("cases")
    if not isinstance(fixture_cases, list) or not fixture_cases:
        raise ValueError("registered evaluator fixture must contain cases")
    if case_count != len(fixture_cases):
        raise ValueError("evaluator case_count must match the registered fixture")
    if not isinstance(results, list):
        raise TypeError("evaluator results must be a list")
    if len(results) != case_count:
        raise ValueError("evaluator case_count must match results")

    result_ids = []
    for index, (case, result) in enumerate(zip(fixture_cases, results)):
        if not isinstance(case, Mapping):
            raise ValueError("registered evaluator fixture case must be an object")
        case_id = case.get("id")
        expected = case.get("expected")
        if not isinstance(case_id, str) or not isinstance(expected, Mapping):
            raise ValueError("registered evaluator fixture case is malformed")
        if not isinstance(result, Mapping):
            raise TypeError("each evaluator result must be a mapping")
        result_id = result.get("id")
        if not isinstance(result_id, str) or not result_id or result_id != result_id.strip():
            raise ValueError("each evaluator result requires a non-empty, unpadded id")
        if result_id != case_id:
            raise ValueError(
                f"evaluator result {index} must match registered fixture case {case_id!r}"
            )
        expected_keys = {"id"} | set(expected)
        if set(result) != expected_keys:
            raise ValueError("each evaluator result must match the exact registered result schema")
        actual_metrics = {key: result[key] for key in expected}
        if canonical_json_bytes(actual_metrics) != canonical_json_bytes(expected):
            raise ValueError(
                f"evaluator result for case {case_id!r} does not match the registered oracle"
            )
        result_ids.append(result_id)
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("evaluator result ids must be unique")

    # This round-trip both deep-copies the report and proves it is strict JSON.
    return cast(Dict[str, Any], json.loads(canonical_json_bytes(report).decode("utf-8")))


def build_run_manifest(
    *,
    run_id: str,
    run_kind: str,
    git_commit: str,
    git_dirty: bool,
    artifact_root: Union[Path, str],
    config: Mapping[str, Any],
    config_path: str,
    dataset_manifest_path: str,
    split_manifest_path: str,
    split_partition: str,
    selected_example_ids: Sequence[str],
    corpus_manifest_path: str,
    exact_indexed_passage_count: int,
    index_manifest_path: str,
    evaluator_fixture_path: str,
    evaluator_fixture_report: Mapping[str, Any],
    model_revisions: Mapping[str, str],
    seed: int,
    hardware: Mapping[str, Any],
    software: Mapping[str, str],
    resource_ceilings: Mapping[str, Any],
    output_paths: Sequence[str],
    mock: bool,
    created_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and seal the immutable, pre-run experiment manifest."""

    run_id = _validate_run_id(run_id)
    if not isinstance(run_kind, str):
        raise TypeError("run_kind must be a string")
    allowed_kinds = {"smoke", "pilot", "tuning", "ablation", "final"}
    if run_kind not in allowed_kinds:
        raise ValueError(f"run_kind must be one of {sorted(allowed_kinds)}")
    git_commit = _require_immutable_revision(git_commit, "git_commit")
    if type(git_dirty) is not bool:
        raise TypeError("git_dirty must be boolean")
    if type(mock) is not bool:
        raise TypeError("mock must be boolean")
    if run_kind == "final" and git_dirty:
        raise ValueError("final runs require a clean Git revision")
    if run_kind == "final" and mock:
        raise ValueError("final runs cannot use mock components")
    if not isinstance(split_partition, str):
        raise TypeError("split_partition must be a string")
    if split_partition not in {"train", "tuning", "sealed_final"}:
        raise ValueError("split_partition must be train, tuning or sealed_final")
    if split_partition == "sealed_final" and run_kind != "final":
        raise ValueError("only final runs can access sealed_final")
    if run_kind == "final" and split_partition != "sealed_final":
        raise ValueError("final runs require the sealed_final partition")
    if isinstance(exact_indexed_passage_count, bool) or not isinstance(
        exact_indexed_passage_count, int
    ):
        raise TypeError("exact_indexed_passage_count must be an integer")
    if exact_indexed_passage_count <= 0:
        raise ValueError("exact_indexed_passage_count must be positive")
    config_mapping = _require_nonempty_mapping(config, "config")
    hardware_mapping = _require_nonempty_mapping(hardware, "hardware")
    software_mapping = _require_string_mapping(software, "software")
    resource_mapping = _require_nonempty_mapping(resource_ceilings, "resource_ceilings")
    model_mapping = _require_string_mapping(model_revisions, "model_revisions")
    for model_id, revision in model_mapping.items():
        if not _IDENTITY_RE.fullmatch(model_id):
            raise ValueError("model_revisions keys must be stable component identifiers")
        _require_immutable_revision(revision, f"model_revisions[{model_id!r}]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(selected_example_ids, (str, bytes)) or not isinstance(
        selected_example_ids, Sequence
    ):
        raise TypeError("selected_example_ids must be an ordered sequence")
    selected_ids = list(selected_example_ids)
    if not selected_ids:
        raise ValueError("selected_example_ids must be non-empty")
    selected_ids_sha256 = sha256_ordered_ids(selected_ids)

    paths = {
        "config_path": validate_relative_artifact_path(config_path),
        "dataset_manifest_path": validate_relative_artifact_path(dataset_manifest_path),
        "split_manifest_path": validate_relative_artifact_path(split_manifest_path),
        "corpus_manifest_path": validate_relative_artifact_path(corpus_manifest_path),
        "index_manifest_path": validate_relative_artifact_path(index_manifest_path),
        "evaluator_fixture_path": validate_relative_artifact_path(evaluator_fixture_path),
    }
    input_aliases = [_windows_artifact_alias(path) for path in paths.values()]
    if len(set(input_aliases)) != len(input_aliases):
        raise ValueError("input artifact paths must be unique case-insensitively")
    if not isinstance(artifact_root, (str, Path)):
        raise TypeError("artifact_root must be a path")
    if isinstance(artifact_root, str) and (
        not artifact_root or artifact_root != artifact_root.strip()
    ):
        raise ValueError("artifact_root must be non-empty and unpadded")
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact_root must identify a directory")
    artifact_bytes = {name: _read_bound_artifact(root, path) for name, path in paths.items()}
    hashes = {
        name.replace("_path", "_sha256"): sha256_bytes(artifact_bytes[name]) for name in paths
    }

    config_file = _load_strict_yaml_bytes(artifact_bytes["config_path"])
    config_file_mapping = _require_nonempty_mapping(config_file, "config file")
    config_copy = json.loads(canonical_json_bytes(config_mapping).decode("utf-8"))
    config_file_copy = json.loads(canonical_json_bytes(config_file_mapping).decode("utf-8"))
    if canonical_json_bytes(config_copy) != canonical_json_bytes(config_file_copy):
        raise ValueError("config must exactly match the parsed config_path YAML")
    config_seed = config_copy.get("seed")
    if isinstance(config_seed, bool) or not isinstance(config_seed, int):
        raise ValueError("config seed must be an integer")
    if config_seed != seed:
        raise ValueError("config seed must exactly match the top-level seed")

    dataset_manifest = _load_strict_json_bytes(
        artifact_bytes["dataset_manifest_path"], "dataset manifest"
    )
    _validate_sealed_artifact_manifest(
        dataset_manifest,
        expected_schema=DATASET_MANIFEST_SCHEMA,
        expected_fields=_DATASET_MANIFEST_FIELDS,
        name="dataset manifest",
    )
    corpus_manifest = _load_strict_json_bytes(
        artifact_bytes["corpus_manifest_path"], "corpus manifest"
    )
    _validate_sealed_artifact_manifest(
        corpus_manifest,
        expected_schema=CORPUS_MANIFEST_SCHEMA,
        expected_fields=_CORPUS_MANIFEST_FIELDS,
        name="corpus manifest",
    )
    index_manifest = _load_strict_json_bytes(
        artifact_bytes["index_manifest_path"], "index manifest"
    )
    _validate_sealed_artifact_manifest(
        index_manifest,
        expected_schema=INDEX_MANIFEST_SCHEMA,
        expected_fields=_INDEX_MANIFEST_FIELDS,
        name="index manifest",
    )
    _validate_indexed_passage_count(index_manifest, exact_indexed_passage_count)

    if not isinstance(evaluator_fixture_report, Mapping):
        raise TypeError("evaluator_fixture_report must be a mapping")
    split_manifest = _load_strict_json_bytes(
        artifact_bytes["split_manifest_path"], "split manifest"
    )
    _validate_split_selection(
        split_manifest,
        split_partition=split_partition,
        selected_ids=selected_ids,
    )
    _validate_dataset_split_binding(dataset_manifest, split_manifest)
    _validate_corpus_index_binding(
        corpus_manifest,
        index_manifest,
        corpus_manifest_sha256=hashes["corpus_manifest_sha256"],
    )

    from factuality_rag.eval.sanity import (
        EVALUATOR_SANITY_V1_CONTENT_SHA256,
        load_sanity_fixture_bytes,
    )

    registered_fixture = load_sanity_fixture_bytes(artifact_bytes["evaluator_fixture_path"])
    fixture_content_sha256 = sha256_json(registered_fixture)
    if not hmac.compare_digest(fixture_content_sha256, EVALUATOR_SANITY_V1_CONTENT_SHA256):
        raise ValueError("production runs require the registered canonical evaluator fixture")
    evaluator_report = _validate_evaluator_report(
        evaluator_fixture_report,
        fixture=registered_fixture,
        fixture_sha256=hashes["evaluator_fixture_sha256"],
        fixture_content_sha256=fixture_content_sha256,
        model_revisions=model_mapping,
    )
    if isinstance(output_paths, (str, bytes)) or not isinstance(output_paths, Sequence):
        raise TypeError("output_paths must be an ordered sequence")
    normalized_outputs = [validate_relative_artifact_path(path) for path in output_paths]
    if not normalized_outputs:
        raise ValueError("output_paths must be non-empty")
    output_aliases = [_windows_artifact_alias(path) for path in normalized_outputs]
    if len(set(output_aliases)) != len(output_aliases):
        raise ValueError("output_paths must not contain case-insensitive duplicates")
    if set(output_aliases) & set(input_aliases):
        raise ValueError("output_paths must not overwrite input artifacts")
    timestamp_value = (
        datetime.now(timezone.utc).isoformat() if created_at_utc is None else created_at_utc
    )
    timestamp = _require_utc_timestamp(timestamp_value)

    # Canonical JSON round-trips detach the result from mutable caller-owned
    # objects while validating JSON types and finite numeric values up front.
    hardware_copy = json.loads(canonical_json_bytes(hardware_mapping).decode("utf-8"))
    resource_copy = json.loads(canonical_json_bytes(resource_mapping).decode("utf-8"))

    payload: Dict[str, Any] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_kind": run_kind,
        "created_at_utc": timestamp,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "config": config_copy,
        "config_sha256": sha256_json(config_copy),
        "config_path": paths["config_path"],
        "config_file_sha256": hashes["config_sha256"],
        "data": {
            "dataset_manifest_path": paths["dataset_manifest_path"],
            "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
            "split_manifest_path": paths["split_manifest_path"],
            "split_manifest_sha256": hashes["split_manifest_sha256"],
            "split_partition": split_partition,
            "selected_example_ids_sha256": selected_ids_sha256,
            "selected_example_count": len(selected_ids),
        },
        "retrieval": {
            "corpus_manifest_path": paths["corpus_manifest_path"],
            "corpus_manifest_sha256": hashes["corpus_manifest_sha256"],
            "index_manifest_path": paths["index_manifest_path"],
            "index_manifest_sha256": hashes["index_manifest_sha256"],
            "exact_indexed_passage_count": exact_indexed_passage_count,
        },
        "evaluator": {
            "fixture_path": paths["evaluator_fixture_path"],
            "fixture_sha256": hashes["evaluator_fixture_sha256"],
            "fixture_content_sha256": fixture_content_sha256,
            "fixture_report": evaluator_report,
            "fixture_report_sha256": sha256_json(evaluator_report),
            "scorer_id": evaluator_report["scorer_id"],
            "scorer_revision": evaluator_report["scorer_revision"],
            "production_gate_passed": True,
        },
        "model_revisions": dict(sorted(model_mapping.items())),
        "seed": seed,
        "hardware": hardware_copy,
        "software": dict(sorted(software_mapping.items())),
        "resource_ceilings": resource_copy,
        "output_paths": normalized_outputs,
        "mock": mock,
    }
    return seal_manifest(payload)
