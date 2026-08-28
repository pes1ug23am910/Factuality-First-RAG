"""
factuality_rag.experiment_runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structured experiment execution with metadata tracking and
prediction persistence.

Every run is saved to ``runs/<run-id>/`` with:
    - ``predictions.jsonl`` – per-query results
    - ``resume_manifest.json`` – immutable checkpoint identity
    - ``references_by_example_id.json`` – lossless reference records
    - ``references.json`` – legacy unambiguous query lookup
    - ``metrics.json`` – aggregated evaluation metrics
    - ``metadata.json`` – full run metadata

Example::

    >>> import yaml
    >>> cfg = yaml.safe_load(open("configs/exp_sample.yaml"))  # doctest: +SKIP
    >>> results = run(cfg, mock_mode=True)  # doctest: +SKIP
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import os
import re
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from urllib.parse import quote

import numpy as np

Reference = Union[str, Sequence[str]]

logger = logging.getLogger(__name__)

_RUN_ID_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_SOURCE_ROOT = Path(__file__).resolve().parent
_RUNTIME_SOURCE_SCHEMA = "factuality-rag.runtime-source.v1"
_RESUME_MANIFEST_NAME = "resume_manifest.json"
_RESUME_MANIFEST_SCHEMA = "factuality-rag.resume-checkpoint.v2"


def _get_git_state() -> Tuple[str, Optional[bool]]:
    """Return Git state only when ``_SOURCE_ROOT`` is the checkout root.

    Git normally walks upward. Requiring a direct ``.git`` marker and an exact
    ``--show-toplevel`` match prevents an installed package from inheriting the
    revision of an unrelated repository containing the environment.

    Returns:
        ``(full_commit, dirty)`` or ``("git-not-available", None)``.
    """
    import subprocess

    if not (_SOURCE_ROOT / ".git").exists():
        return "git-not-available", None

    try:
        top_level = Path(
            subprocess.check_output(
                ["git", "-C", str(_SOURCE_ROOT), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        ).resolve(strict=True)
        if top_level != _SOURCE_ROOT.resolve(strict=True):
            return "git-not-available", None

        commit = (
            subprocess.check_output(
                ["git", "-C", str(_SOURCE_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
            return "git-not-available", None
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(_SOURCE_ROOT),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            stderr=subprocess.DEVNULL,
        )
        return commit, bool(status.strip())
    except Exception:
        return "git-not-available", None


def _get_git_commit() -> str:
    """Return the exact source checkout's full Git revision, or a fallback."""
    return _get_git_state()[0]


def _validate_runtime_package_origin() -> None:
    """Fail if imports resolve outside the package tree this runner will hash."""
    import factuality_rag

    package_file = getattr(factuality_rag, "__file__", None)
    if not isinstance(package_file, str) or not package_file:
        raise RuntimeError("imported factuality_rag package has no inspectable origin")
    try:
        imported_root = Path(package_file).resolve(strict=True).parent
        runner_root = Path(__file__).resolve(strict=True).parent
    except OSError as exc:
        raise RuntimeError("runtime package origin could not be resolved") from exc
    if imported_root != runner_root:
        raise RuntimeError(
            "imported factuality_rag package does not match the experiment runner source tree"
        )


def _runtime_source_snapshot_once() -> Dict[str, Any]:
    """Hash the exact importable package source and resource bytes once."""
    from factuality_rag.reproducibility import sha256_bytes, sha256_json

    source_root = _RUNTIME_SOURCE_ROOT
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError("runtime source root must be a regular package directory")
    resolved_root = source_root.resolve(strict=True)
    entries: List[Dict[str, str]] = []

    def fail_walk(error: OSError) -> None:
        raise RuntimeError("runtime source tree could not be enumerated") from error

    for directory, directory_names, file_names in os.walk(
        source_root,
        topdown=True,
        onerror=fail_walk,
        followlinks=False,
    ):
        current = Path(directory)
        retained_directories: List[str] = []
        for name in sorted(directory_names):
            relative = (current / name).relative_to(source_root).as_posix()
            if name == ".env" or name.startswith(".env."):
                raise RuntimeError("runtime source tree must not contain .env files")
            if name == "__pycache__":
                continue
            candidate = current / name
            if candidate.is_symlink():
                raise RuntimeError(f"runtime source directory must not be a symlink: {relative}")
            try:
                candidate.resolve(strict=True).relative_to(resolved_root)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"runtime source directory escapes the package root: {relative}"
                ) from exc
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(source_root).as_posix()
            if name == ".env" or name.startswith(".env."):
                raise RuntimeError("runtime source tree must not contain .env files")
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise RuntimeError(f"runtime source file must not be a symlink: {relative}")
            if not path.is_file():
                raise RuntimeError(f"runtime source entry must be a regular file: {relative}")
            try:
                path.resolve(strict=True).relative_to(resolved_root)
                before = path.stat()
                payload = path.read_bytes()
                after = path.stat()
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"runtime source file could not be hashed: {relative}") from exc
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"runtime source file changed while hashing: {relative}")
            entries.append({"path": relative, "sha256": sha256_bytes(payload)})

    paths = {entry["path"] for entry in entries}
    if "__init__.py" not in paths or "experiment_runner.py" not in paths:
        raise RuntimeError("runtime source tree is missing required package files")
    digest_payload = {"schema": _RUNTIME_SOURCE_SCHEMA, "files": entries}
    return {
        "schema": _RUNTIME_SOURCE_SCHEMA,
        "sha256": sha256_json(digest_payload),
        "file_count": len(entries),
    }


def _get_runtime_source_identity() -> Dict[str, Any]:
    """Return a stable package-source identity or fail on concurrent edits."""
    first = _runtime_source_snapshot_once()
    second = _runtime_source_snapshot_once()
    if first != second:
        raise RuntimeError("runtime source tree changed while its identity was being captured")
    return first


def _config_metadata_identity(
    config_path: str,
    *,
    effective_config_sha256: str,
) -> Tuple[str, Optional[str]]:
    """Return a path-free config identity and the exact source-byte digest."""
    from factuality_rag.reproducibility import sha256_bytes, sha256_file
    from factuality_rag.resources import read_experiment_config_bytes

    package_prefix = "package:factuality_rag.resources/configs/"
    if config_path.startswith(package_prefix):
        resource_name = config_path.removeprefix(package_prefix)
        source_sha256 = sha256_bytes(read_experiment_config_bytes(resource_name))
        identity = (
            "package://factuality_rag.resources/configs/"
            f"{quote(resource_name, safe='._-')}?sha256={source_sha256}"
        )
        return identity, source_sha256

    candidate: Optional[Path] = None
    if config_path:
        supplied = Path(config_path).expanduser()
        if supplied.is_absolute():
            candidate = supplied
        else:
            source_candidate = _SOURCE_ROOT / supplied
            candidate = source_candidate if source_candidate.is_file() else supplied
    if candidate is not None and candidate.is_file():
        resolved = candidate.resolve(strict=True)
        source_sha256 = sha256_file(resolved)
        identity = f"external-config://{quote(resolved.name, safe='._-')}?sha256={source_sha256}"
        return identity, source_sha256

    return (
        f"in-memory-config://effective?sha256={effective_config_sha256}",
        None,
    )


def _get_lib_versions() -> Dict[str, str]:
    """Collect library versions for reproducibility.

    Returns:
        Mapping of library name to version string.
    """
    versions: Dict[str, str] = {}

    try:
        import faiss  # type: ignore[import-untyped]

        versions["faiss"] = getattr(faiss, "__version__", "unknown")
    except ImportError:
        versions["faiss"] = "not-installed"

    try:
        import datasets  # type: ignore[import-untyped]

        versions["datasets"] = datasets.__version__
    except ImportError:
        versions["datasets"] = "not-installed"

    try:
        import transformers  # type: ignore[import-untyped]

        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "not-installed"

    try:
        import sentence_transformers  # type: ignore[import-untyped]

        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["sentence_transformers"] = "not-installed"

    return versions


def build_metadata(
    config: Dict[str, Any],
    config_path: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build run metadata dict.

    Args:
        config: Experiment config dict.
        config_path: Config source path or packaged-resource identity. Local
                     paths are never persisted in returned metadata.
        extra: Additional metadata to merge.

    Returns:
        Metadata dict.

    Example::

        >>> m = build_metadata({"seed": 42})
        >>> "timestamp" in m and "git_commit" in m
        True
    """
    from factuality_rag.reproducibility import sha256_json

    effective_config_sha256 = sha256_json(config)
    config_identity, config_source_sha256 = _config_metadata_identity(
        config_path,
        effective_config_sha256=effective_config_sha256,
    )
    git_commit, git_dirty = _get_git_state()
    meta: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_path": config_identity,
        "config_identity": config_identity,
        "config_source_sha256": config_source_sha256,
        "config_sha256": effective_config_sha256,
        "seed": config.get("seed", 42),
        "models": config.get("models", {}),
        "datasets": config.get("data", {}).get("datasets", []),
        "library_versions": _get_lib_versions(),
    }
    if extra:
        reserved = set(meta) & set(extra)
        if reserved:
            raise ValueError(
                "extra metadata must not override source/config identity fields: "
                + ", ".join(sorted(reserved))
            )
        meta.update(extra)
    return meta


def _resolve_support_metric(config: Mapping[str, Any], explicit: Optional[str]) -> str:
    """Resolve the explicit ``none``/``lexical`` evaluation mode.

    Legacy ``factscore`` names are rejected rather than silently reinterpreted
    as lexical overlap. Packaged configs use ``lexical_support`` when that
    diagnostic proxy is intentionally requested.
    """
    if explicit is not None:
        if explicit not in {"none", "lexical"}:
            raise ValueError("support_metric must be 'none' or 'lexical'")
        return explicit

    eval_config = config.get("eval", {})
    if eval_config is None:
        eval_config = {}
    if not isinstance(eval_config, Mapping):
        raise TypeError("config eval section must be a mapping")
    raw_metrics = eval_config.get("metrics", [])
    if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence):
        raise TypeError("config eval.metrics must be an ordered sequence")
    metrics = list(raw_metrics)
    if any(not isinstance(metric, str) for metric in metrics):
        raise TypeError("config eval.metrics entries must be strings")
    legacy = {"factscore", "factscore_stub"} & set(metrics)
    if legacy:
        raise ValueError(
            "legacy FactScore metric names are forbidden because the runner did "
            "not use an immutable NLI evaluator; request lexical_support explicitly"
        )
    return "lexical" if "lexical_support" in metrics else "none"


def load_reference_artifacts(
    run_dir: Union[str, Path],
) -> Tuple[Dict[str, Reference], Dict[str, Reference]]:
    """Load authoritative example-ID references and the legacy query lookup."""
    from factuality_rag.eval.metrics import reference_aliases

    root = Path(run_dir)
    by_example_id: Dict[str, Reference] = {}
    by_query: Dict[str, Reference] = {}

    by_id_path = root / "references_by_example_id.json"
    if by_id_path.exists():
        with open(by_id_path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("references_by_example_id.json must contain an object")
        for example_id, record in payload.items():
            if not isinstance(example_id, str) or not isinstance(record, dict):
                raise ValueError("invalid example-ID reference record")
            reference = cast(Reference, record.get("reference"))
            reference_aliases(reference)
            by_example_id[example_id] = reference

    legacy_path = root / "references.json"
    if legacy_path.exists():
        with open(legacy_path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("references.json must contain an object")
        for query, reference in payload.items():
            if not isinstance(query, str):
                raise ValueError("legacy reference queries must be strings")
            reference_aliases(reference)
            by_query[query] = reference

    return by_example_id, by_query


def resolve_record_reference(
    record: Mapping[str, Any],
    by_example_id: Mapping[str, Reference],
    by_query: Mapping[str, Reference],
) -> Optional[Reference]:
    """Resolve inline, example-ID, then unambiguous legacy query references."""
    inline = record.get("reference")
    if inline is not None:
        return cast(Reference, inline)
    example_id = record.get("example_id")
    if isinstance(example_id, str) and example_id in by_example_id:
        return by_example_id[example_id]
    query = record.get("input")
    if isinstance(query, str):
        return by_query.get(query)
    return None


def _copy_scored_passages_artifact(value: Any) -> List[Dict[str, Any]]:
    """Validate and copy the pipeline's minimal pre-threshold score artifact."""
    if not isinstance(value, list):
        raise ValueError("pipeline scored_passages must be a list")

    artifact: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"id", "final_score"}:
            raise ValueError(
                f"pipeline scored_passages[{index}] must contain only id and final_score"
            )
        passage_id = item.get("id")
        if not isinstance(passage_id, str) or not passage_id or passage_id != passage_id.strip():
            raise ValueError(f"pipeline scored_passages[{index}].id is invalid")
        if passage_id in seen_ids:
            raise ValueError(f"pipeline scored_passages contains duplicate id {passage_id!r}")
        seen_ids.add(passage_id)

        score = item.get("final_score")
        if isinstance(score, bool) or not isinstance(score, numbers.Real):
            raise ValueError(f"pipeline scored_passages[{index}].final_score must be numeric")
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(
                f"pipeline scored_passages[{index}].final_score must be finite and in [0, 1]"
            )
        artifact.append({"id": passage_id, "final_score": numeric_score})

    return artifact


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON numeric constants."""
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_json_float(value: str) -> float:
    """Parse a JSON float while rejecting overflow to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite")
    return parsed


def _reject_duplicate_json_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Build an object while rejecting duplicate JSON member names."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _load_strict_json_object(raw: bytes, label: str) -> Dict[str, Any]:
    """Load one UTF-8 JSON object with strict keys and finite numbers."""
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return cast(Dict[str, Any], payload)


def _canonical_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/integer equality aliasing."""
    from factuality_rag.reproducibility import canonical_json_bytes

    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _resume_bindings(
    *,
    base_metadata: Mapping[str, Any],
    queries: Sequence[str],
    references: Optional[Sequence[Reference]],
    mock_mode: bool,
    support_metric: str,
    gate: bool,
    rerank: bool,
    top_k: int,
    score_threshold: float,
) -> Dict[str, Any]:
    """Bind a checkpoint to the exact inputs and execution environment."""
    from factuality_rag.reproducibility import sha256_json

    return {
        "config": {
            "sha256": base_metadata["config_sha256"],
            "identity": base_metadata["config_identity"],
            "source_sha256": base_metadata["config_source_sha256"],
        },
        "inputs": {
            "queries_sha256": sha256_json(list(queries)),
            "references_sha256": sha256_json(references),
            "n_queries": len(queries),
            "has_references": references is not None,
        },
        "execution": {
            "seed": base_metadata["seed"],
            "mock_mode": mock_mode,
            "support_metric": support_metric,
            "gate": gate,
            "rerank": rerank,
            "top_k": top_k,
            "score_threshold": score_threshold,
        },
        "environment": {
            "git_commit": base_metadata["git_commit"],
            "git_dirty": base_metadata["git_dirty"],
            "runtime_source": base_metadata["runtime_source"],
            "library_versions": base_metadata["library_versions"],
        },
    }


def _build_resume_manifest(
    *,
    run_id: str,
    bindings: Mapping[str, Any],
    base_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the immutable pre-loop checkpoint identity."""
    from factuality_rag.reproducibility import canonical_json_bytes, sha256_json

    detached_metadata = cast(
        Dict[str, Any],
        json.loads(canonical_json_bytes(base_metadata).decode("utf-8")),
    )
    plain: Dict[str, Any] = {
        "schema": _RESUME_MANIFEST_SCHEMA,
        "run_id": run_id,
        "bindings": dict(bindings),
        "base_metadata": detached_metadata,
    }
    return {**plain, "manifest_sha256": sha256_json(plain)}


def _durable_create(path: Path, payload: bytes) -> None:
    """Create a file once and fsync its initial contents."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_resume_files(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Persist the pre-loop manifest before creating the checkpoint."""
    from factuality_rag.reproducibility import canonical_json_bytes

    manifest_path = run_dir / _RESUME_MANIFEST_NAME
    _durable_create(manifest_path, canonical_json_bytes(manifest) + b"\n")
    checkpoint_path = run_dir / "predictions.jsonl"
    _durable_create(checkpoint_path, b"")
    return checkpoint_path


def _load_resume_manifest(
    run_dir: Path,
    *,
    expected_bindings: Mapping[str, Any],
    current_metadata: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Load and validate an existing run's immutable checkpoint identity."""
    from factuality_rag.reproducibility import sha256_json

    manifest_path = run_dir / _RESUME_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"resume directory must contain a regular {_RESUME_MANIFEST_NAME} file")
    manifest = _load_strict_json_object(
        manifest_path.read_bytes(),
        _RESUME_MANIFEST_NAME,
    )
    expected_fields = {
        "schema",
        "run_id",
        "bindings",
        "base_metadata",
        "manifest_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("resume manifest does not match its exact schema")
    if manifest["schema"] != _RESUME_MANIFEST_SCHEMA:
        raise ValueError("resume manifest schema is unsupported")

    supplied_digest = manifest["manifest_sha256"]
    plain = dict(manifest)
    plain.pop("manifest_sha256")
    if (
        not isinstance(supplied_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", supplied_digest) is None
        or supplied_digest != sha256_json(plain)
    ):
        raise ValueError("resume manifest integrity check failed")

    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or run_id != run_dir.name:
        raise ValueError("resume manifest run_id does not match its directory")

    stored_bindings = manifest["bindings"]
    if not isinstance(stored_bindings, dict):
        raise ValueError("resume manifest bindings must be an object")
    if set(stored_bindings) != set(expected_bindings):
        raise ValueError("resume manifest binding schema does not match this runner")
    for name, expected in expected_bindings.items():
        if not _canonical_equal(stored_bindings[name], expected):
            raise ValueError(f"resume manifest {name} binding does not match current invocation")

    stored_metadata = manifest["base_metadata"]
    if not isinstance(stored_metadata, dict) or set(stored_metadata) != set(current_metadata):
        raise ValueError("resume manifest base metadata does not match its exact schema")
    timestamp = stored_metadata.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("resume manifest timestamp must be a string")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("resume manifest timestamp is invalid") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(
        parsed_timestamp
    ):
        raise ValueError("resume manifest timestamp must be UTC")

    stored_without_timestamp = dict(stored_metadata)
    current_without_timestamp = dict(current_metadata)
    stored_without_timestamp.pop("timestamp")
    current_without_timestamp.pop("timestamp")
    if not _canonical_equal(stored_without_timestamp, current_without_timestamp):
        raise ValueError("resume manifest metadata does not match current invocation")
    return run_id, cast(Dict[str, Any], stored_metadata)


def _validate_checkpoint_record(
    record: Dict[str, Any],
    *,
    index: int,
    run_id: str,
    query: str,
    reference: Optional[Reference],
    has_references: bool,
    seed: Any,
    mock_mode: bool,
    support_metric: str,
) -> None:
    """Validate one completed checkpoint record against its expected row."""
    required_fields = {
        "example_id",
        "input",
        "answer",
        "trusted_passages",
        "provenance",
        "confidence_tag",
        "retrieval_triggered",
        "scorer_enabled",
        "run_metadata",
    }
    allowed_fields = required_fields | {"scored_passages"}
    if has_references:
        required_fields.add("reference")
        allowed_fields.add("reference")
    if not required_fields.issubset(record) or not set(record).issubset(allowed_fields):
        raise ValueError(f"checkpoint record {index + 1} does not match its exact schema")

    expected_id = f"row-{index:08d}"
    if record["example_id"] != expected_id:
        raise ValueError(f"checkpoint record {index + 1} expected example_id {expected_id!r}")
    if record["input"] != query:
        raise ValueError(f"checkpoint record {index + 1} input does not match the dataset")
    if not isinstance(record["answer"], str):
        raise ValueError(f"checkpoint record {index + 1} answer must be a string")
    if not isinstance(record["trusted_passages"], list):
        raise ValueError(f"checkpoint record {index + 1} trusted_passages must be a list")
    if not isinstance(record["provenance"], (dict, list)):
        raise ValueError(f"checkpoint record {index + 1} provenance must be an object or list")
    if not isinstance(record["confidence_tag"], str):
        raise ValueError(f"checkpoint record {index + 1} confidence_tag must be a string")
    if type(record["retrieval_triggered"]) is not bool:
        raise ValueError(f"checkpoint record {index + 1} retrieval_triggered must be exactly bool")
    if type(record["scorer_enabled"]) is not bool:
        raise ValueError(f"checkpoint record {index + 1} scorer_enabled must be exactly bool")

    expected_run_metadata = {
        "run_id": run_id,
        "seed": seed,
        "mock_mode": mock_mode,
        "support_metric": support_metric,
        "scorer_enabled": record["scorer_enabled"],
    }
    if not _canonical_equal(record["run_metadata"], expected_run_metadata):
        raise ValueError(f"checkpoint record {index + 1} run metadata does not match")

    if has_references:
        if not _canonical_equal(record["reference"], reference):
            raise ValueError(f"checkpoint record {index + 1} reference does not match")
    elif "reference" in record:
        raise ValueError(f"checkpoint record {index + 1} unexpectedly contains a reference")

    if "scored_passages" in record:
        normalized_scores = _copy_scored_passages_artifact(record["scored_passages"])
        if not _canonical_equal(record["scored_passages"], normalized_scores):
            raise ValueError(f"checkpoint record {index + 1} scored_passages is not canonical")


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    run_id: str,
    queries: Sequence[str],
    references: Optional[Sequence[Reference]],
    seed: Any,
    mock_mode: bool,
    support_metric: str,
) -> List[Dict[str, Any]]:
    """Load durable records and discard only a final non-newline fragment."""
    if not checkpoint_path.exists():
        _durable_create(checkpoint_path, b"")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("predictions.jsonl must be a regular file")

    predictions: List[Dict[str, Any]] = []
    torn_offset: Optional[int] = None
    with checkpoint_path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.endswith(b"\n"):
                torn_offset = offset
                break
            index = len(predictions)
            if index >= len(queries):
                raise ValueError("checkpoint contains more records than the current dataset")
            record = _load_strict_json_object(
                raw_line,
                f"predictions.jsonl line {index + 1}",
            )
            reference = references[index] if references is not None else None
            _validate_checkpoint_record(
                record,
                index=index,
                run_id=run_id,
                query=queries[index],
                reference=reference,
                has_references=references is not None,
                seed=seed,
                mock_mode=mock_mode,
                support_metric=support_metric,
            )
            predictions.append(record)

        if torn_offset is not None:
            handle.seek(torn_offset)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            logger.warning(
                "Discarded an uncommitted final JSONL fragment from %s",
                checkpoint_path,
            )
    return predictions


def _append_checkpoint_record(
    handle: Any,
    record: Dict[str, Any],
    *,
    index: int,
    run_id: str,
    query: str,
    reference: Optional[Reference],
    has_references: bool,
    seed: Any,
    mock_mode: bool,
    support_metric: str,
) -> Dict[str, Any]:
    """Validate, append, flush, and fsync one checkpoint record."""
    _validate_checkpoint_record(
        record,
        index=index,
        run_id=run_id,
        query=query,
        reference=reference,
        has_references=has_references,
        seed=seed,
        mock_mode=mock_mode,
        support_metric=support_metric,
    )
    serialized = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    persisted = _load_strict_json_object(serialized, f"prediction record {index + 1}")
    _validate_checkpoint_record(
        persisted,
        index=index,
        run_id=run_id,
        query=query,
        reference=reference,
        has_references=has_references,
        seed=seed,
        mock_mode=mock_mode,
        support_metric=support_metric,
    )
    handle.write(serialized + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
    return persisted


def run(
    config: Dict[str, Any],
    queries: Optional[Sequence[str]] = None,
    references: Optional[Sequence[Reference]] = None,
    config_path: str = "",
    mock_mode: bool = False,
    runs_dir: str = "runs",
    run_id_prefix: Optional[str] = None,
    support_metric: Optional[str] = None,
    resume_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute an experiment: run the pipeline on each query, save results.

    Args:
        config: Parsed YAML config dict.
        queries: Non-empty ordered sequence of trimmed, non-blank query strings.
                 ``None`` is rejected; experiment inputs must be explicit.
        references: Optional parallel sequence whose items are a gold answer
                    string or a non-empty sequence of answer aliases.
        config_path: Explicit source identity/path for metadata. Empty means
                     the supplied config exists only in memory.
        mock_mode: Run all components in mock-mode.
        runs_dir: Base directory for run outputs.
        run_id_prefix: Optional prefix for the run directory name.
                       If set, the run ID starts with ``<prefix>_<timestamp>``
                       and ends with a random collision-resistant suffix.
        support_metric: Explicit ``"none"`` or ``"lexical"`` mode. When
                        omitted, ``eval.metrics`` may request ``lexical_support``.
        resume_dir: Existing run directory to resume. Its immutable manifest
                    must match the current config, inputs, and runtime flags.

    Returns:
        Dict with ``run_id``, ``predictions``, ``metrics``, and
        ``metadata``.

    Example::

        >>> result = run({"seed": 42}, queries=["test?"], mock_mode=True)
        >>> result["run_id"]  # doctest: +ELLIPSIS
        '...'
    """
    if resume_dir is not None and run_id_prefix is not None:
        raise ValueError("resume_dir and run_id_prefix are mutually exclusive")
    if resume_dir is not None and (
        not isinstance(resume_dir, (str, Path))
        or isinstance(resume_dir, str)
        and (not resume_dir or resume_dir != resume_dir.strip())
    ):
        raise ValueError("resume_dir must be a non-empty path")
    if run_id_prefix is not None and (
        not isinstance(run_id_prefix, str) or not _RUN_ID_PREFIX_RE.fullmatch(run_id_prefix)
    ):
        raise ValueError(
            "run_id_prefix must be 1-64 portable characters, start with an "
            "alphanumeric, and contain only alphanumerics, dot, underscore, or hyphen"
        )

    if type(mock_mode) is not bool:
        raise TypeError("mock_mode must be exactly bool")
    gating_cfg = config.get("gating", {})
    scorer_cfg = config.get("scorer", {})
    retriever_cfg = config.get("retriever", {})
    for name, value in (
        ("gating", gating_cfg),
        ("scorer", scorer_cfg),
        ("retriever", retriever_cfg),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"config.{name} must be a mapping")
    gate = gating_cfg.get("enabled", True)
    if type(gate) is not bool:
        raise TypeError("gating.enabled must be exactly bool")
    rerank = retriever_cfg.get("rerank", True)
    if type(rerank) is not bool:
        raise TypeError("retriever.rerank must be exactly bool")
    k = retriever_cfg.get("top_k", 10)
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or int(k) < 0:
        raise ValueError("retriever.top_k must be a non-negative integer")
    k = int(k)
    score_threshold = scorer_cfg.get("score_threshold", 0.4)
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, numbers.Real):
        raise TypeError("scorer.score_threshold must be a real number")
    score_threshold = float(score_threshold)
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("scorer.score_threshold must be finite and in [0, 1]")

    resolved_support_metric = _resolve_support_metric(config, support_metric)

    if queries is None:
        raise ValueError("queries must be provided explicitly; demo fallback is disabled")
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise TypeError("queries must be a non-empty ordered sequence of strings")
    query_list = list(queries)

    if not query_list:
        raise ValueError("queries must be non-empty")
    for index, query in enumerate(query_list):
        if not isinstance(query, str):
            raise TypeError(f"queries[{index}] must be a string")
        if not query or query != query.strip():
            raise ValueError(f"queries[{index}] must be non-blank and already trimmed")
        if any(unicodedata.category(character) == "Cc" for character in query):
            raise ValueError(f"queries[{index}] must not contain control characters")
    queries = query_list

    normalized_references: Optional[List[Reference]] = None
    if references is not None:
        if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
            raise TypeError("references must be an ordered sequence of reference items")
        if len(references) != len(queries):
            raise ValueError(
                "references and queries must have the same length: "
                f"{len(references)} != {len(queries)}"
            )
        normalized_references = []
        for reference in references:
            if isinstance(reference, str):
                aliases = [reference]
            elif isinstance(reference, Sequence):
                aliases = list(reference)
            else:
                raise TypeError("each reference must be a string or sequence of strings")
            if not aliases or any(
                not isinstance(alias, str) or not alias.strip() for alias in aliases
            ):
                raise ValueError("each prediction requires at least one non-blank string reference")
            normalized_references.append(aliases[0] if isinstance(reference, str) else aliases)

    seed = config.get("seed", 42)
    current_metadata = build_metadata(config, config_path)
    _validate_runtime_package_origin()
    current_metadata["runtime_source"] = _get_runtime_source_identity()
    bindings = _resume_bindings(
        base_metadata=current_metadata,
        queries=queries,
        references=normalized_references,
        mock_mode=mock_mode,
        support_metric=resolved_support_metric,
        gate=gate,
        rerank=rerank,
        top_k=k,
        score_threshold=score_threshold,
    )

    if resume_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uuid_suffix = uuid.uuid4().hex[:8]
        if run_id_prefix:
            run_id = f"{run_id_prefix}_{timestamp}_{uuid_suffix}"
        else:
            run_id = timestamp + "_" + uuid_suffix
        run_dir = Path(runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        base_metadata = current_metadata
        resume_manifest = _build_resume_manifest(
            run_id=run_id,
            bindings=bindings,
            base_metadata=base_metadata,
        )
        pred_file = _write_new_resume_files(run_dir, resume_manifest)
        logger.info("Initialized durable run '%s' at %s", run_id, run_dir)
        predictions: List[Dict[str, Any]] = []
    else:
        try:
            run_dir = Path(resume_dir).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError("resume_dir must identify an existing run directory") from exc
        if not run_dir.is_dir():
            raise ValueError("resume_dir must identify an existing run directory")
        run_id, base_metadata = _load_resume_manifest(
            run_dir,
            expected_bindings=bindings,
            current_metadata=current_metadata,
        )
        pred_file = run_dir / "predictions.jsonl"
        predictions = _load_checkpoint(
            pred_file,
            run_id=run_id,
            queries=queries,
            references=normalized_references,
            seed=seed,
            mock_mode=mock_mode,
            support_metric=resolved_support_metric,
        )
        logger.info(
            "Resuming run '%s' after %d / %d durable records",
            run_id,
            len(predictions),
            len(queries),
        )

    retrieval_count = sum(1 for record in predictions if record["retrieval_triggered"] is True)
    start_index = len(predictions)
    if start_index < len(queries):
        from factuality_rag.pipeline.orchestrator import Pipeline

        # Build the pipeline once for only the unfinished suffix.
        pipe = Pipeline(
            config_path=config_path,
            mock_mode=mock_mode,
            seed=seed,
            config=config,
        )
        with pred_file.open("ab") as checkpoint:
            for idx in range(start_index, len(queries)):
                query = queries[idx]
                info: Dict[str, Any] = {}
                answer, trusted, provenance, confidence = pipe.run(
                    query,
                    k=k,
                    gate=gate,
                    score_threshold=score_threshold,
                    seed=seed,
                    info=info,
                )
                retrieval_triggered = info.get("retrieval_triggered", True)
                scorer_enabled = info.get("scorer_enabled", True)
                if type(retrieval_triggered) is not bool:
                    raise TypeError("pipeline retrieval_triggered metadata must be exactly bool")
                if type(scorer_enabled) is not bool:
                    raise TypeError("pipeline scorer_enabled metadata must be exactly bool")
                if retrieval_triggered:
                    retrieval_count += 1

                record: Dict[str, Any] = {
                    "example_id": f"row-{idx:08d}",
                    "input": query,
                    "answer": answer,
                    "trusted_passages": trusted,
                    "provenance": provenance,
                    "confidence_tag": confidence,
                    "retrieval_triggered": retrieval_triggered,
                    "scorer_enabled": scorer_enabled,
                    "run_metadata": {
                        "run_id": run_id,
                        "seed": seed,
                        "mock_mode": mock_mode,
                        "support_metric": resolved_support_metric,
                        "scorer_enabled": scorer_enabled,
                    },
                }
                if "scored_passages" in info:
                    record["scored_passages"] = _copy_scored_passages_artifact(
                        info["scored_passages"]
                    )
                expected_reference: Optional[Reference] = (
                    normalized_references[idx] if normalized_references is not None else None
                )
                if normalized_references is not None:
                    record["reference"] = expected_reference

                persisted = _append_checkpoint_record(
                    checkpoint,
                    record,
                    index=idx,
                    run_id=run_id,
                    query=query,
                    reference=expected_reference,
                    has_references=normalized_references is not None,
                    seed=seed,
                    mock_mode=mock_mode,
                    support_metric=resolved_support_metric,
                )
                predictions.append(persisted)

                if (idx + 1) % 50 == 0:
                    logger.info("Progress: %d / %d queries", idx + 1, len(queries))

    logger.info("Saved %d predictions → %s", len(predictions), pred_file)

    from factuality_rag.eval.metrics import evaluate_predictions

    # Evaluate
    metrics = evaluate_predictions(
        predictions,
        references=normalized_references,
        support_metric=resolved_support_metric,
    )

    # Add retrieval rate
    metrics["retrieval_rate"] = retrieval_count / len(queries) if queries else 0.0
    metrics["retrieval_count"] = float(retrieval_count)

    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ambiguous_reference_queries = 0
    if normalized_references is not None:
        refs_by_example_id = {
            str(record["example_id"]): {
                "input": record["input"],
                "reference": record["reference"],
            }
            for record in predictions
        }
        with open(run_dir / "references_by_example_id.json", "w", encoding="utf-8") as f:
            json.dump(refs_by_example_id, f, indent=2, ensure_ascii=False)

        query_counts = Counter(queries)
        ambiguous_reference_queries = sum(1 for count in query_counts.values() if count > 1)
        ref_map = {
            query: reference
            for query, reference in zip(queries, normalized_references)
            if query_counts[query] == 1
        }
        refs_file = run_dir / "references.json"
        with open(refs_file, "w", encoding="utf-8") as f:
            json.dump(ref_map, f, indent=2, ensure_ascii=False)

    # Preserve the original run-start metadata across process restarts.
    metadata = dict(base_metadata)
    metadata.update(
        {
            "run_id": run_id,
            "n_queries": len(queries),
            "has_references": references is not None,
            "reference_artifact": "references_by_example_id.json"
            if references is not None
            else None,
            "ambiguous_legacy_reference_queries": ambiguous_reference_queries,
            "retrieval_rate": metrics["retrieval_rate"],
            "mock_mode": mock_mode,
            "support_metric": resolved_support_metric,
            "publication_artifact": False,
        }
    )
    meta_file = run_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Run '%s' complete → %s", run_id, run_dir)
    logger.info(
        "Metrics: EM=%.4f  F1=%.4f  LexicalSupport=%s  Ret%%=%.1f%%",
        metrics.get("exact_match", 0),
        metrics.get("f1", 0),
        metrics.get("lexical_support_answered_only", "disabled"),
        metrics.get("retrieval_rate", 0) * 100,
    )

    return {
        "run_id": run_id,
        "predictions": predictions,
        "metrics": metrics,
        "metadata": metadata,
        "run_dir": str(run_dir),
    }


# ── Dataset loading utilities ────────────────────────────────


def _resolve_dataset_selection(
    config: Mapping[str, Any],
    *,
    cli_dataset: Optional[str],
    cli_split: Optional[str],
    cli_sample: Optional[int],
) -> Tuple[str, str, Optional[int]]:
    """Resolve one exact dataset selection from CLI overrides or config.

    Experiment execution never falls back to demo questions.  A config with
    multiple dataset declarations requires an explicit ``--dataset`` choice so
    the effective config can be bound to exactly what ran.
    """

    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise TypeError("config.data must be a mapping")
    raw_datasets = data_cfg.get("datasets", [])
    if isinstance(raw_datasets, (str, bytes)) or not isinstance(raw_datasets, Sequence):
        raise TypeError("config.data.datasets must be a sequence of mappings")

    declared: List[Tuple[str, str]] = []
    for index, entry in enumerate(raw_datasets):
        if not isinstance(entry, Mapping):
            raise TypeError(f"config.data.datasets[{index}] must be a mapping")
        name = entry.get("name")
        split = entry.get("split", "validation")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"config.data.datasets[{index}].name must be non-empty and trimmed")
        if not isinstance(split, str) or not split or split != split.strip():
            raise ValueError(f"config.data.datasets[{index}].split must be non-empty and trimmed")
        declared.append((name, split))

    if cli_dataset is not None:
        if (
            not isinstance(cli_dataset, str)
            or not cli_dataset
            or cli_dataset != cli_dataset.strip()
        ):
            raise ValueError("--dataset must be non-empty and trimmed")
        dataset_name = cli_dataset
        matching = [item for item in declared if item[0].casefold() == cli_dataset.casefold()]
        configured_split = matching[0][1] if len(matching) == 1 else "validation"
        configured_sample = data_cfg.get("dev_sample_size") if len(matching) == 1 else None
    else:
        if len(declared) != 1:
            raise ValueError(
                "experiment config must declare exactly one dataset when --dataset is omitted"
            )
        dataset_name, configured_split = declared[0]
        configured_sample = data_cfg.get("dev_sample_size")

    split = cli_split if cli_split is not None else configured_split
    if not isinstance(split, str) or not split or split != split.strip():
        raise ValueError("dataset split must be non-empty and trimmed")
    sample = cli_sample if cli_sample is not None else configured_sample
    if sample is not None and (
        isinstance(sample, bool) or not isinstance(sample, numbers.Integral) or int(sample) <= 0
    ):
        raise ValueError("dataset sample size must be a positive integer or null")
    return dataset_name, split, int(sample) if sample is not None else None


def _extract_queries_and_references(
    dataset_name: str,
    split: str = "validation",
    sample: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[Reference]]:
    """Load queries and reference answers from a HuggingFace dataset.

    Generic QA extraction is available for the enabled loader adapters.
    Classification and multiple-choice datasets fail closed pending dedicated
    task adapters.

    Args:
        dataset_name: Dataset identifier (e.g. ``"natural_questions"``).
        split: Dataset split.
        sample: Number of examples to sample (``None`` = all).
        seed: Random seed for sampling.

    Returns:
        ``(queries, references)`` — parallel lists of question
        strings and gold answer strings or alias lists.
    """
    from factuality_rag.data.loader import load_dataset

    ds = load_dataset(dataset_name, split=split, dev_sample_size=sample, seed=seed)

    queries: List[str] = []
    references: List[Reference] = []

    for row in ds:
        # Extract question — column name varies by dataset
        q = row.get("question", "") or row.get("query", "") or row.get("claim", "")
        if isinstance(q, dict):
            q = q.get("text", str(q))
        q = str(q).strip()
        if not q:
            continue

        # Extract reference answer (dataset-specific)
        ref = _extract_reference(row, dataset_name)
        queries.append(q)
        references.append(ref)

    logger.info(
        "Loaded %d queries from %s/%s (sample=%s)",
        len(queries),
        dataset_name,
        split,
        sample,
    )
    return queries, references


def _reference_from_values(*values: Any) -> Reference:
    """Flatten answer fields into a stable, de-duplicated reference."""
    aliases: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate and candidate not in aliases:
                aliases.append(candidate)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)
        elif value is not None:
            candidate = str(value).strip()
            if candidate and candidate not in aliases:
                aliases.append(candidate)

    for value in values:
        visit(value)
    if not aliases:
        return ""
    return aliases[0] if len(aliases) == 1 else aliases


def _extract_reference(row: Dict[str, Any], dataset_name: str) -> Reference:
    """Extract one gold reference, preserving every supplied answer alias.

    Args:
        row: A single row from the HF dataset.
        dataset_name: Name of the dataset.

    Returns:
        A reference answer string or a list of accepted aliases.
    """
    dn = dataset_name.lower()

    if "popqa" in dn:
        raise NotImplementedError("PopQA requires a metadata-preserving popularity-stratum adapter")
    if "hagrid" in dn:
        raise NotImplementedError(
            "HAGRID requires informative/attributable answer and quotation evaluation"
        )

    # ── 2WikiMultiHopQA: answer field is a string ────────────
    if "2wiki" in dn or "wikimultihop" in dn:
        return str(row.get("answer", ""))

    # FEVER is a classification task, not generic free-form QA.
    if "fever" in dn:
        raise NotImplementedError(
            "FEVER requires a task-specific classification prompt and evaluator"
        )

    # ── NQ-Open: answer is a list of strings ─────────────────
    answer = row.get("answer", "")
    if isinstance(answer, list):
        return _reference_from_values(answer)

    if isinstance(answer, dict):
        return _reference_from_values(
            answer.get("value", ""),
            answer.get("text", ""),
            answer.get("normalized_aliases", ""),
            answer.get("aliases", ""),
        )

    # ── HotpotQA: answer field is a string ───────────────────
    if "hotpot" in dn:
        return str(row.get("answer", ""))

    # The configured TruthfulQA source is multiple-choice and requires choices
    # plus its integer label; generic free-form QA extraction would be invalid.
    if "truthful" in dn:
        raise NotImplementedError(
            "TruthfulQA MC requires a task-specific choices/label prompt and evaluator"
        )

    return _reference_from_values(answer)


def _apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-path YAML overrides to a config dict.

    Each override has the form ``"key.subkey=value"``.

    Args:
        config: The base config dict to modify (mutated in place).
        overrides: List of override strings.

    Returns:
        The modified config dict.

    Example::

        >>> cfg = {"scorer": {"score_threshold": 0.4}}
        >>> _apply_overrides(cfg, ["scorer.score_threshold=0.3"])
        {'scorer': {'score_threshold': 0.3}}
    """
    import yaml as _yaml

    for override in overrides:
        if "=" not in override:
            logger.warning("Skipping invalid override (no '='): %s", override)
            continue
        key_path, value_str = override.split("=", 1)
        keys = key_path.strip().split(".")

        # Parse value (try YAML-style: numbers, bools, strings)
        try:
            value = _yaml.safe_load(value_str)
        except Exception:
            value = value_str

        # Navigate to parent dict
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        logger.info("Override: %s = %s", key_path, value)

    return config


# ── CLI entry point ──────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for experiment_runner."""
    p = argparse.ArgumentParser(
        prog="python -m factuality_rag.experiment_runner",
        description="Run structured experiments with dataset loading and metric tracking.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit experiment YAML path (omitted: packaged exp_full_pipeline.yaml).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Exact HF dataset selection; required when config declares multiple datasets.",
    )
    p.add_argument(
        "--split", type=str, default=None, help="Dataset split override (default: config)."
    )
    p.add_argument("--sample", type=int, default=None, help="Number of queries to sample.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    run_identity = p.add_mutually_exclusive_group()
    run_identity.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Prefix for a new run directory name.",
    )
    run_identity.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an existing manifest-bound run directory.",
    )
    p.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="YAML dot-path overrides, e.g. scorer.score_threshold=0.3",
    )
    p.add_argument("--mock", action="store_true", help="Run in mock mode.")
    p.add_argument(
        "--support-metric",
        choices=["none", "lexical"],
        default=None,
        help="Evidence-support mode (default: derive explicit lexical_support from config).",
    )
    p.add_argument("--runs-dir", type=str, default="runs", help="Base directory for outputs.")
    return p.parse_args()


def main() -> None:
    """CLI entry point for experiment runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_args()

    from factuality_rag.pipeline.orchestrator import _load_config
    from factuality_rag.resources import (
        DEFAULT_EXPERIMENT_CONFIG,
        experiment_config_identity,
    )

    # ``None`` deliberately selects package data. Resolve an explicit path once
    # before loading; persisted metadata later uses only a path-free digest
    # identity, never this absolute local path.
    explicit_config_path = (
        Path(args.config).expanduser().resolve(strict=True) if args.config is not None else None
    )
    config = _load_config(
        str(explicit_config_path) if explicit_config_path is not None else None,
        default_resource=DEFAULT_EXPERIMENT_CONFIG,
    )
    config_identity = (
        str(explicit_config_path)
        if explicit_config_path is not None
        else experiment_config_identity(DEFAULT_EXPERIMENT_CONFIG)
    )

    # Apply seed override
    if args.seed is not None:
        config["seed"] = args.seed

    # Apply dot-path overrides
    if args.override:
        _apply_overrides(config, args.override)

    # Set numpy/random seeds
    seed = config.get("seed", 42)
    np.random.seed(seed)

    dataset_name, split, sample = _resolve_dataset_selection(
        config,
        cli_dataset=args.dataset,
        cli_split=args.split,
        cli_sample=args.sample,
    )
    # Persist the effective selection, not a broader or stale YAML declaration.
    data_cfg = config.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError("config.data must be a mutable mapping")
    data_cfg["datasets"] = [{"name": dataset_name, "split": split}]
    data_cfg["dev_sample_size"] = sample

    queries, references = _extract_queries_and_references(
        dataset_name=dataset_name,
        split=split,
        sample=sample,
        seed=seed,
    )

    pipeline_cfg = config.get("pipeline", {})
    if not isinstance(pipeline_cfg, Mapping):
        raise TypeError("config.pipeline must be a mapping")
    configured_mock_mode = pipeline_cfg.get("mock_mode", False)
    if type(configured_mock_mode) is not bool:
        raise TypeError("pipeline.mock_mode must be exactly bool")
    mock_mode = True if args.mock else configured_mock_mode

    # Run experiment
    result = run(
        config=config,
        queries=queries,
        references=references,
        config_path=config_identity,
        mock_mode=mock_mode,
        runs_dir=args.runs_dir,
        run_id_prefix=args.run_id,
        support_metric=getattr(args, "support_metric", None),
        resume_dir=getattr(args, "resume", None),
    )

    print(f"\n{'=' * 60}")
    print(f"Run ID:    {result['run_id']}")
    print(f"Run Dir:   {result['run_dir']}")
    print(f"Queries:   {len(result['predictions'])}")
    print("Metrics:")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
