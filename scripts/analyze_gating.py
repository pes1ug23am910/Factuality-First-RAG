#!/usr/bin/env python
"""
scripts/analyze_gating.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 4A: exploratory gating/closed-book-error proxy analysis.

Compares gating decisions with a limited proxy: whether the closed-book answer
is exact-match wrong.  That proxy does not establish that retrieval would help,
so this script does not estimate an oracle policy or retrieval utility.  Its
outputs are explicitly exploratory and are not publication claims.

Usage::

    python scripts/analyze_gating.py \\
        --config configs/exp_full_pipeline.yaml \\
        --dataset natural_questions \\
        --split validation \\
        --sample 500 \\
        --seed 42 \\
        --output analysis/gating_closedbook_error_proxy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

Reference = Union[str, Sequence[str]]

GATING_ANALYSIS_SCHEMA = "factuality-rag.gating-closedbook-error-proxy-analysis.v1"
CLOSED_BOOK_CONFIG_PATH = Path("configs/exp_b1_closed_book.yaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exploratory gating proxy analysis.")
    p.add_argument("--config", type=str, default="configs/exp_full_pipeline.yaml")
    p.add_argument(
        "--full-run",
        type=str,
        default=None,
        help="Path to full pipeline run directory (loads predictions).",
    )
    p.add_argument(
        "--closedbook-run",
        type=str,
        default=None,
        help="Path to closed-book run directory (loads predictions).",
    )
    p.add_argument("--dataset", type=str, default="natural_questions")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/gating_closedbook_error_proxy.json")
    p.add_argument("--mock", action="store_true", help="Run in mock mode for testing.")
    return p.parse_args()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token!r}")
    return value


def _object_without_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_float=_parse_finite_float,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_artifact(path: Path, label: str) -> Tuple[bytes, str]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    return raw, text


def _portable_path_key(path: Path) -> Tuple[str, ...]:
    """Return a conservative identity key with Windows filename semantics."""
    try:
        absolute = path.resolve(strict=False)
    except OSError:
        absolute = Path(os.path.abspath(path))
    normalized = str(absolute).replace("\\", "/")
    return tuple(
        component.rstrip(" .").casefold()
        for component in normalized.split("/")
        if component not in ("", ".")
    )


def _paths_alias(left: Path, right: Path) -> bool:
    if _portable_path_key(left) == _portable_path_key(right):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _assert_output_disjoint(output_path: Path, consumed_paths: Sequence[Path]) -> None:
    for consumed_path in consumed_paths:
        if _paths_alias(output_path, consumed_path):
            raise ValueError(
                f"output path must not alias consumed input artifact: {consumed_path.name}"
            )


def _snapshot_yaml_config(path: Path, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Hash one immutable byte snapshot and parse that exact snapshot."""
    import yaml  # type: ignore[import-untyped]

    raw, _ = _read_artifact(path, label)
    try:
        loaded: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is not valid YAML: {exc}") from exc
    if loaded is None:
        config: Dict[str, Any] = {}
    elif isinstance(loaded, dict):
        config = cast(Dict[str, Any], loaded)
    else:
        raise ValueError(f"{label} must contain a YAML mapping")
    artifact = {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(raw),
        "byte_count": len(raw),
    }
    return config, artifact


def _require_trimmed_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and already trimmed")
    return value


def _require_path_argument(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string path")
    if not value.strip():
        raise ValueError(f"{label} must be a non-blank path")
    return value


def _optional_example_id(record: Mapping[str, Any], label: str) -> Optional[str]:
    if "example_id" not in record:
        return None
    return _require_trimmed_string(record["example_id"], f"{label}.example_id")


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be exactly bool")
    return value


def _validate_reference(reference: Any, label: str) -> Reference:
    from factuality_rag.eval.metrics import reference_aliases

    try:
        reference_aliases(reference)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{label}: {exc}") from exc
    return cast(Reference, reference)


def _reference_key(reference: Reference) -> frozenset[str]:
    from factuality_rag.eval.metrics import reference_aliases

    return frozenset(alias.strip().casefold() for alias in reference_aliases(reference))


def _read_predictions(
    run_dir: Path,
    role: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    raw, text = _read_artifact(path, f"{role} predictions")
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, f"{role} predictions line {line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"{role} predictions line {line_number} must be a JSON object")
        record_index = len(records)
        record_label = f"{role} predictions[{record_index}]"
        _require_trimmed_string(value.get("input"), f"{record_label}.input")
        example_id = _optional_example_id(value, record_label)
        if example_id is not None:
            if example_id in seen_ids:
                raise ValueError(f"duplicate {role} prediction example_id: {example_id}")
            seen_ids.add(example_id)
        if "reference" in value and value["reference"] is not None:
            _validate_reference(value["reference"], f"{record_label}.reference")
        if role == "full-run":
            _require_exact_bool(
                value.get("retrieval_triggered"),
                f"{record_label}.retrieval_triggered",
            )
        else:
            if type(value.get("answer")) is not str:
                raise TypeError(f"{record_label}.answer must be a string")
        records.append(value)

    provenance = {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(raw),
        "record_count": len(records),
    }
    return records, provenance


def _read_json_object(path: Path, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw, text = _read_artifact(path, label)
    value = _strict_json_loads(text, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    provenance = {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(raw),
        "record_count": len(value),
    }
    return value, provenance


def _references_agree(left: Reference, right: Reference) -> bool:
    return _reference_key(left) == _reference_key(right)


def _load_run_references(
    run_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    role: str,
) -> Tuple[Dict[str, Tuple[str, Reference]], Dict[str, Reference], Dict[str, Any]]:
    """Load one run's references without crossing its example-ID namespace."""
    by_id: Dict[str, Tuple[str, Reference]] = {}
    by_query: Dict[str, Reference] = {}
    provenance: Dict[str, Any] = {
        "references_by_example_id": None,
        "legacy_references": None,
    }

    by_id_path = run_dir / "references_by_example_id.json"
    if by_id_path.exists():
        payload, artifact = _read_json_object(
            by_id_path,
            f"{role} references_by_example_id.json",
        )
        provenance["references_by_example_id"] = artifact
        for example_id_value, record in payload.items():
            artifact_example_id = _require_trimmed_string(
                example_id_value,
                f"{role} references_by_example_id key",
            )
            if not isinstance(record, dict):
                raise ValueError(
                    f"{role} reference record for {artifact_example_id!r} must be a JSON object"
                )
            query = _require_trimmed_string(
                record.get("input"),
                f"{role} references_by_example_id[{artifact_example_id!r}].input",
            )
            if "reference" not in record:
                raise ValueError(
                    f"{role} references_by_example_id[{artifact_example_id!r}] is missing reference"
                )
            reference = _validate_reference(
                record["reference"],
                f"{role} references_by_example_id[{artifact_example_id!r}].reference",
            )
            by_id[artifact_example_id] = (query, reference)

    legacy_path = run_dir / "references.json"
    if legacy_path.exists():
        payload, artifact = _read_json_object(legacy_path, f"{role} references.json")
        provenance["legacy_references"] = artifact
        for query_value, reference_value in payload.items():
            query = _require_trimmed_string(query_value, f"{role} legacy reference query")
            by_query[query] = _validate_reference(
                reference_value,
                f"{role} references.json[{query!r}]",
            )

    prediction_id_list: List[str] = []
    for index, prediction in enumerate(predictions):
        prediction_example_id = _optional_example_id(
            prediction,
            f"{role} predictions[{index}]",
        )
        if prediction_example_id is not None:
            prediction_id_list.append(prediction_example_id)
    prediction_ids = set(prediction_id_list)
    if provenance["references_by_example_id"] is not None and (
        len(prediction_id_list) != len(predictions) or prediction_ids != set(by_id)
    ):
        predictions_without_ids = len(predictions) - len(prediction_id_list)
        missing = predictions_without_ids + len(prediction_ids - set(by_id))
        extra = len(set(by_id) - prediction_ids)
        raise ValueError(
            f"{role} example-ID reference coverage mismatch: missing={missing}, extra={extra}"
        )

    query_counts = Counter(prediction["input"] for prediction in predictions)
    for query in by_query:
        if query_counts[query] != 1:
            raise ValueError(
                f"{role} legacy reference query must bind exactly one prediction: {query!r}"
            )

    for index, prediction in enumerate(predictions):
        query = prediction["input"]
        prediction_example_id = _optional_example_id(
            prediction,
            f"{role} predictions[{index}]",
        )
        candidates: List[Tuple[str, Reference]] = []
        if "reference" in prediction and prediction["reference"] is not None:
            candidates.append(("inline", prediction["reference"]))
        if prediction_example_id is not None and prediction_example_id in by_id:
            stored_query, stored_reference = by_id[prediction_example_id]
            if stored_query != query:
                raise ValueError(
                    f"{role} stored query mismatch for example_id {prediction_example_id!r}"
                )
            candidates.append(("example-ID artifact", stored_reference))
        if query in by_query:
            candidates.append(("legacy query artifact", by_query[query]))
        if candidates:
            expected_source, expected = candidates[0]
            for source, candidate in candidates[1:]:
                if not _references_agree(expected, candidate):
                    raise ValueError(
                        f"{role} conflicting references between {expected_source} and {source} "
                        f"for predictions[{index}]"
                    )

    return by_id, by_query, provenance


def _resolve_run_reference(
    record: Mapping[str, Any],
    by_id: Mapping[str, Tuple[str, Reference]],
    by_query: Mapping[str, Reference],
    label: str,
) -> Optional[Reference]:
    """Resolve inline, then same-run ID, then same-run unambiguous query reference."""
    query = record["input"]
    candidates: List[Tuple[str, Reference]] = []
    if "reference" in record and record["reference"] is not None:
        candidates.append(
            ("inline", _validate_reference(record["reference"], f"{label}.reference"))
        )
    example_id = _optional_example_id(record, label)
    if example_id is not None and example_id in by_id:
        stored_query, reference = by_id[example_id]
        if stored_query != query:
            raise ValueError(f"{label} query does not match stored example-ID reference")
        candidates.append(("example-ID artifact", reference))
    if query in by_query:
        candidates.append(("legacy query artifact", by_query[query]))
    if not candidates:
        return None
    source, reference = candidates[0]
    for other_source, candidate in candidates[1:]:
        if not _references_agree(reference, candidate):
            raise ValueError(f"{label} has conflicting {source} and {other_source} references")
    return reference


def _match_closedbook_records(
    full_predictions: Sequence[Mapping[str, Any]],
    closedbook_predictions: Sequence[Mapping[str, Any]],
) -> List[Tuple[int, str]]:
    """Return a one-to-one closed-book index/method pairing for every full record."""
    closedbook_by_id: Dict[str, int] = {}
    for index, prediction in enumerate(closedbook_predictions):
        example_id = _optional_example_id(prediction, f"closed-book predictions[{index}]")
        if example_id is not None:
            closedbook_by_id[example_id] = index
    query_counts = Counter(prediction["input"] for prediction in closedbook_predictions)
    unique_query_index = {
        prediction["input"]: index
        for index, prediction in enumerate(closedbook_predictions)
        if query_counts[prediction["input"]] == 1
    }

    matches: List[Tuple[int, str]] = []
    consumed: set[int] = set()
    for index, prediction in enumerate(full_predictions):
        query = prediction["input"]
        example_id = _optional_example_id(prediction, f"full-run predictions[{index}]")
        matched_index: Optional[int] = None
        method = ""

        if example_id is not None and example_id in closedbook_by_id:
            candidate_index = closedbook_by_id[example_id]
            candidate_query = closedbook_predictions[candidate_index]["input"]
            if candidate_query != query:
                raise ValueError(f"closed-book query mismatch for shared example_id {example_id!r}")
            matched_index = candidate_index
            method = "example_id"
        elif (
            index < len(closedbook_predictions) and closedbook_predictions[index]["input"] == query
        ):
            matched_index = index
            method = "position_and_query"
        elif query in unique_query_index:
            matched_index = unique_query_index[query]
            method = "unique_query"

        if matched_index is None:
            raise ValueError(
                f"no unambiguous closed-book record matches full-run predictions[{index}]"
            )
        closedbook_example_id = _optional_example_id(
            closedbook_predictions[matched_index],
            f"closed-book predictions[{matched_index}]",
        )
        if (
            example_id is not None
            and closedbook_example_id is not None
            and example_id != closedbook_example_id
        ):
            raise ValueError(
                "full-run and closed-book example_id values differ for fallback-matched "
                f"predictions[{index}]"
            )
        if matched_index in consumed:
            raise ValueError(
                f"closed-book predictions[{matched_index}] would match more than one full record"
            )
        consumed.add(matched_index)
        matches.append((matched_index, method))

    unused = len(closedbook_predictions) - len(consumed)
    if unused:
        raise ValueError(f"closed-book run contains {unused} unmatched prediction record(s)")
    return matches


def _offline_analysis(
    full_dir: Path,
    closedbook_dir: Path,
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], List[bool], List[bool], Dict[str, Any], Dict[str, Any]]:
    from factuality_rag.eval.metrics import compute_em_aliases

    if _paths_alias(full_dir, closedbook_dir):
        raise ValueError("--full-run and --closedbook-run must identify different run directories")

    full_predictions, full_prediction_artifact = _read_predictions(full_dir, "full-run")
    closedbook_predictions, closedbook_prediction_artifact = _read_predictions(
        closedbook_dir,
        "closed-book",
    )
    if not full_predictions:
        raise ValueError("No gating decisions were available for analysis")

    full_by_id, full_by_query, full_reference_artifacts = _load_run_references(
        full_dir,
        full_predictions,
        "full-run",
    )
    closedbook_by_id, closedbook_by_query, closedbook_reference_artifacts = _load_run_references(
        closedbook_dir,
        closedbook_predictions,
        "closed-book",
    )
    consumed_paths = [
        full_dir / "predictions.jsonl",
        closedbook_dir / "predictions.jsonl",
    ]
    for run_dir in (full_dir, closedbook_dir):
        for filename in ("references_by_example_id.json", "references.json"):
            artifact_path = run_dir / filename
            if artifact_path.exists():
                consumed_paths.append(artifact_path)
    _assert_output_disjoint(output_path, consumed_paths)
    matches = _match_closedbook_records(full_predictions, closedbook_predictions)

    results: List[Dict[str, Any]] = []
    gate_decisions: List[bool] = []
    proxy_decisions: List[bool] = []
    for index, (prediction, (closedbook_index, match_method)) in enumerate(
        zip(full_predictions, matches)
    ):
        query = prediction["input"]
        reference = _resolve_run_reference(
            prediction,
            full_by_id,
            full_by_query,
            f"full-run predictions[{index}]",
        )
        if reference is None:
            raise ValueError(f"unresolved reference for full-run predictions[{index}]")

        closedbook_prediction = closedbook_predictions[closedbook_index]
        closedbook_reference = _resolve_run_reference(
            closedbook_prediction,
            closedbook_by_id,
            closedbook_by_query,
            f"closed-book predictions[{closedbook_index}]",
        )
        if closedbook_reference is not None and not _references_agree(
            reference,
            closedbook_reference,
        ):
            raise ValueError(f"closed-book reference mismatch for full-run predictions[{index}]")

        closedbook_answer = closedbook_prediction["answer"]
        closedbook_correct = compute_em_aliases(closedbook_answer, reference) > 0.5
        closedbook_error_proxy = not closedbook_correct
        retrieval_triggered = _require_exact_bool(
            prediction["retrieval_triggered"],
            f"full-run predictions[{index}].retrieval_triggered",
        )

        gate_decisions.append(retrieval_triggered)
        proxy_decisions.append(closedbook_error_proxy)
        results.append(
            {
                "example_id": prediction.get("example_id"),
                "query": query,
                "reference": reference,
                "closed_book_answer": closedbook_answer,
                "closed_book_correct": closedbook_correct,
                "closed_book_match_method": match_method,
                "gate_decision": retrieval_triggered,
                "closed_book_error_proxy": closedbook_error_proxy,
                "gate_matches_proxy": retrieval_triggered == closedbook_error_proxy,
            }
        )

    full_inputs = {
        "predictions": full_prediction_artifact,
        **full_reference_artifacts,
    }
    closedbook_inputs = {
        "predictions": closedbook_prediction_artifact,
        **closedbook_reference_artifacts,
    }
    hash_binding = {
        "mode": "offline",
        "full_run": {
            key: value["sha256"] if value is not None else None
            for key, value in full_inputs.items()
        },
        "closedbook_run": {
            key: value["sha256"] if value is not None else None
            for key, value in closedbook_inputs.items()
        },
    }
    inputs = {
        "sha256": _sha256_json(hash_binding),
        "full_run": full_inputs,
        "closedbook_run": closedbook_inputs,
    }
    n_analyzed = len(results)
    coverage = {
        "n_full_records": len(full_predictions),
        "n_closedbook_records": len(closedbook_predictions),
        "n_references_resolved": n_analyzed,
        "n_closedbook_matched": n_analyzed,
        "n_analyzed": n_analyzed,
        "n_unresolved_references": 0,
        "n_unmatched_full_records": 0,
        "n_unused_closedbook_records": 0,
        "analysis_fraction": n_analyzed / len(full_predictions),
    }
    return results, gate_decisions, proxy_decisions, inputs, coverage


def _live_analysis(
    args: argparse.Namespace,
    output_path: Path,
) -> Tuple[
    List[Dict[str, Any]],
    List[bool],
    List[bool],
    Dict[str, Any],
    Dict[str, Any],
]:
    from factuality_rag.eval.metrics import compute_em_aliases

    full_config_path = Path(args.config)
    closedbook_config_path = CLOSED_BOOK_CONFIG_PATH
    full_config, full_config_artifact = _snapshot_yaml_config(
        full_config_path,
        "full-pipeline config",
    )
    closedbook_config, closedbook_config_artifact = _snapshot_yaml_config(
        closedbook_config_path,
        "closed-book config",
    )
    _assert_output_disjoint(
        output_path,
        [full_config_path, closedbook_config_path],
    )

    from factuality_rag.pipeline.orchestrator import Pipeline

    pipe_closed = Pipeline(
        config_path=str(closedbook_config_path),
        mock_mode=args.mock,
        seed=args.seed,
        config=closedbook_config,
    )
    pipe_full = Pipeline(
        config_path=str(full_config_path),
        mock_mode=args.mock,
        seed=args.seed,
        config=full_config,
    )

    if args.mock:
        queries = ["What is the capital of France?", "Who wrote Hamlet?", "What is DNA?"]
        references_list: Sequence[Reference] = [
            "Paris",
            "Shakespeare",
            "Deoxyribonucleic acid",
        ]
    else:
        from factuality_rag.experiment_runner import _extract_queries_and_references

        queries, references_list = _extract_queries_and_references(
            args.dataset,
            args.split,
            args.sample,
            args.seed,
        )

    if len(queries) != len(references_list):
        raise ValueError("live gating queries and references must have the same length")
    if not queries:
        raise ValueError("No gating decisions were available for analysis")

    results: List[Dict[str, Any]] = []
    gate_decisions: List[bool] = []
    proxy_decisions: List[bool] = []
    canonical_inputs: List[Dict[str, Any]] = []

    for index, (query_value, reference_value) in enumerate(zip(queries, references_list)):
        query = _require_trimmed_string(query_value, f"live inputs[{index}].input")
        reference = _validate_reference(reference_value, f"live inputs[{index}].reference")
        canonical_inputs.append({"input": query, "reference": reference})

        closedbook_answer, _, _, _ = pipe_closed.run(query, gate=False)
        if type(closedbook_answer) is not str:
            raise TypeError(f"closed-book live result[{index}].answer must be a string")
        closedbook_correct = compute_em_aliases(closedbook_answer, reference) > 0.5
        closedbook_error_proxy = not closedbook_correct

        info: Dict[str, Any] = {}
        pipe_full.run(query, info=info)
        retrieval_triggered = _require_exact_bool(
            info.get("retrieval_triggered"),
            f"full live result[{index}].retrieval_triggered",
        )

        gate_decisions.append(retrieval_triggered)
        proxy_decisions.append(closedbook_error_proxy)
        results.append(
            {
                "example_id": None,
                "query": query,
                "reference": reference,
                "closed_book_answer": closedbook_answer,
                "closed_book_correct": closedbook_correct,
                "closed_book_match_method": "live_same_input",
                "gate_decision": retrieval_triggered,
                "closed_book_error_proxy": closedbook_error_proxy,
                "gate_matches_proxy": retrieval_triggered == closedbook_error_proxy,
            }
        )

        if (index + 1) % 50 == 0:
            logger.info("Processed %d / %d queries.", index + 1, len(queries))

    records_sha256 = _sha256_json(canonical_inputs)
    hash_binding = {
        "mode": "live",
        "records_sha256": records_sha256,
        "full_config_sha256": full_config_artifact["sha256"],
        "closedbook_config_sha256": closedbook_config_artifact["sha256"],
        "dataset": args.dataset,
        "split": args.split,
        "sample": args.sample,
        "seed": args.seed,
        "mock": args.mock,
    }
    inputs = {
        "sha256": _sha256_json(hash_binding),
        "records": {"sha256": records_sha256, "record_count": len(canonical_inputs)},
        "full_config": full_config_artifact,
        "closedbook_config": closedbook_config_artifact,
        "selection": {
            "dataset": args.dataset,
            "split": args.split,
            "sample": args.sample,
            "seed": args.seed,
            "mock": args.mock,
        },
    }
    n_analyzed = len(results)
    coverage = {
        "n_full_records": n_analyzed,
        "n_closedbook_records": n_analyzed,
        "n_references_resolved": n_analyzed,
        "n_closedbook_matched": n_analyzed,
        "n_analyzed": n_analyzed,
        "n_unresolved_references": 0,
        "n_unmatched_full_records": 0,
        "n_unused_closedbook_records": 0,
        "analysis_fraction": 1.0,
    }
    return results, gate_decisions, proxy_decisions, inputs, coverage


def _compute_metrics(
    gate_decisions: Sequence[bool], proxy_decisions: Sequence[bool]
) -> Dict[str, Any]:
    if not gate_decisions:
        raise ValueError("No gating decisions were available for analysis")
    if len(gate_decisions) != len(proxy_decisions):
        raise ValueError("gate and closed-book-error proxy decision counts differ")

    gate_arr = np.asarray(gate_decisions, dtype=bool)
    proxy_arr = np.asarray(proxy_decisions, dtype=bool)
    tp = int((gate_arr & proxy_arr).sum())
    fp = int((gate_arr & ~proxy_arr).sum())
    fn = int((~gate_arr & proxy_arr).sum())
    tn = int((~gate_arr & ~proxy_arr).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / len(gate_decisions)
    return {
        "n_queries": len(gate_decisions),
        "n_analyzed": len(gate_decisions),
        "agreement_with_proxy": round(accuracy, 4),
        "retrieve_precision_against_proxy": round(precision, 4),
        "retrieve_recall_against_proxy": round(recall, 4),
        "retrieve_f1_against_proxy": round(f1, 4),
        "gate_and_proxy_positive_count": tp,
        "gate_only_positive_count": fp,
        "proxy_only_positive_count": fn,
        "gate_and_proxy_negative_count": tn,
        "gate_retrieve_rate": round(float(gate_arr.mean()), 4),
        "closed_book_error_proxy_rate": round(float(proxy_arr.mean()), 4),
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically without replacing an existing artifact."""
    try:
        raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis output is not strict JSON serializable") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(raw)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace existing analysis output: {path}") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    _require_path_argument(args.output, "--output")
    out_path = Path(args.output)
    if out_path.exists():
        raise FileExistsError(f"refusing to replace existing analysis output: {out_path}")
    full_run_supplied = args.full_run is not None
    closedbook_run_supplied = args.closedbook_run is not None
    if full_run_supplied != closedbook_run_supplied:
        raise ValueError("--full-run and --closedbook-run must be provided together")
    if full_run_supplied:
        _require_path_argument(args.full_run, "--full-run")
        _require_path_argument(args.closedbook_run, "--closedbook-run")

    np.random.seed(args.seed)

    if full_run_supplied:
        results, gate_decisions, proxy_decisions, inputs, coverage = _offline_analysis(
            Path(args.full_run),
            Path(args.closedbook_run),
            out_path,
        )
        mode = "offline"
    else:
        results, gate_decisions, proxy_decisions, inputs, coverage = _live_analysis(
            args,
            out_path,
        )
        mode = "live"

    metrics = _compute_metrics(gate_decisions, proxy_decisions)
    output = {
        "schema": GATING_ANALYSIS_SCHEMA,
        "status": "exploratory_proxy_not_retrieval_utility",
        "publication_safe": False,
        "limitation": (
            "closed-book exact-match error is a need-retrieval proxy only; it does "
            "not show that retrieval would improve the answer"
        ),
        "mode": mode,
        "input_sha256": inputs["sha256"],
        "inputs": inputs,
        "coverage": coverage,
        "metrics": metrics,
        "per_query": results,
    }
    _write_new_json(out_path, output)

    logger.info("Exploratory gating proxy analysis saved → %s", out_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
