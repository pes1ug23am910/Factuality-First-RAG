#!/usr/bin/env python
"""Paired bootstrap significance testing for sealed EM/F1 predictions.

Records are paired by an exact, unique ``example_id``. The two artifacts must
contain the same IDs, inputs, and reference aliases. Publication mode also
requires sealed run and predictions manifests; ``--exploratory`` is the
explicit escape hatch for local, unsealed analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PREDICTIONS_MANIFEST_SCHEMA = "factuality-rag.predictions.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PUBLICATION_STATUS = "structurally_validated_but_not_claim_safe"
_PUBLICATION_REASON = "runtime_immutable_model_revision_binding_is_unestablished"
_EXPLORATORY_STATUS = "exploratory_not_claim_safe"
_EXPLORATORY_REASON = "exploratory_mode_accepts_unsealed_inputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap significance test.")
    p.add_argument("--system-a", type=str, required=True, help="predictions.jsonl for system A")
    p.add_argument("--system-b", type=str, required=True, help="predictions.jsonl for system B")
    p.add_argument(
        "--metric",
        type=str,
        default="exact_match",
        choices=["exact_match", "f1"],
        help="Reference-based metric to compare.",
    )
    p.add_argument("--n-bootstrap", type=int, default=10_000, help="Number of bootstrap samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--exploratory",
        action="store_true",
        help="Allow unsealed predictions; output is stamped publication_safe=false.",
    )
    p.add_argument("--output", type=str, default="analysis/bootstrap_test.json")
    return p.parse_args()


def _reject_duplicate_keys(pairs: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_key, value in pairs:
        if raw_key in result:
            raise ValueError(f"duplicate JSON key {raw_key!r}")
        result[raw_key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    """Load strict JSONL prediction objects."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"predictions artifact not found: {source}")
    predictions: List[Dict[str, Any]] = []
    with source.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonfinite,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid strict JSON at {source}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"prediction at {source}:{line_number} must be an object")
            predictions.append(cast(Dict[str, Any], record))
    if not predictions:
        raise ValueError(f"predictions artifact is empty: {source}")
    return predictions


def _index_predictions(
    predictions: Sequence[Mapping[str, Any]], label: str
) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(predictions):
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id != example_id.strip():
            raise ValueError(f"{label} prediction {position} has an invalid example_id")
        if example_id in indexed:
            raise ValueError(f"{label} contains duplicate example_id {example_id!r}")
        query = record.get("input")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{label} prediction {example_id!r} has an invalid input")
        answer = record.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"{label} prediction {example_id!r} has a non-string answer")
        indexed[example_id] = record
    return indexed


def pair_predictions(
    predictions_a: Sequence[Mapping[str, Any]],
    predictions_b: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Pair two prediction sequences by exact ID and validate query identity."""
    by_id_a = _index_predictions(predictions_a, "system A")
    by_id_b = _index_predictions(predictions_b, "system B")
    ids_a = set(by_id_a)
    ids_b = set(by_id_b)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)
        only_b = sorted(ids_b - ids_a)
        raise ValueError(
            f"prediction example_id sets differ (only A={only_a[:3]}, only B={only_b[:3]})"
        )
    pairs = []
    for example_id in sorted(ids_a):
        record_a = by_id_a[example_id]
        record_b = by_id_b[example_id]
        if record_a["input"] != record_b["input"]:
            raise ValueError(f"input mismatch for example_id {example_id!r}")
        pairs.append((example_id, record_a, record_b))
    return pairs


def _resolve_references(
    pairs: Sequence[Tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    run_dir_a: Path,
    run_dir_b: Path,
) -> List[List[str]]:
    from factuality_rag.eval.metrics import reference_aliases
    from factuality_rag.experiment_runner import load_reference_artifacts, resolve_record_reference

    by_id_a, by_query_a = load_reference_artifacts(run_dir_a)
    by_id_b, by_query_b = load_reference_artifacts(run_dir_b)
    resolved: List[List[str]] = []
    for example_id, record_a, record_b in pairs:
        reference_a = resolve_record_reference(record_a, by_id_a, by_query_a)
        reference_b = resolve_record_reference(record_b, by_id_b, by_query_b)
        if reference_a is None or reference_b is None:
            raise ValueError(f"missing reference for example_id {example_id!r}")
        aliases_a = reference_aliases(reference_a)
        aliases_b = reference_aliases(reference_b)
        if aliases_a != aliases_b:
            raise ValueError(f"reference mismatch for example_id {example_id!r}")
        resolved.append(aliases_a)
    return resolved


def compute_paired_scores(
    pairs: Sequence[Tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    references: Sequence[Sequence[str]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute alias-aware EM or token F1 for already paired records."""
    from factuality_rag.eval.metrics import compute_em_aliases, compute_f1_aliases

    if metric not in {"exact_match", "f1"}:
        raise ValueError("bootstrap metric must be 'exact_match' or 'f1'")
    if len(pairs) != len(references):
        raise ValueError("paired predictions and references must have the same length")
    metric_fn = compute_em_aliases if metric == "exact_match" else compute_f1_aliases
    # The explicit loop avoids any positional truncation and keeps each score
    # visibly bound to one validated pair and one exact alias list.
    values_a: List[float] = []
    values_b: List[float] = []
    for (_, record_a, record_b), reference in zip(pairs, references):
        values_a.append(metric_fn(cast(str, record_a["answer"]), reference))
        values_b.append(metric_fn(cast(str, record_b["answer"]), reference))
    return np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    with path.open(encoding="utf-8") as f:
        value = json.load(
            f,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Dict[str, Any], value)


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise ValueError(f"{label} must be a non-zero lowercase SHA-256 digest")
    return value


def _validate_and_bind_publication_predictions(
    path: Path, predictions: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from factuality_rag.eval.metrics import reference_aliases
    from factuality_rag.reproducibility import (
        sha256_file,
        sha256_ordered_ids,
        validate_publication_run_manifest,
        verify_manifest,
    )

    example_ids = [cast(str, record.get("example_id")) for record in predictions]
    _index_predictions(predictions, "publication predictions")

    run_manifest = _load_json_object(path.parent / "run_manifest.json", "run manifest")
    run_manifest = validate_publication_run_manifest(
        run_manifest,
        expected_run_id=path.parent.name,
        artifact_root=path.parent,
        selected_example_ids=example_ids,
    )

    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    sidecar = _load_json_object(sidecar_path, "predictions manifest")
    if sidecar.get("schema") != _PREDICTIONS_MANIFEST_SCHEMA or not verify_manifest(sidecar):
        raise ValueError(f"{sidecar_path} is not a sealed {_PREDICTIONS_MANIFEST_SCHEMA} artifact")
    expected_fields = {
        "schema",
        "run_id",
        "run_manifest_sha256",
        "predictions_sha256",
        "example_ids_sha256",
        "example_count",
        "mock",
        "manifest_sha256",
    }
    if set(sidecar) != expected_fields:
        raise ValueError(f"{sidecar_path} does not match the exact predictions schema")
    for field in (
        "run_manifest_sha256",
        "predictions_sha256",
        "example_ids_sha256",
        "manifest_sha256",
    ):
        _require_digest(sidecar.get(field), f"predictions manifest {field}")
    example_count = sidecar.get("example_count")
    if isinstance(example_count, bool) or not isinstance(example_count, int) or example_count <= 0:
        raise ValueError("predictions manifest example_count must be a positive integer")
    for record in predictions:
        if record.get("reference") is None:
            raise ValueError("publication predictions must seal an inline reference per example")
        reference_aliases(record["reference"])
    data = run_manifest.get("data")
    assert isinstance(data, Mapping)
    if (
        sidecar.get("run_id") != path.parent.name
        or sidecar.get("run_manifest_sha256") != run_manifest.get("manifest_sha256")
        or sidecar.get("predictions_sha256") != sha256_file(path)
        or sidecar.get("example_ids_sha256") != sha256_ordered_ids(example_ids)
        or sidecar.get("example_ids_sha256") != data.get("selected_example_ids_sha256")
        or sidecar.get("example_count") != len(example_ids)
        or data.get("selected_example_count") != len(example_ids)
        or sidecar.get("mock") is not False
    ):
        raise ValueError(f"{sidecar_path} does not bind the exact non-mock predictions artifact")
    binding = {
        "run_id": path.parent.name,
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "run_manifest_file_sha256": sha256_file(path.parent / "run_manifest.json"),
        "predictions_manifest_sha256": sidecar["manifest_sha256"],
        "predictions_manifest_file_sha256": sha256_file(sidecar_path),
        "predictions_sha256": sidecar["predictions_sha256"],
        "example_ids_sha256": sidecar["example_ids_sha256"],
        "example_count": example_count,
        "dataset_manifest_sha256": data["dataset_manifest_sha256"],
        "split_manifest_sha256": data["split_manifest_sha256"],
    }
    return run_manifest, binding


def _validate_publication_predictions(
    path: Path, predictions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Validate one bundle while preserving the historical manifest return API."""
    manifest, _ = _validate_and_bind_publication_predictions(path, predictions)
    return manifest


def _validate_publication_pair(
    path_a: Path,
    predictions_a: Sequence[Mapping[str, Any]],
    path_b: Path,
    predictions_b: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_a, binding_a = _validate_and_bind_publication_predictions(path_a, predictions_a)
    manifest_b, binding_b = _validate_and_bind_publication_predictions(path_b, predictions_b)
    data_a = manifest_a.get("data")
    data_b = manifest_b.get("data")
    assert isinstance(data_a, Mapping) and isinstance(data_b, Mapping)
    identity_fields = (
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "split_partition",
        "selected_example_ids_sha256",
        "selected_example_count",
    )
    if any(data_a.get(field) != data_b.get(field) for field in identity_fields):
        raise ValueError("publication bootstrap inputs do not bind the same dataset split and IDs")
    return binding_a, binding_b


def _exploratory_input_binding(
    path: Path, predictions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Hash an unsealed input without serializing its machine-local path."""
    from factuality_rag.reproducibility import sha256_file, sha256_ordered_ids

    ids = [cast(str, record["example_id"]) for record in predictions]
    return {
        "predictions_sha256": sha256_file(path),
        "example_ids_sha256": sha256_ordered_ids(ids),
        "example_count": len(ids),
    }


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, Tuple[float, float]]:
    """Test the one-sided alternative that system B outperforms system A."""
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if scores_a.ndim != 1 or scores_b.ndim != 1 or scores_a.size == 0:
        raise ValueError("scores must be non-empty one-dimensional arrays")
    if scores_a.shape != scores_b.shape:
        raise ValueError("scores must be exactly paired")
    if not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all():
        raise ValueError("scores must be finite")

    rng = np.random.RandomState(seed)
    n = len(scores_a)
    observed_delta = float(scores_b.mean() - scores_a.mean())
    deltas = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        deltas[i] = scores_b[indices].mean() - scores_a[indices].mean()

    # Add-one correction prevents an impossible exact zero Monte Carlo p-value.
    p_value = float((np.count_nonzero(deltas <= 0.0) + 1) / (n_bootstrap + 1))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return p_value, observed_delta, (ci_low, ci_high)


def main() -> None:
    args = parse_args()
    path_a = Path(args.system_a)
    path_b = Path(args.system_b)
    predictions_a = load_predictions(path_a)
    predictions_b = load_predictions(path_b)
    pairs = pair_predictions(predictions_a, predictions_b)
    references = _resolve_references(pairs, path_a.parent, path_b.parent)

    if args.exploratory:
        binding_a = _exploratory_input_binding(path_a, predictions_a)
        binding_b = _exploratory_input_binding(path_b, predictions_b)
        system_a = "unsealed-input-a"
        system_b = "unsealed-input-b"
        publication_status = _EXPLORATORY_STATUS
        publication_reason = _EXPLORATORY_REASON
    else:
        binding_a, binding_b = _validate_publication_pair(
            path_a, predictions_a, path_b, predictions_b
        )
        system_a = cast(str, binding_a["run_id"])
        system_b = cast(str, binding_b["run_id"])
        publication_status = _PUBLICATION_STATUS
        publication_reason = _PUBLICATION_REASON

    scores_a, scores_b = compute_paired_scores(pairs, references, args.metric)
    p_value, delta, (ci_low, ci_high) = paired_bootstrap_test(
        scores_a,
        scores_b,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    from factuality_rag.reproducibility import (
        seal_manifest,
        sha256_ordered_ids,
        write_immutable_json,
    )

    result = seal_manifest(
        {
            "schema": "factuality-rag.bootstrap.v3",
            "publication_safe": False,
            "publication_status": publication_status,
            "publication_safety_reason": publication_reason,
            "system_a": system_a,
            "system_b": system_b,
            "input_bindings": {"system_a": binding_a, "system_b": binding_b},
            "metric": args.metric,
            "alternative": "system_b_greater",
            "n_queries": len(scores_a),
            "paired_example_ids_sha256": sha256_ordered_ids([pair[0] for pair in pairs]),
            "mean_a": round(float(scores_a.mean()), 4),
            "mean_b": round(float(scores_b.mean()), 4),
            "delta": round(delta, 4),
            "p_value": round(p_value, 6),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        }
    )

    out_path = Path(args.output)
    write_immutable_json(out_path, result)

    logger.info("Bootstrap test saved → %s", out_path)
    logger.info(
        "Result: Δ=%.4f, p=%.6f %s",
        delta,
        p_value,
        "(significant at p<0.05)" if p_value < 0.05 else "(not significant)",
    )


if __name__ == "__main__":
    main()
