#!/usr/bin/env python
"""
scripts/aggregate_results.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Aggregate metrics from multiple experiment runs across seeds.

Computes mean ± std for each metric and produces a summary table.

Usage::

    python scripts/aggregate_results.py \\
        --runs-dir runs \\
        --configs B1 B2 B3 B4 full \\
        --output analysis/aggregated_results.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import math
import numbers
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRICS_SCHEMA = "factuality-rag.metrics.v2"
_METADATA_SCHEMA = "factuality-rag.run-metadata.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASE_METRICS_FIELDS = {
    "answer_coverage",
    "answered_count",
    "exact_match",
    "f1",
    "manifest_sha256",
    "n_predictions",
    "retrieval_count",
    "retrieval_rate",
    "run_id",
    "run_manifest_sha256",
    "schema",
    "support_metric",
}
_LEXICAL_METRICS_FIELDS = {
    "lexical_support_answered_count",
    "lexical_support_answered_only",
}
_PUBLICATION_STATUS = "structurally_validated_but_not_claim_safe"
_PUBLICATION_REASON = "runtime_immutable_model_revision_binding_is_unestablished"
_EXPLORATORY_STATUS = "exploratory_not_claim_safe"
_EXPLORATORY_REASON = "exploratory_mode_accepts_unsealed_inputs"


def _reject_duplicate_json_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _require_digest(value: Any, label: str, run_dir: Path) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise ValueError(f"{run_dir} {label} must be a non-zero lowercase SHA-256 digest")
    return value


def _finite_metric(metrics: Mapping[str, Any], key: str, run_dir: Path) -> float:
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{run_dir} metric {key!r} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{run_dir} metric {key!r} must be finite")
    return numeric


def _rate_metric(metrics: Mapping[str, Any], key: str, run_dir: Path) -> float:
    value = _finite_metric(metrics, key, run_dir)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{run_dir} metric {key!r} must be in the closed interval [0, 1]")
    return value


def _count_metric(
    metrics: Mapping[str, Any], key: str, run_dir: Path, *, positive: bool = False
) -> int:
    value = _finite_metric(metrics, key, run_dir)
    if value < 0.0 or not value.is_integer() or (positive and value == 0.0):
        qualifier = "a positive" if positive else "a non-negative"
        raise ValueError(f"{run_dir} metric {key!r} must be {qualifier} integer-valued number")
    return int(value)


def _validate_publication_metrics(metrics: Mapping[str, Any], run_dir: Path) -> None:
    support_metric = metrics.get("support_metric")
    if support_metric not in {"none", "lexical"}:
        raise ValueError(f"{run_dir} has an unsupported or unpinned support metric")
    expected_fields = set(_BASE_METRICS_FIELDS)
    if support_metric == "lexical":
        expected_fields.update(_LEXICAL_METRICS_FIELDS)
    if set(metrics) != expected_fields:
        missing = sorted(expected_fields - set(metrics))
        extra = sorted(set(metrics) - expected_fields)
        raise ValueError(
            f"{run_dir} sealed metrics fields do not match the exact {support_metric!r} schema "
            f"(missing={missing}, extra={extra})"
        )

    run_id = metrics.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
        raise ValueError(f"{run_dir} metrics run_id must be a non-empty, unpadded string")
    _require_digest(metrics.get("run_manifest_sha256"), "run_manifest_sha256", run_dir)
    _require_digest(metrics.get("manifest_sha256"), "manifest_sha256", run_dir)

    n_predictions = _count_metric(metrics, "n_predictions", run_dir, positive=True)
    answered_count = _count_metric(metrics, "answered_count", run_dir)
    retrieval_count = _count_metric(metrics, "retrieval_count", run_dir)
    if answered_count > n_predictions or retrieval_count > n_predictions:
        raise ValueError(f"{run_dir} metric counts cannot exceed n_predictions")

    _rate_metric(metrics, "exact_match", run_dir)
    _rate_metric(metrics, "f1", run_dir)
    answer_coverage = _rate_metric(metrics, "answer_coverage", run_dir)
    retrieval_rate = _rate_metric(metrics, "retrieval_rate", run_dir)
    if not math.isclose(
        answer_coverage,
        answered_count / n_predictions,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{run_dir} answer_coverage does not equal answered_count/n_predictions")
    if not math.isclose(
        retrieval_rate,
        retrieval_count / n_predictions,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{run_dir} retrieval_rate does not equal retrieval_count/n_predictions")

    if support_metric == "lexical":
        lexical_count = _count_metric(metrics, "lexical_support_answered_count", run_dir)
        _rate_metric(metrics, "lexical_support_answered_only", run_dir)
        if lexical_count != answered_count:
            raise ValueError(
                f"{run_dir} lexical_support_answered_count does not equal answered_count"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment results.")
    p.add_argument("--runs-dir", type=str, default="runs", help="Base runs directory.")
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional exact config-path or config-stem allow-list.",
    )
    p.add_argument(
        "--exploratory",
        action="store_true",
        help="Allow legacy/unsealed runs; output is stamped publication_safe=false.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern to match run directory names (e.g. 'full_nq_500_s*').",
    )
    p.add_argument("--output", type=str, default="analysis/aggregated_results.json")
    return p.parse_args()


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, encoding="utf-8") as f:
        payload = json.load(
            f,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return cast(Dict[str, Any], payload)


def _reject_or_remove_legacy_factscore(
    metrics: Dict[str, Any], *, exploratory: bool, run_dir: Path
) -> None:
    legacy_keys = [key for key in metrics if "factscore" in key.lower()]
    if not legacy_keys:
        return
    if not exploratory:
        raise ValueError(
            f"{run_dir} contains legacy/unverified FactScore fields {legacy_keys}; "
            "publication aggregation refuses them"
        )
    logger.warning("Ignoring legacy/unverified FactScore fields in %s: %s", run_dir, legacy_keys)
    for key in legacy_keys:
        metrics.pop(key, None)


def load_run_metrics(run_dir: Path, *, exploratory: bool = False) -> Dict[str, Any]:
    """Load metrics.json from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict of metric name → value.
    """
    metrics = _load_json_object(run_dir / "metrics.json", "metrics artifact")
    _reject_or_remove_legacy_factscore(metrics, exploratory=exploratory, run_dir=run_dir)
    if exploratory:
        return metrics

    from factuality_rag.reproducibility import verify_manifest

    if metrics.get("schema") != _METRICS_SCHEMA or not verify_manifest(metrics):
        raise ValueError(f"{run_dir} metrics.json is not a valid sealed {_METRICS_SCHEMA} artifact")
    _validate_publication_metrics(metrics, run_dir)
    return metrics


def load_run_metadata(run_dir: Path, *, exploratory: bool = False) -> Dict[str, Any]:
    """Load metadata.json from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Metadata dict.
    """
    meta_path = run_dir / "metadata.json"
    if exploratory and not meta_path.is_file():
        return {}
    metadata = _load_json_object(meta_path, "run metadata")
    if exploratory:
        return metadata

    from factuality_rag.reproducibility import verify_manifest

    if metadata.get("schema") != _METADATA_SCHEMA or not verify_manifest(metadata):
        raise ValueError(
            f"{run_dir} metadata.json is not a valid sealed {_METADATA_SCHEMA} artifact"
        )
    required = {
        "schema",
        "run_id",
        "git_commit",
        "git_dirty",
        "config_path",
        "config_identity",
        "config_source_sha256",
        "config_sha256",
        "mock_mode",
        "support_metric",
        "publication_artifact",
        "manifest_sha256",
    }
    if not required <= set(metadata):
        raise ValueError(f"{run_dir} sealed metadata artifact is missing identity fields")
    identity = metadata.get("config_identity")
    if (
        metadata.get("config_path") != identity
        or not isinstance(identity, str)
        or not identity.startswith(("package://", "external-config://"))
    ):
        raise ValueError(f"{run_dir} metadata has an unsafe config identity")
    return metadata


def _load_publication_manifest(run_dir: Path) -> Dict[str, Any]:
    from factuality_rag.reproducibility import (
        sha256_json,
        validate_publication_run_manifest,
    )

    manifest = _load_json_object(run_dir / "run_manifest.json", "run manifest")
    manifest = validate_publication_run_manifest(
        manifest,
        expected_run_id=run_dir.name,
        artifact_root=run_dir,
    )
    git = manifest.get("git")
    config = manifest.get("config")
    if (
        not isinstance(git, Mapping)
        or set(git) != {"commit", "dirty"}
        or not isinstance(config, Mapping)
        or manifest.get("config_sha256") != sha256_json(config)
        or not isinstance(manifest.get("config_file_sha256"), str)
        or _SHA256_RE.fullmatch(cast(str, manifest.get("config_file_sha256"))) is None
    ):
        raise ValueError(f"{run_dir} run manifest has invalid source/config identity")
    return cast(Dict[str, Any], manifest)


def _publication_signature(manifest: Mapping[str, Any], support_metric: str) -> str:
    """Hash fields that must match when combining runs across seeds."""
    from factuality_rag.reproducibility import sha256_json

    config = deepcopy(manifest.get("config"))
    if not isinstance(config, dict):
        raise ValueError("sealed run manifest config must be an object")
    config.pop("seed", None)
    signature = {
        "run_kind": manifest.get("run_kind"),
        "git": manifest.get("git"),
        "config_without_seed": config,
        "data": manifest.get("data"),
        "retrieval": manifest.get("retrieval"),
        "evaluator": manifest.get("evaluator"),
        "model_revisions": manifest.get("model_revisions"),
        "mock": manifest.get("mock"),
        "support_metric": support_metric,
    }
    return cast(str, sha256_json(signature))


def load_run_bundle(
    run_dir: Path, *, exploratory: bool
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    metrics = load_run_metrics(run_dir, exploratory=exploratory)
    metadata = load_run_metadata(run_dir, exploratory=exploratory)
    if exploratory:
        return metrics, metadata, None

    manifest = _load_publication_manifest(run_dir)
    manifest_hash = manifest.get("manifest_sha256")
    if metrics.get("run_id") != run_dir.name or metadata.get("run_id") != run_dir.name:
        raise ValueError(f"{run_dir} contains cross-run metrics or metadata")
    if metrics.get("run_manifest_sha256") != manifest_hash:
        raise ValueError(f"{run_dir} metrics are not bound to its sealed run manifest")
    data = manifest.get("data")
    if not isinstance(data, Mapping) or _count_metric(
        metrics, "n_predictions", run_dir, positive=True
    ) != data.get("selected_example_count"):
        raise ValueError(f"{run_dir} metrics count is not bound to the selected result population")
    if metrics.get("support_metric") != metadata.get("support_metric"):
        raise ValueError(f"{run_dir} metric identity disagrees with sealed metadata")
    if metadata.get("mock_mode") is not False:
        raise ValueError(
            f"{run_dir} metadata is mock or does not explicitly declare mock_mode=false"
        )
    if metadata.get("publication_artifact") is not True:
        raise ValueError(f"{run_dir} metadata does not declare publication_artifact=true")
    git = manifest.get("git")
    if not isinstance(git, Mapping):
        raise ValueError(f"{run_dir} run manifest has no Git identity")
    if (
        metadata.get("git_commit") != git.get("commit")
        or metadata.get("git_dirty") != git.get("dirty")
        or metadata.get("config_sha256") != manifest.get("config_sha256")
        or metadata.get("config_source_sha256") != manifest.get("config_file_sha256")
    ):
        raise ValueError(f"{run_dir} metadata is not bound to the sealed source/config identity")
    return metrics, metadata, manifest


def _config_identity(metadata: Mapping[str, Any], manifest: Optional[Mapping[str, Any]]) -> str:
    raw = manifest.get("config_path") if manifest is not None else metadata.get("config_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("each run must declare a non-empty config_path")
    return cast(str, raw)


def _config_allowed(config_path: str, allow_list: Optional[List[str]]) -> bool:
    if allow_list is None:
        return True
    accepted = {item.casefold() for item in allow_list}
    return config_path.casefold() in accepted or Path(config_path).stem.casefold() in accepted


def _numeric_metric(metrics: Mapping[str, Any], key: str, run_dir: Path) -> float:
    return _finite_metric(metrics, key, run_dir)


def _result_input_binding(
    run_dir: Path,
    *,
    group_identity: str,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    manifest: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Bind an aggregate row to exact input bytes without emitting local paths."""
    from factuality_rag.reproducibility import sha256_file

    binding: Dict[str, Any] = {
        "group_identity": group_identity,
        "run_id": run_dir.name,
        "metrics_file_sha256": sha256_file(run_dir / "metrics.json"),
    }
    metrics_seal = metrics.get("manifest_sha256")
    if isinstance(metrics_seal, str) and _SHA256_RE.fullmatch(metrics_seal):
        binding["metrics_manifest_sha256"] = metrics_seal
    metadata_path = run_dir / "metadata.json"
    if metadata_path.is_file():
        binding["metadata_file_sha256"] = sha256_file(metadata_path)
    metadata_seal = metadata.get("manifest_sha256")
    if isinstance(metadata_seal, str) and _SHA256_RE.fullmatch(metadata_seal):
        binding["metadata_manifest_sha256"] = metadata_seal
    if manifest is not None:
        binding.update(
            {
                "run_manifest_file_sha256": sha256_file(run_dir / "run_manifest.json"),
                "run_manifest_sha256": manifest["manifest_sha256"],
            }
        )
    return binding


def main() -> None:
    args = parse_args()
    runs_base = Path(args.runs_dir)

    if not runs_base.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_base}")

    # Discover run directories — filter by --pattern if provided
    import fnmatch

    all_runs = sorted(d for d in runs_base.iterdir() if d.is_dir())
    if args.pattern:
        all_runs = [d for d in all_runs if fnmatch.fnmatch(d.name, args.pattern)]

    if not all_runs:
        raise ValueError("no run directories matched the requested selection")

    logger.info(
        "Found %d run directories in %s (pattern=%s)", len(all_runs), runs_base, args.pattern or "*"
    )

    # Group only exact config identities. Exploratory groups also encode mock and
    # support-metric identity so incompatible measurements are never averaged.
    grouped: Dict[str, List[Tuple[Path, Dict[str, Any], Optional[Dict[str, Any]]]]] = {}
    publication_signatures: Dict[str, str] = {}
    input_bindings: List[Dict[str, Any]] = []
    for run_dir in all_runs:
        metrics, metadata, manifest = load_run_bundle(run_dir, exploratory=args.exploratory)
        config_path = _config_identity(metadata, manifest)
        if not _config_allowed(config_path, args.configs):
            continue

        support_metric = metrics.get("support_metric", metadata.get("support_metric", "unknown"))
        if support_metric not in {"none", "lexical"}:
            if not args.exploratory:
                raise ValueError(f"{run_dir} has an unsupported support metric")
            support_metric = "unknown"
        mock_mode = metadata.get("mock_mode", "unknown")
        group_name = config_path
        if args.exploratory:
            group_name = f"{config_path}|support={support_metric}|mock={mock_mode}"
        else:
            assert manifest is not None
            signature = _publication_signature(manifest, cast(str, support_metric))
            prior = publication_signatures.setdefault(group_name, signature)
            if prior != signature:
                raise ValueError(
                    f"sealed runs for {config_path!r} differ in fields other than seed"
                )
        grouped.setdefault(group_name, []).append((run_dir, metrics, manifest))
        input_bindings.append(
            _result_input_binding(
                run_dir,
                group_identity=group_name,
                metrics=metrics,
                metadata=metadata,
                manifest=manifest,
            )
        )

    if not grouped:
        raise ValueError("no run directories matched the exact config allow-list")

    # Aggregate
    summary: Dict[str, Dict[str, Any]] = {}
    metric_keys = {
        "exact_match",
        "f1",
        "lexical_support_answered_only",
        "lexical_support_answered_count",
        "answer_coverage",
        "answered_count",
        "n_predictions",
        "retrieval_rate",
        "retrieval_count",
    }

    for config_name, run_bundles in grouped.items():
        if not args.exploratory:
            seeds = [manifest.get("seed") for _, _, manifest in run_bundles if manifest is not None]
            if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
                raise ValueError(f"group {config_name!r} has an invalid sealed seed")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"group {config_name!r} contains duplicate seeds")
        agg: Dict[str, Any] = {"n_runs": len(run_bundles)}
        all_keys: set[str] = set()
        for _, metrics, _ in run_bundles:
            all_keys.update(metrics)

        for key in sorted(all_keys & metric_keys):
            missing = [run_dir.name for run_dir, metrics, _ in run_bundles if key not in metrics]
            if missing:
                raise ValueError(
                    f"metric {key!r} is missing from runs in group {config_name!r}: {missing}"
                )
            values = [_numeric_metric(metrics, key, run_dir) for run_dir, metrics, _ in run_bundles]
            agg[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1 if len(values) > 1 else 0)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "values": [round(value, 4) for value in values],
            }

        summary[config_name] = agg

    # Print table
    print("\n" + "=" * 106)
    print(f"{'Config identity':<43} {'EM':>12} {'F1':>12} {'Lexical':>12} {'Ret%':>10} {'Runs':>6}")
    print("-" * 106)

    for name, s in summary.items():
        em = s.get("exact_match", {})
        f1 = s.get("f1", {})
        lexical = s.get("lexical_support_answered_only", {})
        rr = s.get("retrieval_rate", {})
        rr_str = f"{rr.get('mean', 0) * 100:.1f}%" if rr else "—"
        lexical_str = (
            f"{lexical.get('mean', 0):.4f}±{lexical.get('std', 0):.4f}" if lexical else "—"
        )
        print(
            f"{name:<43} "
            f"{em.get('mean', 0):.4f}±{em.get('std', 0):.4f}  "
            f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}  "
            f"{lexical_str:>12}  "
            f"{rr_str:>10}  "
            f"{s['n_runs']:>6}"
        )
    print("=" * 106 + "\n")

    # Save
    from factuality_rag.reproducibility import seal_manifest, write_immutable_json

    publication_status = _EXPLORATORY_STATUS if args.exploratory else _PUBLICATION_STATUS
    publication_reason = _EXPLORATORY_REASON if args.exploratory else _PUBLICATION_REASON
    output = seal_manifest(
        {
            "schema": "factuality-rag.aggregate.v3",
            "publication_safe": False,
            "publication_status": publication_status,
            "publication_safety_reason": publication_reason,
            "input_bindings": sorted(
                input_bindings,
                key=lambda item: (cast(str, item["group_identity"]), cast(str, item["run_id"])),
            ),
            "groups": summary,
        }
    )
    out_path = Path(args.output)
    write_immutable_json(out_path, output)

    logger.info("Aggregated results saved → %s", out_path)


if __name__ == "__main__":
    main()
