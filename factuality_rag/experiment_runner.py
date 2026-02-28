"""
factuality_rag.experiment_runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structured experiment execution with metadata tracking and
prediction persistence.

Every run is saved to ``runs/<run-id>/`` with:
    - ``predictions.jsonl`` – per-query results
    - ``metrics.json`` – aggregated evaluation metrics
    - ``metadata.json`` – full run metadata

Example::

    >>> import yaml
    >>> cfg = yaml.safe_load(open("configs/exp_sample.yaml"))  # doctest: +SKIP
    >>> results = run(cfg, mock_mode=True)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Return the short git commit hash, or a fallback.

    Returns:
        Short commit hash or ``"git-not-available"``.
    """
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "git-not-available"


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
        config_path: Path to the config file.
        extra: Additional metadata to merge.

    Returns:
        Metadata dict.

    Example::

        >>> m = build_metadata({"seed": 42})
        >>> "timestamp" in m and "git_commit" in m
        True
    """
    meta: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "config_path": config_path,
        "seed": config.get("seed", 42),
        "models": config.get("models", {}),
        "datasets": config.get("data", {}).get("datasets", []),
        "library_versions": _get_lib_versions(),
    }
    if extra:
        meta.update(extra)
    return meta


def run(
    config: Dict[str, Any],
    queries: Optional[List[str]] = None,
    config_path: str = "configs/exp_sample.yaml",
    mock_mode: bool = False,
    runs_dir: str = "runs",
) -> Dict[str, Any]:
    """Execute an experiment: run the pipeline on each query, save results.

    Args:
        config: Parsed YAML config dict.
        queries: List of query strings. If ``None``, a small demo
                 set is used.
        config_path: Path to the YAML config (for metadata).
        mock_mode: Run all components in mock-mode.
        runs_dir: Base directory for run outputs.

    Returns:
        Dict with ``run_id``, ``predictions``, ``metrics``, and
        ``metadata``.

    Example::

        >>> result = run({"seed": 42}, queries=["test?"], mock_mode=True)
        >>> result["run_id"]  # doctest: +ELLIPSIS
        '...'
    """
    from factuality_rag.eval.metrics import evaluate_predictions
    from factuality_rag.pipeline.orchestrator import Pipeline

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What is photosynthesis?",
        ]

    seed = config.get("seed", 42)
    score_threshold = config.get("scorer", {}).get("score_threshold", 0.4)
    gate = config.get("gating", {}).get("enabled", True)
    k = config.get("retriever", {}).get("top_k", 10)

    # Build pipeline ONCE, reuse for all queries
    pipe = Pipeline(
        config_path=config_path,
        mock_mode=mock_mode,
        seed=seed,
    )

    predictions: List[Dict[str, Any]] = []
    pred_file = run_dir / "predictions.jsonl"

    with open(pred_file, "w", encoding="utf-8") as f:
        for query in queries:
            answer, trusted, provenance, confidence = pipe.run(
                query,
                k=k,
                gate=gate,
                score_threshold=score_threshold,
                seed=seed,
            )
            record = {
                "input": query,
                "answer": answer,
                "trusted_passages": trusted,
                "provenance": provenance,
                "confidence_tag": confidence,
                "run_metadata": {
                    "run_id": run_id,
                    "seed": seed,
                    "mock_mode": mock_mode,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            predictions.append(record)

    logger.info("Saved %d predictions → %s", len(predictions), pred_file)

    # Evaluate
    metrics = evaluate_predictions(predictions)

    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Metadata
    metadata = build_metadata(config, config_path)
    meta_file = run_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Run '%s' complete → %s", run_id, run_dir)

    return {
        "run_id": run_id,
        "predictions": predictions,
        "metrics": metrics,
        "metadata": metadata,
        "run_dir": str(run_dir),
    }
