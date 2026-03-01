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

import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    references: Optional[List[str]] = None,
    config_path: str = "configs/exp_sample.yaml",
    mock_mode: bool = False,
    runs_dir: str = "runs",
    run_id_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an experiment: run the pipeline on each query, save results.

    Args:
        config: Parsed YAML config dict.
        queries: List of query strings. If ``None``, a small demo
                 set is used.
        references: Optional parallel list of gold reference answers
                    for EM/F1 computation.
        config_path: Path to the YAML config (for metadata).
        mock_mode: Run all components in mock-mode.
        runs_dir: Base directory for run outputs.
        run_id_prefix: Optional prefix for the run directory name.
                       If set, run dir is ``<prefix>_<timestamp>``.

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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if run_id_prefix:
        run_id = f"{run_id_prefix}_{timestamp}"
    else:
        run_id = timestamp + "_" + uuid.uuid4().hex[:8]

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
        config=config,
    )

    predictions: List[Dict[str, Any]] = []
    pred_file = run_dir / "predictions.jsonl"
    retrieval_count = 0

    with open(pred_file, "w", encoding="utf-8") as f:
        for idx, query in enumerate(queries):
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
            if retrieval_triggered:
                retrieval_count += 1

            record: Dict[str, Any] = {
                "input": query,
                "answer": answer,
                "trusted_passages": trusted,
                "provenance": provenance,
                "confidence_tag": confidence,
                "retrieval_triggered": retrieval_triggered,
                "run_metadata": {
                    "run_id": run_id,
                    "seed": seed,
                    "mock_mode": mock_mode,
                },
            }
            if references and idx < len(references):
                record["reference"] = references[idx]

            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            predictions.append(record)

            if (idx + 1) % 50 == 0:
                logger.info("Progress: %d / %d queries", idx + 1, len(queries))

    logger.info("Saved %d predictions → %s", len(predictions), pred_file)

    # Evaluate
    ref_list = references if references and len(references) == len(predictions) else None
    metrics = evaluate_predictions(predictions, references=ref_list)

    # Add retrieval rate
    metrics["retrieval_rate"] = retrieval_count / len(queries) if queries else 0.0
    metrics["retrieval_count"] = float(retrieval_count)

    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save references separately for analysis scripts
    if references:
        refs_file = run_dir / "references.json"
        ref_map = {q: r for q, r in zip(queries, references)}
        with open(refs_file, "w", encoding="utf-8") as f:
            json.dump(ref_map, f, indent=2, ensure_ascii=False)

    # Metadata
    metadata = build_metadata(config, config_path, extra={
        "run_id": run_id,
        "n_queries": len(queries),
        "has_references": references is not None,
        "retrieval_rate": metrics["retrieval_rate"],
    })
    meta_file = run_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Run '%s' complete → %s", run_id, run_dir)
    logger.info(
        "Metrics: EM=%.4f  F1=%.4f  FactScore=%.4f  Ret%%=%.1f%%",
        metrics.get("exact_match", 0),
        metrics.get("f1", 0),
        metrics.get("factscore", 0),
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


def _extract_queries_and_references(
    dataset_name: str,
    split: str = "validation",
    sample: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Load queries and reference answers from a HuggingFace dataset.

    Supports NQ-Open, HotpotQA, TruthfulQA, and FEVER.

    Args:
        dataset_name: Dataset identifier (e.g. ``"natural_questions"``).
        split: Dataset split.
        sample: Number of examples to sample (``None`` = all).
        seed: Random seed for sampling.

    Returns:
        ``(queries, references)`` — parallel lists of question
        strings and gold answer strings.
    """
    from factuality_rag.data.loader import load_dataset

    ds = load_dataset(dataset_name, split=split, dev_sample_size=sample)

    queries: List[str] = []
    references: List[str] = []

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
        len(queries), dataset_name, split, sample,
    )
    return queries, references


def _extract_reference(row: Dict[str, Any], dataset_name: str) -> str:
    """Extract a single reference answer string from a dataset row.

    Args:
        row: A single row from the HF dataset.
        dataset_name: Name of the dataset.

    Returns:
        Reference answer string.
    """
    dn = dataset_name.lower()

    # ── PopQA: possible_answers is a list ─────────────────────
    if "popqa" in dn:
        pa = row.get("possible_answers", [])
        if isinstance(pa, list) and pa:
            return pa[0]
        obj = row.get("obj", "")
        return str(obj) if obj else ""

    # ── HAGRID: answers list with attributable text ───────────
    if "hagrid" in dn:
        answers = row.get("answers", [])
        if isinstance(answers, list) and answers:
            first = answers[0]
            if isinstance(first, dict):
                return str(first.get("answer", ""))
            return str(first)
        return ""

    # ── 2WikiMultiHopQA: answer field is a string ────────────
    if "2wiki" in dn or "wikimultihop" in dn:
        return str(row.get("answer", ""))

    # ── FEVER: label as reference ────────────────────────────
    if "fever" in dn:
        return str(row.get("label", ""))

    # ── NQ-Open: answer is a list of strings ─────────────────
    answer = row.get("answer", "")
    if isinstance(answer, list):
        return answer[0] if answer else ""

    if isinstance(answer, dict):
        val = answer.get("value", "")
        if val:
            return str(val)
        for key in ("text", "normalized_aliases"):
            v = answer.get(key, "")
            if v:
                return str(v) if not isinstance(v, list) else v[0]
        return str(answer)

    # ── HotpotQA: answer field is a string ───────────────────
    if "hotpot" in dn:
        return str(row.get("answer", ""))

    # ── TruthfulQA: best_answer or correct_answers ───────────
    if "truthful" in dn:
        best = row.get("best_answer", "")
        if best:
            return str(best)
        correct = row.get("correct_answers", [])
        if correct:
            return correct[0] if isinstance(correct, list) else str(correct)

    return str(answer)


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
        "--config", type=str, default="configs/exp_full_pipeline.yaml",
        help="Path to experiment YAML config.",
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="HF dataset name (e.g. natural_questions, hotpot_qa, truthful_qa).",
    )
    p.add_argument("--split", type=str, default="validation", help="Dataset split.")
    p.add_argument("--sample", type=int, default=None, help="Number of queries to sample.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    p.add_argument("--run-id", type=str, default=None, help="Prefix for run directory name.")
    p.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="YAML dot-path overrides, e.g. scorer.score_threshold=0.3",
    )
    p.add_argument("--mock", action="store_true", help="Run in mock mode.")
    p.add_argument("--runs-dir", type=str, default="runs", help="Base directory for outputs.")
    return p.parse_args()


def main() -> None:
    """CLI entry point for experiment runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_args()

    import yaml

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config not found: %s — using defaults.", config_path)
        config = {}

    # Apply seed override
    if args.seed is not None:
        config["seed"] = args.seed

    # Apply dot-path overrides
    if args.override:
        _apply_overrides(config, args.override)

    # Set numpy/random seeds
    seed = config.get("seed", 42)
    np.random.seed(seed)

    # Load queries
    queries: Optional[List[str]] = None
    references: Optional[List[str]] = None

    if args.dataset:
        queries, references = _extract_queries_and_references(
            dataset_name=args.dataset,
            split=args.split,
            sample=args.sample,
            seed=seed,
        )
    elif args.sample:
        # Use demo queries repeated
        demo = [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What is photosynthesis?",
        ]
        queries = (demo * (args.sample // len(demo) + 1))[:args.sample]

    # Determine mock mode
    mock_mode = args.mock or config.get("pipeline", {}).get("mock_mode", False)

    # Run experiment
    result = run(
        config=config,
        queries=queries,
        references=references,
        config_path=str(args.config),
        mock_mode=mock_mode,
        runs_dir=args.runs_dir,
        run_id_prefix=args.run_id,
    )

    print(f"\n{'='*60}")
    print(f"Run ID:    {result['run_id']}")
    print(f"Run Dir:   {result['run_dir']}")
    print(f"Queries:   {len(result['predictions'])}")
    print(f"Metrics:")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
