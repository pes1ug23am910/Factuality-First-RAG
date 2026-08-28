"""
factuality_rag.cli.__main__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Console entry-point for the Factuality-first RAG toolkit.

Usage::

    python -m factuality_rag.cli --help
    python -m factuality_rag.cli build_index --help
    python -m factuality_rag.cli run --query "..." --mock-mode
    python -m factuality_rag.cli chunk_wiki --help

Example::

    python -m factuality_rag.cli build_index \\
        --corpus data/wiki_chunks.jsonl \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --faiss-out indexes/faiss.index \\
        --pyserini-out indexes/pyserini_dir \\
        --dev-sample-size 50 --dry-run --mock-mode
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

logger = logging.getLogger("factuality_rag")


def _add_build_index_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``build_index`` sub-command."""
    p = subparsers.add_parser(
        "build_index",
        help="Build FAISS & Pyserini indexes from a JSONL corpus.",
    )
    p.add_argument("--corpus", required=True, help="Path to chunked JSONL corpus.")
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace embedding model name.",
    )
    p.add_argument("--faiss-out", default="indexes/faiss.index", help="FAISS index output path.")
    p.add_argument(
        "--pyserini-out", default="indexes/pyserini_dir", help="Pyserini collection output dir."
    )
    p.add_argument("--faiss-type", default="hnsw_flat", choices=["hnsw_flat", "ivfpq"])
    p.add_argument("--dev-sample-size", type=int, default=None, help="Limit docs for dev runs.")
    p.add_argument("--dry-run", action="store_true", help="Print plan without writing files.")
    p.add_argument(
        "--mock-mode", action="store_true", help="Use random embeddings; skip downloads."
    )
    p.set_defaults(func=_cmd_build_index)


def _add_chunk_wiki_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``chunk_wiki`` sub-command."""
    p = subparsers.add_parser(
        "chunk_wiki",
        help="Generate explicitly marked synthetic chunks in --mock-mode.",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Reserved for a future real dump parser; currently fails closed.",
    )
    p.add_argument("--output", default="data/wiki_chunks.jsonl", help="Output JSONL path.")
    p.add_argument("--chunk-size", type=int, default=200, help="Tokens per chunk.")
    p.add_argument("--chunk-overlap", type=int, default=50, help="Overlap tokens.")
    p.add_argument("--dev-sample-size", type=int, default=None, help="Limit articles.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock-mode", action="store_true")
    p.set_defaults(func=_cmd_chunk_wiki)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run`` sub-command."""
    p = subparsers.add_parser("run", help="Run the full pipeline on a query.")
    p.add_argument("--query", required=True, help="Input query.")
    p.add_argument("--k", type=int, default=10, help="Number of passages to retrieve.")
    p.add_argument("--no-gate", action="store_true", help="Disable gating probe.")
    p.add_argument("--score-threshold", type=float, default=0.4)
    p.add_argument(
        "--config",
        default=None,
        help="Explicit YAML config path (omitted: packaged exp_sample.yaml).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mock-mode", action="store_true")
    p.set_defaults(func=_cmd_run)


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``evaluate`` sub-command."""
    p = subparsers.add_parser("evaluate", help="Evaluate predictions JSONL.")
    p.add_argument("--predictions", required=True, help="Path to predictions JSONL.")
    p.add_argument("--references", default=None, help="Path to references (one per line).")
    p.add_argument(
        "--support-metric",
        choices=["none", "lexical"],
        default="none",
        help="Optional lexical support proxy; NLI FactScore is not available here.",
    )
    p.set_defaults(func=_cmd_evaluate)


# ── Command handlers ─────────────────────────────────────────


def _cmd_build_index(args: argparse.Namespace) -> None:
    """Handle ``build_index`` command."""
    from factuality_rag.index.builder import build_faiss_index, build_pyserini_index

    if args.dry_run:
        logger.info(
            "[DRY RUN] Would build FAISS index from '%s' → '%s'",
            args.corpus,
            args.faiss_out,
        )
        logger.info(
            "[DRY RUN] Would build Pyserini Lucene index → '%s'",
            args.pyserini_out,
        )
        return

    faiss_path = build_faiss_index(
        jsonl_path=args.corpus,
        embed_model=args.embedding_model,
        out_path=args.faiss_out,
        mock_mode=args.mock_mode,
        faiss_type=args.faiss_type,
        dev_sample_size=args.dev_sample_size,
    )
    logger.info("FAISS index saved: %s", faiss_path)

    # FAISS writes the validated, ID-canonical corpus beside the index.  Build
    # Lucene from that exact sidecar so dense and sparse results share one
    # document identity/order contract, including development sampling.
    canonical_corpus = str(Path(faiss_path).with_suffix(".jsonl"))
    pyserini_path = build_pyserini_index(
        jsonl_path=canonical_corpus,
        out_dir=args.pyserini_out,
    )
    logger.info("Pyserini Lucene index saved: %s", pyserini_path)


def _cmd_chunk_wiki(args: argparse.Namespace) -> None:
    """Handle ``chunk_wiki`` command."""
    from factuality_rag.data.wikipedia import WikiChunker

    chunker = WikiChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dry_run=args.dry_run,
        mock_mode=args.mock_mode,
        dev_sample_size=args.dev_sample_size,
    )

    if not args.mock_mode:
        raise RuntimeError(
            "chunk_wiki has no real dump parser; use scripts/build_corpus.py for the "
            "current HuggingFace acquisition path, or pass --mock-mode for synthetic data"
        )
    if args.input is not None:
        raise ValueError(
            "--input cannot be combined with --mock-mode; synthetic input is generated"
        )

    n = args.dev_sample_size or 20
    articles = chunker.generate_mock_articles(n)
    logger.info("Using %d explicitly synthetic mock articles.", len(articles))

    chunks = chunker.process_articles(articles, output_path=args.output)
    logger.info("Generated %d chunks → %s", len(chunks), args.output)


def _cmd_run(args: argparse.Namespace) -> None:
    """Handle ``run`` command."""
    from factuality_rag.pipeline.orchestrator import Pipeline

    pipe = Pipeline(
        config_path=args.config,
        mock_mode=args.mock_mode,
        seed=args.seed,
    )

    answer, trusted, provenance, confidence = pipe.run(
        args.query,
        k=args.k,
        gate=not args.no_gate,
        score_threshold=args.score_threshold,
    )

    print(f"\n{'=' * 60}")
    print(f"Query:       {args.query}")
    print(f"Answer:      {answer}")
    print(f"Confidence:  {confidence}")
    print(f"Trusted:     {len(trusted)} passage(s)")
    print(f"Provenance:  {provenance}")
    print(f"{'=' * 60}\n")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Handle ``evaluate`` command."""
    import json

    from factuality_rag.eval.metrics import evaluate_predictions

    preds_path = Path(args.predictions)
    if not preds_path.exists():
        logger.error("Predictions file not found: %s", preds_path)
        sys.exit(1)

    with open(preds_path, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    references = None
    if args.references:
        ref_path = Path(args.references)
        if not ref_path.is_file():
            logger.error("References file not found: %s", ref_path)
            sys.exit(1)
        with open(ref_path, encoding="utf-8") as f:
            references = [line.strip() for line in f]

    metrics = evaluate_predictions(
        predictions,
        references,
        support_metric=args.support_metric,
    )
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        prog="factuality-rag",
        description="Factuality-first RAG: adaptive retrieval gating + passage-level scoring.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    _add_build_index_parser(subparsers)
    _add_chunk_wiki_parser(subparsers)
    _add_run_parser(subparsers)
    _add_evaluate_parser(subparsers)

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
