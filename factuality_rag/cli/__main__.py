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
    p.add_argument("--mock-mode", action="store_true", help="Use random embeddings; skip downloads.")
    p.set_defaults(func=_cmd_build_index)


def _add_chunk_wiki_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``chunk_wiki`` sub-command."""
    p = subparsers.add_parser(
        "chunk_wiki",
        help="Chunk a Wikipedia dump (or mock articles) into JSONL.",
    )
    p.add_argument("--input", default=None, help="Path to Wikipedia XML dump (optional).")
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
    p.add_argument("--config", default="configs/exp_sample.yaml", help="YAML config path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mock-mode", action="store_true")
    p.set_defaults(func=_cmd_run)


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``evaluate`` sub-command."""
    p = subparsers.add_parser("evaluate", help="Evaluate predictions JSONL.")
    p.add_argument("--predictions", required=True, help="Path to predictions JSONL.")
    p.add_argument("--references", default=None, help="Path to references (one per line).")
    p.set_defaults(func=_cmd_evaluate)


# ── Command handlers ─────────────────────────────────────────


def _cmd_build_index(args: argparse.Namespace) -> None:
    """Handle ``build_index`` command."""
    from factuality_rag.index.builder import build_faiss_index, prepare_pyserini_collection

    if args.dry_run:
        logger.info(
            "[DRY RUN] Would build FAISS index from '%s' → '%s'",
            args.corpus,
            args.faiss_out,
        )
        logger.info(
            "[DRY RUN] Would prepare Pyserini collection → '%s'",
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

    pyserini_path = prepare_pyserini_collection(
        jsonl_path=args.corpus,
        out_dir=args.pyserini_out,
        dev_sample_size=args.dev_sample_size,
    )
    logger.info("Pyserini collection saved: %s", pyserini_path)


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

    # If no input dump provided, generate mock articles
    if args.input is None or args.mock_mode:
        n = args.dev_sample_size or 20
        articles = chunker.generate_mock_articles(n)
        logger.info("Using %d mock articles.", len(articles))
    else:
        # TODO: implement real Wikipedia XML dump parsing
        logger.error("Real Wikipedia dump parsing not yet implemented. Use --mock-mode.")
        sys.exit(1)

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

    print(f"\n{'='*60}")
    print(f"Query:       {args.query}")
    print(f"Answer:      {answer}")
    print(f"Confidence:  {confidence}")
    print(f"Trusted:     {len(trusted)} passage(s)")
    print(f"Provenance:  {provenance}")
    print(f"{'='*60}\n")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Handle ``evaluate`` command."""
    import json
    from pathlib import Path

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
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                references = [line.strip() for line in f]

    metrics = evaluate_predictions(predictions, references)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


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
