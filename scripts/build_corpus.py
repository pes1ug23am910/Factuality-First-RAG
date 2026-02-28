#!/usr/bin/env python
"""
scripts/build_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~
Phase 3A-1: Download the Wikipedia corpus via HuggingFace datasets,
build FAISS + Pyserini (Lucene) indexes.

Usage::

    python scripts/build_corpus.py \\
        --n-docs 100000 \\
        --faiss-out indexes/wiki100k.faiss \\
        --pyserini-out indexes/wiki100k_lucene \\
        --embed-model sentence-transformers/all-mpnet-base-v2

Requires GPU for embedding; index building is CPU-bound.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Wikipedia corpus indexes.")
    p.add_argument("--n-docs", type=int, default=100_000, help="Number of documents to index.")
    p.add_argument("--faiss-out", type=str, default="indexes/wiki100k.faiss", help="FAISS output path.")
    p.add_argument("--pyserini-out", type=str, default="indexes/wiki100k_lucene", help="Pyserini Lucene output dir.")
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--batch-size", type=int, default=256, help="Encoding batch size.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Download Wikipedia passages from HF
    logger.info("Loading Wikipedia passages from HuggingFace (n=%d) ...", args.n_docs)
    from factuality_rag.data.wikipedia import load_from_hf

    docs = load_from_hf(n_docs=args.n_docs, seed=args.seed)
    logger.info("Loaded %d documents.", len(docs))

    # 2. Build dense FAISS index
    logger.info("Building FAISS index → %s", args.faiss_out)
    import numpy as np

    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    model = SentenceTransformer(args.embed_model)
    texts = [d["text"] for d in docs]

    logger.info("Encoding %d documents (batch_size=%d) ...", len(texts), args.batch_size)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss_path = Path(args.faiss_out)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS index saved (%d vectors, dim=%d).", index.ntotal, dim)

    # Save document mapping (id → text)
    doc_map_path = faiss_path.with_suffix(".json")
    doc_map = {str(i): {"id": d.get("id", str(i)), "text": d["text"][:500]} for i, d in enumerate(docs)}
    with open(doc_map_path, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, ensure_ascii=False)
    logger.info("Document mapping saved → %s", doc_map_path)

    # 3. Build Lucene index for BM25 (Pyserini)
    logger.info("Building Pyserini Lucene index → %s", args.pyserini_out)
    lucene_dir = Path(args.pyserini_out)
    jsonl_dir = lucene_dir / "jsonl_input"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = jsonl_dir / "docs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            record = {"id": doc.get("id", str(i)), "contents": doc["text"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d JSONL records. Now run Pyserini indexing:", len(docs))
    logger.info(
        "  python -m pyserini.index.lucene "
        "--collection JsonCollection "
        "--input %s "
        "--index %s "
        "--generator DefaultLuceneDocumentGenerator "
        "--threads 4 "
        "--storePositions --storeDocvectors --storeRaw",
        jsonl_dir,
        lucene_dir,
    )

    logger.info("Done. Corpus build complete.")


if __name__ == "__main__":
    main()
