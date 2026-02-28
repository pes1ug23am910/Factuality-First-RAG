#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_sample_experiment.sh
# Quick demo: chunk → build index → run pipeline (all mock-mode)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== Step 1: Chunk mock Wikipedia articles ==="
python -m factuality_rag.cli chunk_wiki \
    --output data/wiki_chunks.jsonl \
    --chunk-size 200 \
    --chunk-overlap 50 \
    --dev-sample-size 50 \
    --mock-mode

echo ""
echo "=== Step 2: Build FAISS + Pyserini indexes ==="
python -m factuality_rag.cli build_index \
    --corpus data/wiki_chunks.jsonl \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --faiss-out indexes/faiss.index \
    --pyserini-out indexes/pyserini_dir \
    --dev-sample-size 50 \
    --mock-mode

echo ""
echo "=== Step 3: Run pipeline (mock-mode) ==="
python -m factuality_rag.cli run \
    --query "What is the capital of France?" \
    --k 5 \
    --mock-mode

echo ""
echo "=== Done! ==="
