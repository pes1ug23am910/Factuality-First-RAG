#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_sample_experiment.sh
# Idempotent smoke: mock data/embeddings, real JDK-21 Lucene build/query, mock pipeline
# ──────────────────────────────────────────────────────────────
set -euo pipefail

: "${HOME:?HOME must be set to the JarvisLabs persistent home}"

smoke_root="${FACTUALITY_RAG_SMOKE_ROOT:-${HOME}/factuality-rag-runtime/smoke}"
if [[ "${smoke_root}" != /* || "${smoke_root}" == "/" || "${smoke_root}" == "${HOME}" ]]; then
    echo "FACTUALITY_RAG_SMOKE_ROOT must be an absolute child directory, not / or HOME" >&2
    exit 2
fi

data_root="${smoke_root}/data"
corpus_path="${data_root}/wiki_chunks.jsonl"
index_root="${smoke_root}/indexes"
faiss_path="${index_root}/faiss.index"
lucene_path="${index_root}/pyserini_dir"

indexes_complete() {
    local root="$1"
    local marker

    [[ -s "${root}/faiss.index" ]] || return 1
    [[ -s "${root}/faiss.ids.json" ]] || return 1
    [[ -s "${root}/faiss.jsonl" ]] || return 1
    [[ -d "${root}/pyserini_dir" ]] || return 1
    for marker in "${root}/pyserini_dir"/segments_*; do
        [[ -s "${marker}" ]] && return 0
    done
    return 1
}

mkdir -p "${smoke_root}"

echo "=== Step 1: Prepare mock Wikipedia chunks ==="
if [[ -s "${corpus_path}" ]]; then
    echo "Reusing ${corpus_path}"
elif [[ -e "${corpus_path}" ]]; then
    echo "Refusing to overwrite incomplete corpus: ${corpus_path}" >&2
    exit 1
else
    mkdir -p "${data_root}"
    corpus_build_root="$(mktemp -d "${smoke_root}/.corpus-build.XXXXXX")"
    if ! python -m factuality_rag.cli chunk_wiki \
        --output "${corpus_build_root}/wiki_chunks.jsonl" \
        --chunk-size 200 \
        --chunk-overlap 50 \
        --dev-sample-size 50 \
        --mock-mode; then
        echo "Corpus build failed; partial diagnostics remain at ${corpus_build_root}" >&2
        exit 1
    fi
    [[ -s "${corpus_build_root}/wiki_chunks.jsonl" ]] || {
        echo "Corpus build produced no non-empty JSONL file" >&2
        exit 1
    }
    mv -- "${corpus_build_root}/wiki_chunks.jsonl" "${corpus_path}"
    rmdir -- "${corpus_build_root}"
fi

echo ""
echo "=== Step 2: Build/reuse mock FAISS + real Pyserini index (requires JDK 21) ==="
if [[ -e "${index_root}" ]]; then
    if ! indexes_complete "${index_root}"; then
        echo "Refusing to overwrite incomplete index root: ${index_root}" >&2
        exit 1
    fi
    echo "Reusing complete indexes under ${index_root}"
else
    index_build_root="$(mktemp -d "${smoke_root}/.index-build.XXXXXX")"
    if ! python -m factuality_rag.cli build_index \
        --corpus "${corpus_path}" \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --faiss-out "${index_build_root}/faiss.index" \
        --pyserini-out "${index_build_root}/pyserini_dir" \
        --dev-sample-size 50 \
        --mock-mode; then
        echo "Index build failed; partial diagnostics remain at ${index_build_root}" >&2
        exit 1
    fi
    if ! indexes_complete "${index_build_root}"; then
        echo "Index build completed without the required FAISS/Lucene artifacts" >&2
        exit 1
    fi
    mv -- "${index_build_root}" "${index_root}"
fi

echo ""
echo "=== Step 3: Exercise the production Lucene BM25 query path ==="
python - "${faiss_path}" "${lucene_path}" <<'PY'
from __future__ import annotations

import math
import sys

from factuality_rag.retriever.hybrid import HybridRetriever

retriever = HybridRetriever(faiss_index_path=sys.argv[1], pyserini_index_path=sys.argv[2])
hits = retriever._bm25_search("mock article", 5)
if not hits:
    raise SystemExit("Lucene smoke query returned no hits")
if any(not doc_id or not math.isfinite(score) for doc_id, score in hits.items()):
    raise SystemExit("Lucene smoke query returned malformed hits")
print(f"Lucene smoke returned {len(hits)} hit(s): {', '.join(sorted(hits))}")
PY

echo ""
echo "=== Step 4: Run pipeline (mock-mode) ==="
python -m factuality_rag.cli run \
    --query "What does mock article 1 describe?" \
    --k 5 \
    --mock-mode

echo ""
echo "=== Done! ==="
