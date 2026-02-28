# Architecture — Factuality-First RAG

> Version 0.2.0 · February 2026 (Session 2)

---

## 1. System Overview

Factuality-First RAG is a 4-stage pipeline that prioritises **factual correctness** over blind retrieval. Unlike standard RAG ("always retrieve, then generate"), this system:

1. **Decides whether to retrieve** (gating) based on the generator's own uncertainty.
2. **Retrieves candidates** via hybrid dense + sparse search.
3. **Scores each passage for factuality** before the generator ever sees it.
4. **Generates conditioned only on trusted evidence**, with a confidence label attached.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                       │
└──────────────┬───────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐     confident      ┌─────────────────────┐
│   STAGE 1: GATING PROBE  │ ──────────────────▶ │  Direct Generation  │
│  (entropy + logit gap)   │                     │  confidence = high  │
└──────────────┬───────────┘                     └─────────────────────┘
               │ uncertain
               ▼
┌──────────────────────────┐
│  STAGE 2: HYBRID RETRIEVAL│
│  Dense (FAISS/HNSW)      │
│  + Sparse (BM25/Pyserini)│
│  → top-K candidates      │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│ STAGE 3: PASSAGE SCORING │
│  NLI entailment P(ent)   │
│  + token overlap F1      │
│  + retriever score norm  │
│  → final_score per psg   │
│  → filter by threshold   │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│ STAGE 4: GENERATION      │
│  Conditioned on trusted  │
│  passages only           │
│  → answer + provenance   │
│  → confidence tag        │
└──────────────────────────┘
```

---

## 2. Module Architecture

```
factuality_rag/
│
├── model_registry.py ─ Shared model layer (NEW)
│                       Singleton registry for loaded models & tokenizers
│
├── data/            ─── Data ingestion layer
│   ├── loader.py         HuggingFace dataset wrapper
│   └── wikipedia.py      Wiki chunker (JSONL streaming) + HF loading
│
├── index/           ─── Index construction layer
│   └── builder.py        FAISS (HNSW/IVFPQ) + Pyserini prep
│
├── retriever/       ─── Retrieval layer
│   └── hybrid.py         HybridRetriever (dense + BM25 fusion)
│
├── gating/          ─── Decision layer
│   └── probe.py          GatingProbe (entropy/logit-gap) + model registry
│
├── scorer/          ─── Evidence evaluation layer
│   └── passage.py        PassageScorer (NLI + overlap + ret)
│
├── generator/       ─── Generation layer
│   └── wrapper.py        Generator (Mistral [INST] template + model registry)
│
├── pipeline/        ─── Orchestration layer
│   └── orchestrator.py   Pipeline class + run_pipeline()
│
├── eval/            ─── Evaluation layer
│   └── metrics.py        EM, F1, FactScore (claim decomposition + NLI)
│
├── cli/             ─── User interface layer
│   └── __main__.py       CLI commands (argparse)
│
└── experiment_runner.py ─ Reproducibility layer
                          Metadata tracking, run persistence
```

---

## 3. Data Flow Diagram

```
Wikipedia Dump ──▶ WikiChunker ──▶ wiki_chunks.jsonl
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                  build_faiss_index  save_embeddings  prepare_pyserini
                        │               │               │
                        ▼               ▼               ▼
                  faiss.index      embeddings.npy   pyserini_dir/
                        │                               │
                        └───────────┬───────────────────┘
                                    ▼
                             HybridRetriever
                                    │
                              ┌─────┴─────┐
                              ▼           ▼
                         FAISS search  BM25 search
                              │           │
                              └─────┬─────┘
                                    ▼
                          Score normalisation
                      (min-max → [0,1] per query)
                                    │
                                    ▼
                    combined = α·dense + (1-α)·bm25
                                    │
                                    ▼
                          PassageScorer (NLI + overlap)
                                    │
                                    ▼
                     Filter: final_score ≥ threshold
                                    │
                                    ▼
                          Generator (trusted passages)
                                    │
                                    ▼
                          (answer, provenance, confidence)
```

---

## 4. Component Details

### 4.1 Gating Probe

**Purpose:** Avoid retrieval when the generator is already confident.

**Mechanism:**
1. Forward the prompt through the generator (single step — next token only).
2. Compute **entropy** $H = -\sum_i p_i \log p_i$ of the softmax distribution.
3. Compute **logit gap** $\Delta = \text{logit}_1 - \text{logit}_2$ (top-2 difference).
4. Decision: `retrieve = (H > entropy_thresh) OR (Δ < logit_gap_thresh)`.

**Temperature calibration:** Grid search over $T \in [0.5, 3.0]$ to minimise ECE on dev set.

**Cost:** One forward pass (no autoregressive decoding). For a 7B model, this is ~50ms on A100.

```
Input prompt
     │
     ▼
┌─────────────┐
│ Forward pass │──▶ logits (vocab_size,)
└─────────────┘         │
                   ┌────┴────┐
                   ▼         ▼
            entropy(H)   logit_gap(Δ)
                   │         │
                   └────┬────┘
                        ▼
               H > θ_H  OR  Δ < θ_Δ ?
                  │              │
                 YES            NO
                  ▼              ▼
              RETRIEVE      SKIP (direct gen)
```

### 4.2 Hybrid Retriever

**Dense path:** Encode query with SentenceTransformer → search FAISS index → get (distance, idx) pairs.

**Sparse path:** BM25 via Pyserini `LuceneSearcher` (or mock scores in dev mode).

**Score normalisation (per query):**

$$\text{dense\_norm}_i = \frac{\text{dense}_i - \min(\text{dense})}{\max(\text{dense}) - \min(\text{dense})}$$

$$\text{bm25\_norm}_i = \frac{\text{bm25}_i - \min(\text{bm25})}{\max(\text{bm25}) - \min(\text{bm25})}$$

$$\text{combined}_i = \alpha \cdot \text{dense\_norm}_i + (1-\alpha) \cdot \text{bm25\_norm}_i$$

Default $\alpha = 0.6$ (tunable in config).

### 4.3 Passage Scorer

**Components:**

| Signal | Source | Range | Weight |
|--------|--------|-------|--------|
| NLI entailment | RoBERTa-MNLI `P(entailment \| query, passage)` | [0,1] | w_nli = 0.5 |
| Token overlap | F1 of query tokens vs passage tokens | [0,1] | w_overlap = 0.2 |
| Retriever score | Normalised `combined_score` from retriever | [0,1] | w_ret = 0.3 |

**Fusion:**

$$\text{final\_score} = w_{\text{nli}} \cdot P(\text{ent}) + w_{\text{overlap}} \cdot \text{overlap} + w_{\text{ret}} \cdot \text{ret\_norm}$$

**Filtering:** Passages with `final_score < threshold` (default 0.4) are dropped before generation.

### 4.4 Generator

Real integration with Mistral-7B-Instruct-v0.3 via model registry.

**Prompt template (RAG with context):**
```
<s>[INST] Answer the question using ONLY the provided context.
If the context does not support an answer, say "I cannot answer based on the provided context."

Context:
{context}

Question: {query} [/INST]
```

**Prompt template (no context):**
```
<s>[INST] Answer the following question concisely and accurately.

Question: {query} [/INST]
```

**Generation config:** `max_new_tokens=256`, `temperature=0.1`, `do_sample=False`.

**Model loading:** Uses `model_registry.get_model()` for singleton loading with optional 4-bit quantization via `BitsAndBytesConfig`. Accepts pre-loaded `model`/`tokenizer` kwargs for `Pipeline` reuse.

### 4.5 Model Registry

**Purpose:** Avoid loading 7B models multiple times across pipeline components.

```
model_registry.py
├── _models: Dict[str, Any]       ─ Module-level singleton
├── _tokenizers: Dict[str, Any]   ─ Module-level singleton
├── get_model(model_id, device, quantize_4bit) → model
├── get_tokenizer(model_id) → tokenizer
├── clear_registry() → None
└── is_loaded(model_id) → bool
```

**Quantization:** When `quantize_4bit=True`, loads with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=float16)`. Graceful fallback if `bitsandbytes` not installed.

**Sharing pattern:** `Pipeline.__init__()` loads the model once via registry. Both `Generator` and `GatingProbe` receive the same model/tokenizer instance.

### 4.6 Pipeline Orchestrator

**Two interfaces:**

1. **`run_pipeline()`** — functional interface, builds or accepts components:
```python
def run_pipeline(query, k=10, gate=True, score_threshold=0.4,
                 config_path="configs/exp_sample.yaml", seed=42,
                 mock_mode=False, *,
                 probe=None, retriever=None, scorer=None, generator=None)
    → (answer, trusted_passages, provenance, confidence_tag)
```

2. **`Pipeline` class** — loads all components once at init:
```python
pipe = Pipeline(config_path="configs/exp_sample.yaml", mock_mode=True)
result1 = pipe.run("What is DNA?")
result2 = pipe.run("Who discovered gravity?")  # reuses same models
```

**Confidence tag logic (updated):**

| Condition | Tag |
|-----------|-----|
| Gating said "skip" (model confident) | `"medium"` |
| avg(final_score of trusted) ≥ 0.7 | `"high"` |
| avg(final_score) ∈ [0.45, 0.7) | `"medium"` |
| avg(final_score) < 0.45 or no trusted passages | `"low"` |

> **Note:** Gating-skipped queries now receive `"medium"` (not `"high"`) because we cannot verify factuality without passages.

### 4.7 Evaluation & FactScore

**Claim decomposition:** `decompose_claims(answer)` splits answer into atomic claims via regex sentence splitting (handles abbreviations like Mr., Dr., U.S., etc.).

**FactScore computation:** `compute_factscore(answer, passages, nli_fn)` performs real claim-level verification:
1. Decompose answer into claims.
2. For each claim, check NLI entailment against all passages.
3. If any passage entails the claim (score > threshold), claim is supported.
4. FactScore = `n_supported / n_claims`.

Fallback: `compute_factscore_stub()` uses word-overlap proxy when no NLI function is available.

### 4.8 Experiment Runner

Every run saves to `runs/<run-id>/`:

```
runs/20260228_145000_a1b2c3d4/
├── predictions.jsonl   ← per-query: input, answer, passages, provenance
├── metrics.json        ← aggregated EM, F1, FactScore stub
└── metadata.json       ← timestamp, git commit, seed, lib versions, models
```

---

## 5. Configuration Architecture

All tunables live in `configs/exp_sample.yaml`:

```yaml
models:
  dense_embedder: "sentence-transformers/all-mpnet-base-v2"   # 768-dim
  nli_verifier: "ynie/roberta-large-snli_mnli_fever_anli_..."  # NLI
  generator: "mistral-7b-instruct"                             # LLM

retriever:
  alpha: 0.6          # dense weight in combined score
  top_k: 10

gating:
  entropy_thresh: 1.2
  logit_gap_thresh: 2.0

scorer:
  weights: {w_nli: 0.5, w_overlap: 0.2, w_ret: 0.3}
  score_threshold: 0.4
```

Models, thresholds, and weights are all configurable without code changes.

---

## 6. Mock Mode Architecture

Mock mode is a first-class design concern, not an afterthought:

```
┌─────────────┐
│  mock_mode?  │
└──────┬──────┘
       │
  YES  │  NO
  ▼    │  ▼
┌──────────┐  ┌──────────────────┐
│ np.random │  │ HF model.encode()│
│ State(42) │  │ / .forward()     │
│ → fixed   │  │ / pipeline()     │
│ vectors   │  │ → real vectors   │
└──────────┘  └──────────────────┘
```

**Guarantees:**
- Deterministic outputs (same seed → same results).
- No network calls (no HuggingFace Hub downloads).
- No GPU required.
- All tests pass in < 2 seconds.

**Where mock is applied:**

| Component | Real Mode | Mock Mode |
|-----------|-----------|-----------|
| Index builder | SentenceTransformer → FAISS | `randn(N, 768)` → FAISS |
| Retriever | FAISS search + Pyserini BM25 | FAISS search + random BM25 scores |
| Gating probe | AutoModelForCausalLM forward | `randn(32000)` logits |
| Passage scorer | NLI pipeline | `uniform(0.3, 0.95)` per passage |
| Generator | text-generation pipeline | `f"Mock answer for query: {q}"` |
| Model registry | model_registry calls | Returns `None` (skipped) |

---

## 7. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Dense retrieval | `sentence-transformers/all-mpnet-base-v2` + FAISS | Semantic search |
| Sparse retrieval | Pyserini / Lucene BM25 | Keyword matching |
| NLI scoring | `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` | Entailment detection |
| Generation | Mistral-7B-Instruct-v0.3 (4-bit quant) | Answer production |
| Index | FAISS IndexHNSWFlat (dev) / IndexIVFPQ (prod) | Vector similarity |
| Data | HuggingFace Datasets API | Dataset loading |
| Config | YAML (PyYAML) | Experiment configuration |
| Testing | pytest | Unit & integration tests |
| Packaging | pyproject.toml + setuptools | Distribution |

---

## 8. Scaling Considerations

### Current (dev / v0.1)
- FAISS HNSW Flat — stores full vectors, O(log N) search.
- Single-threaded, CPU-only.
- 10-passage sample corpus for tests.

### Production path
- **FAISS IVFPQ** — quantised vectors, ~32× memory reduction.
- **Sharded indexes** — split corpus across multiple FAISS shards.
- **GPU encoding** — batch SentenceTransformer on GPU for indexing.
- **Pyserini real BM25** — build Lucene index via `pyserini.index` CLI.
- **Model parallelism** — 7B generator with `device_map="auto"` on multi-GPU.
- **Batched inference** — process multiple queries simultaneously.

---

## 9. Security & Reproducibility

- **No proprietary APIs** — everything runs locally.
- **Pinned seeds** — `seed=42` by default; configurable.
- **Git commit tracking** — metadata includes `git rev-parse HEAD`.
- **Library version tracking** — faiss, transformers, datasets versions logged.
- **Config persistence** — full YAML config saved per run.
- **Deterministic mock** — CI tests are independent of network and hardware.
