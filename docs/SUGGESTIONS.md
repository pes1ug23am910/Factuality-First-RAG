# Suggestions & Improvements — Factuality-First RAG

> Prioritised recommendations for evolving v0.2 → v1.0
>
> Items marked ✅ were implemented in Session 2 (2026-02-28).

---

## Priority Legend

| Label | Meaning |
|-------|---------|
| **P0** | Must-have for first real experiment |
| **P1** | Should-have for paper-quality results |
| **P2** | Nice-to-have / future work |

---

## 1. ✅ Generator Integration (P0) — DONE

### Status: Implemented in v0.2
- Real `Generator` class now loads via `model_registry` with 4-bit quantisation.
- Uses Mistral `[INST]` chat template for RAG prompts.
- Shares model weights with gating probe (no double-loading).
- Mock-mode preserved for CI testing.

### Remaining work
- Add few-shot prompt variants for comparison.
- Integrate Llama-3.1-8B-Instruct as fallback model.

---

## 2. ✅ Pyserini BM25 Integration (P0) — DONE

### Status: Implemented in v0.2
- `_bm25_search()` now uses real `LuceneSearcher` when a valid index exists.
- Graceful fallback: if Pyserini not installed or index not found, logs a warning and returns empty scores.
- Mock-mode unchanged.

### Remaining work
- Build a real Lucene index from the chunked corpus and validate end-to-end.

---

## 3. ✅ Real FactScore Implementation (P1) — DONE

### Status: Implemented in v0.2
- `decompose_claims()` splits answers into atomic sentences.
- `compute_factscore()` does per-claim NLI verification against passages.
- `evaluate_predictions()` accepts `nli_fn` for real scoring.
- Falls back to word-overlap stub when no NLI function provided.

### Remaining work
- Use an LLM for higher-quality claim decomposition (rule-based may miss compound claims).
- Consider integrating the open-source [FactScore library](https://github.com/shmsw25/FActScore) for comparison.

---

## 4. Gating Probe Improvements (P1)

### Current issues
- **ECE calibration** uses entropy std-dev as a proxy — not true Expected Calibration Error.
- **Single-token probe** may miss multi-step uncertainty.
- **No learned gating** — thresholds are hand-tuned.

### Recommendations

#### 4a. Proper ECE calibration
Implement binned ECE:
```python
def compute_ece(confidences, accuracies, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += mask.sum() * abs(avg_conf - avg_acc)
    return ece / len(confidences)
```

#### 4b. Multi-token probe
Extend to probe k=3 tokens (average entropy over the first 3 generated positions) for richer uncertainty signal.

#### 4c. Learned gating (future)
Train a lightweight binary classifier on dev set features `[entropy, logit_gap, query_length, query_type]` → `{retrieve, skip}`. Logistic regression is sufficient and interpretable.

---

## 5. Scorer Enhancements (P1)

### 5a. Learned fusion weights
Instead of fixed `w_nli=0.5, w_overlap=0.2, w_ret=0.3`, learn optimal weights via:
- Grid search on dev set (FEVER or NQ dev split).
- Or lightweight logistic regression: `final_score = σ(w·[nli, overlap, ret] + b)` trained on passage-level labels.

### 5b. Cross-encoder reranking
Add an optional cross-encoder stage between retrieval and NLI scoring:
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
scores = reranker.predict([(query, p["text"]) for p in passages])
```
This is more accurate than bi-encoder similarity for reranking.

### 5c. Sentence-level NLI
Currently NLI is applied at the passage level. For dense passages, break into sentences first and compute max P(entailment) across sentences. This avoids drowning a supporting sentence in a long irrelevant passage.

---

## 6. Evaluation Enhancements (P1)

### 6a. Human evaluation protocol
Design a lightweight annotation task (n ≈ 300 queries):

| Field | Values | Description |
|-------|--------|-------------|
| Correctness | 0/1 | Is the answer factually correct? |
| Groundedness | 0/1 | Is every claim in the answer supported by a provided passage? |
| Relevance | 0/1 | Are the retrieved passages relevant to the query? |
| Hallucination | 0/1 | Does the answer contain unsupported claims? |

Use 2 annotators with inter-annotator agreement (Cohen's κ).

### 6b. Calibrated LLM evaluation
Use a strong open-source LLM (e.g., Llama-3.1-70B or Qwen-2.5-72B) as an automated judge:
- Prompt: "Given this context and question, rate the answer for factual accuracy (1–5)."
- Calibrate against human labels to measure LLM-judge reliability.

### 6c. Per-component ablation metrics
Track intermediate metrics to understand each component's contribution:

| Metric | Component | What it measures |
|--------|-----------|-----------------|
| Retrieval recall@K | Retriever | Does the correct passage appear in top-K? |
| Gating accuracy | Gating probe | % of queries correctly gated (needs oracle labels) |
| Score separation | Scorer | AUC-ROC of final_score for relevant vs irrelevant passages |
| Provenance precision | Pipeline | % of cited passages that are actually relevant |

---

## 7. DPR / Contriever / ColBERT Support (P1)

### Current state
Dense retrieval uses only `sentence-transformers/all-mpnet-base-v2`.

### Recommendation
Add pluggable encoder interface:

```python
class DenseEncoder(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...

class SentenceTransformerEncoder:
    def __init__(self, model_name: str): ...
    def encode(self, texts): ...

class DPREncoder:
    def __init__(self, question_encoder: str, ctx_encoder: str): ...
    def encode_queries(self, queries): ...
    def encode_passages(self, passages): ...

class ContrieverEncoder:
    def __init__(self, model_name: str = "facebook/contriever-msmarco"): ...
    def encode(self, texts): ...
```

This allows comparing retrieval quality across encoder architectures.

---

## 8. ✅ Wikipedia Dump Ingestion (P0) — DONE

### Status: Implemented in v0.2
- `WikiChunker.load_from_hf()` streams Wikipedia from HuggingFace.
- Supports `sample_size` for dev runs, auto-generates output path.
- Chunks output matches existing JSONL schema.

### Remaining work
- Add `mwparserfromhell` support for offline XML dump parsing.

---

## 9. ✅ CI/CD Pipeline (P1) — DONE

### Status: Implemented in v0.2
- `.github/workflows/ci.yml` created with test, lint, and type-check steps.
- `pyproject.toml` updated with `integration` pytest marker for real-model tests.
- All 53 tests pass in mock-mode in CI.

---

## 10. Research Direction Suggestions (P2)

### 10a. Adaptive threshold learning
Instead of fixed `score_threshold=0.4`, learn a per-query threshold:
- Use query difficulty features (length, entity count, question type).
- Train a small regressor: `threshold(q) = f(features(q))`.

### 10b. Iterative retrieval
If the first retrieval round yields low-scoring passages, reformulate the query and retrieve again. This is similar to CRAG but driven by the passage scorer.

### 10c. Multi-hop gating
For multi-hop questions (HotpotQA), the gating probe should detect that multiple retrieval steps are needed. Extend the probe to output `{skip, single_hop, multi_hop}`.

### 10d. Uncertainty quantification for answers
Beyond confidence tags, return calibrated probability estimates for the answer. This requires multiple forward passes or dropout-based uncertainty estimation.

### 10e. Comparative study vs Self-RAG
Implement a clean Self-RAG baseline using the same models/data and provide head-to-head comparison on all metrics. This is the strongest "related work" comparison for the paper.

---

## Summary: Suggested Implementation Order

| Phase | Items | Duration | Status |
|-------|-------|----------|--------|
| **Phase 1** (P0) | Generator integration, BM25 wiring, Wiki ingestion | 2 weeks | ✅ Done |
| **Phase 2** (P1) | Real FactScore, CI/CD, model registry | 2 weeks | ✅ Done |
| **Phase 3** (P1) | Gating improvements, evaluation protocol, cross-encoder | 2 weeks | Open |
| **Phase 4** (P2) | DPR/Contriever, adaptive thresholds, Self-RAG comparison | 3 weeks | Open |
| **Phase 5** | Full experiments, ablations, paper writing | 3 weeks | Open |
