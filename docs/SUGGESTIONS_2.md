# Suggestions & Improvements — v0.2 → v1.0
> Updated after Session 2. Items marked ✅ are complete.
> Prioritised for remaining 3-month timeline (months 2–4).

---

## Priority Legend

| Label | Meaning | Act on |
|-------|---------|--------|
| **P0** | Blocks first real experiment | Immediately |
| **P1** | Required for paper-quality results | Month 2 |
| **P2** | Strengthens paper significantly | Month 3 |
| **P3** | Future work / nice-to-have | If time allows |

---

## COMPLETED ITEMS ✅

| Item | Session | What was done |
|------|---------|--------------|
| Generator integration | 2 | Mistral-7B-Instruct-v0.3 via model_registry, 4-bit quant, [INST] prompt |
| Model registry singleton | 2 | `get_model()`, `get_tokenizer()`, `clear_registry()` — shared across probe + generator |
| Pipeline class (load-once) | 2 | `Pipeline.__init__()` builds all components once; `run()` is call-site friendly |
| BM25 wiring | 2 | `LuceneSearcher` with triple-graceful fallback (ImportError → missing path → mock) |
| NLI argument order fix | 2 | `premise=passage, hypothesis=query` — was reversed in v0.1 |
| Confidence tag fix | 2 | Gating-skip now returns `"medium"` not `"high"` |
| Real FactScore | 2 | `decompose_claims()` + `compute_factscore()` with per-claim NLI |
| Wikipedia HF ingestion | 2 | `load_from_hf(sample_size, output_path)` streaming |
| CI/CD | 2 | GitHub Actions: pytest (mock only) + ruff + mypy on Python 3.10/3.11/3.12 |
| Experiment configs | 2 | B1 (closed-book), B2 (always-RAG), full pipeline YAML configs |

---

## OPEN ITEMS

---

### 1. Real Index Construction (P0)

**Status:** Not started — unblocks all Phase 3 experiments

**What to do:**
- Stream 100K Wikipedia articles via `load_from_hf()` → `data/wiki_100k_chunks.jsonl`
- Build FAISS HNSW index with `sentence-transformers/all-mpnet-base-v2` (768-dim)
- Build Lucene BM25 index via Pyserini

**Estimated time:** 2-3 hours on A100 for 100K articles

**Risk:** If FAISS build runs out of memory, reduce to 50K articles for dev. Target 400K passages for full run.

---

### 2. Sentence-Level NLI (P1)

**Status:** Not started

**Problem with current approach:**
Passage-level NLI scores an entire 256-token passage as a single unit. A passage with one highly relevant sentence surrounded by irrelevant content may score low overall, causing the scorer to incorrectly filter it out. This is likely the biggest source of `SCORER_DROP` failures.

**Recommended fix:**
```python
def _sentence_level_nli(self, query: str, passage_text: str) -> float:
    """Score a passage by its best-scoring individual sentence."""
    sentences = sent_tokenize(passage_text)  # NLTK or regex
    if len(sentences) <= 1:
        return self._nli_entailment(premise=passage_text, hypothesis=query)
    return max(
        self._nli_entailment(premise=s, hypothesis=query)
        for s in sentences
    )
```

Add config option:
```yaml
scorer:
  nli_mode: "sentence"   # "passage" (v0.1 default) or "sentence" (v0.2+)
```

**Expected impact:** +2-5 points FactScore on NQ-Open. High confidence.
**Implementation effort:** 1-2 hours.

---

### 3. Provenance Mapping (P1 — BUG-8)

**Status:** `provenance` dict returned by `run_pipeline()` is currently a mock structure (e.g., `{"0": ["doc_5"]}`). Real claim-to-passage mapping is not implemented.

**Problem:** Without real provenance, you cannot compute provenance precision (key ablation metric) or show qualitative provenance examples in the paper.

**What to implement:**
After `compute_factscore()`, for each claim that was verified as "supported", record which passage provided the entailment:
```python
provenance = {}
for i, claim in enumerate(claims):
    for p in trusted_passages:
        if nli_fn(premise=p["text"], hypothesis=claim) > 0.7:
            provenance[str(i)] = [p["id"]]
            break
```

**Integration point:** `pipeline/orchestrator.py` → pass `nli_fn` from scorer into `compute_factscore()` and build provenance dict from `details` return value.

**Expected effort:** 3-4 hours.

---

### 4. Cross-Encoder Reranking (P2)

**Status:** Not started

**Motivation:** Bi-encoder retrieval (FAISS) optimises recall, not precision. A cross-encoder attending to both query and passage jointly is significantly more accurate for reranking. Adding it before the NLI scorer reduces the number of passages the NLI model needs to process.

**Recommended position in pipeline:**
```
Retrieve top-20 (bi-encoder) → Cross-encoder rerank → Take top-10 → NLI scorer
```

**Implementation:**
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank(self, query: str, passages: List[Dict], top_k: int = 10) -> List[Dict]:
    pairs = [(query, p["text"]) for p in passages]
    scores = self.reranker.predict(pairs)
    for p, s in zip(passages, scores):
        p["cross_encoder_score"] = float(s)
    return sorted(passages, key=lambda p: p["cross_encoder_score"], reverse=True)[:top_k]
```

**Config:**
```yaml
retriever:
  top_k_retrieval: 20      # fetch more for reranker
  top_k_after_rerank: 10   # pass this many to NLI scorer
  use_cross_encoder: false  # off by default
```

**Expected impact:** +3-8 EM on NQ-Open; +AUC-ROC for scorer.
**Memory cost:** Cross-encoder MiniLM is 33M params — negligible on A100.

---

### 5. Learned Scorer Weights (P2)

**Status:** Not started. Fixed weights `w_nli=0.5, w_overlap=0.2, w_ret=0.3` are currently unmotivated.

**Problem:** NLI and retrieval scores may be correlated (both measure relevance). Overlap is a weak proxy. The 0.5/0.2/0.3 split was chosen by intuition, not by data.

**Recommended approach (grid search, low effort):**
```python
# scripts/tune_scorer_weights.py
# Grid search over weight combinations on FEVER dev (passage-level relevance labels available)
best = None
for w_nli in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for w_ret in [0.1, 0.2, 0.3]:
        w_overlap = 1.0 - w_nli - w_ret
        if w_overlap < 0:
            continue
        auc = evaluate_scorer(w_nli, w_overlap, w_ret, fever_dev)
        if best is None or auc > best["auc"]:
            best = {"w_nli": w_nli, "w_overlap": w_overlap, "w_ret": w_ret, "auc": auc}
```

**Alternative (logistic regression, slightly more effort):**
Train a binary classifier `(nli, overlap, ret) → relevant/irrelevant` on FEVER train split. More principled.

**Important for paper:** learned weights let you report an ablation showing the contribution of each signal individually.

---

### 6. Real ECE Calibration for Gating (P2 — BUG-6)

**Status:** Current `calibrate_temperature()` uses entropy std-dev as an ECE proxy — this is not the same as Expected Calibration Error and produces unreliable temperature choices.

**Impact on paper:** If you claim "calibrated confidence estimation" in the paper, reviewers will ask about calibration quality. Using a proxy is a weakness.

**Fix (binned ECE):**
```python
def compute_ece(
    self,
    confidences: np.ndarray,    # max softmax prob per query
    accuracies: np.ndarray,     # 1 if EM=1, 0 otherwise
    n_bins: int = 15
) -> float:
    """Expected Calibration Error (lower is better)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(confidences[mask].mean() - accuracies[mask].mean())
    return ece / max(len(confidences), 1)
```

**For calibrate_temperature():** On dev set, compute (confidence=max_prob, accuracy=EM) pairs at each temperature value. Choose T that minimises ECE. This is straightforward to add.

---

### 7. Multi-Token Gating Probe (P2)

**Status:** `probe_tokens` parameter exists but averaging over multiple positions may not be implemented correctly.

**Problem:** First-token entropy is noisy for factoid queries. The model commits more information over the first 3-5 positions. Averaging entropy over positions 1-3 gives a more stable signal.

**What to check in `gating/probe.py`:**
```python
def _get_multi_token_entropy(self, prompt: str, k: int = 3) -> float:
    """Average entropy over first k generated positions."""
    entropies = []
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    for _ in range(k):
        with torch.no_grad():
            out = self.model(input_ids)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits / self.temp, dim=-1)
        H = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(H)
        # Append most likely token and continue
        next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return np.mean(entropies)
```

**Expected impact:** More stable gating signal, fewer false skips on ambiguous queries.
**Cost:** 3× gating latency (~150ms instead of 50ms) — still cheap vs retrieval.

---

### 8. DPR / Contriever Encoder Comparison (P2)

**Status:** Dense retrieval uses only `all-mpnet-base-v2`. This was not trained on QA tasks.

**Recommended alternative:** `facebook/contriever-msmarco` — trained specifically for open-domain QA retrieval, likely stronger on NQ-Open.

**Pluggable encoder interface:**
```python
class DenseEncoder(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...

class SentenceTransformerEncoder:
    def encode(self, texts): return self.model.encode(texts)

class ContrieverEncoder:
    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        return mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
```

Add `encoder_class` to config and inject at build time. Ablation experiment: Exp 4.5 from experiment plan.

**Note:** Switching encoder requires rebuilding FAISS index — plan accordingly.

---

### 9. Self-RAG Comparison Baseline (P2)

**Status:** Listed as "if time permits" — but this is the most important baseline for the paper.

**Why it matters:** Self-RAG (Asai et al., ICLR 2024) is the closest published work. Reviewers will ask why we didn't compare to it directly. Without this comparison, the paper's contribution is harder to establish.

**Practical approach (without full Self-RAG training):**
- Use the published `selfrag/selfrag_llama2_7b` checkpoint from HuggingFace
- Run on same NQ-Open dev split and Wikipedia corpus
- This gives an apples-to-apples comparison with minimal implementation effort

**If the checkpoint is hard to use:** At minimum, reproduce Self-RAG's gating mechanism (using their reflection tokens on a fine-tuned model) and compare faithfulness metrics. This is the key differentiator: Self-RAG requires fine-tuning; our system doesn't.

---

### 10. Human Evaluation Protocol (P1)

**Status:** Defined in Session 1 docs but not implemented. Required for paper.

**Timeline:** Start recruiting annotators in Month 2. Annotation should run in Month 3 alongside ablations.

**Recommended protocol:**

**Queries:** 300 from NQ-Open test split (stratified by confidence tag: 100 high, 100 medium, 100 low)

**Annotation task per query:**
1. Read the question
2. Read the generated answer + cited passages
3. Rate on 4 binary dimensions:

| Dimension | Question | Notes |
|-----------|----------|-------|
| **Correctness** | Is the answer factually correct? | Compare to gold answer |
| **Groundedness** | Is every claim in the answer supported by a cited passage? | Key faithfulness measure |
| **Relevance** | Are the cited passages relevant to the question? | Retrieval quality signal |
| **Hallucination** | Does the answer contain any unsupported claim? | Inverse of groundedness |

**Annotator setup:**
- 2 independent annotators per query (don't show each other's labels)
- Compute Cohen's κ — target κ > 0.7 for main metrics
- If κ < 0.6, refine annotation guidelines and re-annotate 50 queries

**Comparison plan:**
- Annotate B2 (always-RAG) and Full Pipeline on same 300 queries
- Statistical test: McNemar's test for Groundedness and Hallucination rates
- Primary paper claim: Full Pipeline has significantly higher Groundedness at same Correctness

---

### 11. Prompt Engineering Study (P1)

**Status:** Not started. Currently using a single RAG prompt template.

**Problem:** Mistral-7B may ignore retrieved passages if the prompt doesn't strongly incentivise using them. This would show up as `GEN_IGNORE` errors in Phase 4C analysis.

**Recommended prompt variants to test:**

**Variant A (current):**
```
[INST] Answer the question using ONLY the provided context.

Context:
{context}

Question: {query} [/INST]
```

**Variant B (explicit citation instruction):**
```
[INST] Answer the question using the provided passages. 
You MUST base your answer only on the passages below. 
If the passages don't answer the question, say "I cannot determine this from the provided context."

Passages:
{numbered_passages}

Question: {query}
Answer (cite passage numbers): [/INST]
```

**Variant C (chain-of-thought):**
```
[INST] Read the passages and answer step by step.

Passages:
{context}

Question: {query}
Let me think step by step: [/INST]
```

**Evaluation:** Run each variant on NQ dev 100 samples. Measure FactScore + GEN_IGNORE rate. Choose best for main experiments.

---

### 12. Iterative Retrieval for Low-Confidence Cases (P3)

**Status:** Future work

**Motivation:** If all retrieved passages score below threshold (confidence="low"), the system currently falls back to parametric generation. An alternative: reformulate the query and retrieve again.

**Sketch:**
```python
if confidence_tag == "low" and allow_retry:
    # Reformulate: extract key entities from query, try alternative phrasing
    reformulated = f"What is {query.replace('who is', '').replace('what is', '').strip()}?"
    passages_v2 = retriever.retrieve(reformulated, k=k)
    # Score and filter again...
```

This is similar to CRAG's corrective retrieval but driven by the passage scorer rather than an external retrieval judge.

---

## Implementation Priority Order (Months 2-4)

| Month | Focus | Key deliverable |
|-------|-------|----------------|
| **Month 2** | Experiments | Real indexes built, B1/B2/Full baselines run, error analysis done |
| **Month 2** | Scorer fix | Sentence-level NLI implemented + ablation |
| **Month 2** | Provenance | Real claim→passage mapping in pipeline output |
| **Month 3** | Ablations | α sweep, threshold sweep, top-K sweep, scorer weight tuning |
| **Month 3** | Cross-encoder | Optional reranking stage + ablation |
| **Month 3** | Human eval | 300-query annotation with 2 annotators |
| **Month 3** | Self-RAG baseline | selfrag HF checkpoint comparison |
| **Month 4** | Paper writing | Full results table, ablations, error analysis, human eval |
| **Month 4** | Final eval | Full Wikipedia index, NQ-Open test split |
