# External Claude Validation Prompt

> Copy everything below the line into a new Claude conversation to get an independent code review and improvement suggestions for your Factuality-First RAG project.

---

## PROMPT START — COPY FROM HERE

You are an expert NLP/IR researcher and Python engineer. I'm sharing my complete project for a **Factuality-First RAG** system. Please review the design, code structure, and experiment plan, then provide detailed feedback.

---

### 1. Project Brief

**Title:** Factuality-First Retrieval-Augmented Generation with Adaptive Gating and Passage-Level Provenance Scoring

**Version:** 0.2.0 (Session 2 — 2026-02-28)  
**Team:** 3 members (graduate-level)  
**Compute:** 1× NVIDIA A100-80GB  
**Timeline:** 4 months  

**Core idea:** Extend standard RAG with two novel components:
1. **Adaptive retrieval gating** — a single-step logit probe on the generator's next-token distribution decides whether retrieval is needed (based on entropy and logit gap thresholds).
2. **Passage-level factuality scoring** — each retrieved passage is scored using NLI entailment probability + token overlap + normalised retrieval score, via a linear fusion formula.

The system reduces unnecessary retrieval calls (efficiency) while improving faithfulness (factuality) compared to always-retrieve baselines.

---

### 2. System Architecture (ASCII)

```
User Query
    │
    ▼
┌─────────────────────┐
│   Gating Probe      │  entropy > 1.2 OR logit_gap < 2.0 → RETRIEVE
│ (single forward pass)│  else → SKIP (return parametric answer)
└────────┬────────────┘
         │ retrieve=True
         ▼
┌─────────────────────┐
│  Hybrid Retriever   │  combined = α·dense_norm + (1−α)·bm25_norm
│ (FAISS + Pyserini)  │  α = 0.6 (default), min-max normalisation per query
└────────┬────────────┘
         │ top-K passages
         ▼
┌─────────────────────┐
│  Passage Scorer     │  final_score = 0.5·P(ent) + 0.2·overlap + 0.3·ret_norm
│ (NLI + overlap)     │  filter: keep if final_score ≥ 0.4
└────────┬────────────┘
         │ trusted passages
         ▼
┌─────────────────────┐
│    Generator        │  Mistral-7B-Instruct-v0.3 (4-bit quantised)
│  (model_registry)   │  [INST] RAG prompt template
└────────┬────────────┘
         │
         ▼
   (answer, trusted_passages, provenance, confidence_tag)
```

---

### 3. Package Structure

```
factuality_rag/
├── __init__.py              # __version__ = "0.1.0"
├── model_registry.py        # Singleton model cache — get_model(), get_tokenizer(), 4-bit quant
├── data/
│   ├── __init__.py
│   ├── loader.py            # HuggingFace dataset loading (NQ, HotpotQA, FEVER, TriviaQA, TruthfulQA)
│   └── wikipedia.py         # WikiChunker: chunk_text(), process_articles(), load_from_hf(), JSONL output
├── index/
│   ├── __init__.py
│   └── builder.py           # build_faiss_index() (HNSW/IVFPQ), prepare_pyserini_collection()
├── retriever/
│   ├── __init__.py
│   └── hybrid.py            # HybridRetriever: retrieve(), build_mock(), real BM25 via LuceneSearcher
├── gating/
│   ├── __init__.py
│   └── probe.py             # GatingProbe: should_retrieve(), calibrate_temperature(), shares model via registry
├── scorer/
│   ├── __init__.py
│   └── passage.py           # PassageScorer: score_passages() with NLI + overlap + ret fusion
├── generator/
│   ├── __init__.py
│   └── wrapper.py           # Generator: generate() — real Mistral inference via model_registry
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py      # run_pipeline(), Pipeline class (load-once, run-many)
├── eval/
│   ├── __init__.py
│   └── metrics.py           # compute_em(), compute_f1(), decompose_claims(), compute_factscore(), evaluate_predictions()
├── cli/
│   ├── __init__.py
│   └── __main__.py          # CLI: build-index, chunk-wiki, run, evaluate
└── experiment_runner.py     # Structured experiment runner using Pipeline class
```

Tests: 8 files, 53 tests, all passing. Every component supports `mock_mode=True` for deterministic CI testing using `np.random.RandomState(42)`. GitHub Actions CI runs pytest, ruff, and mypy on push.

---

### 4. Key Design Decisions (Please Critique)

1. **Gating probe uses entropy + logit gap (no learned classifier).** Decision rule: `retrieve = (entropy > 1.2) OR (logit_gap < 2.0)`. Is this sufficient, or should we train a small binary classifier on dev set features?

2. **Linear fusion for passage scoring.** `final_score = 0.5 * P(entailment) + 0.2 * overlap + 0.3 * retrieval_norm`. Should we use learned weights? A different combination function (e.g., multiplicative)?

3. **Per-query min-max normalisation for hybrid retrieval.** `score_norm = (s - min) / (max - min + ε)` then `combined = α * dense_norm + (1-α) * bm25_norm`. Is min-max appropriate, or should we use z-score / softmax normalisation?

4. **FactScore uses rule-based claim decomposition + NLI.** `decompose_claims()` splits on sentence boundaries; `compute_factscore()` verifies each claim via NLI (premise=passage, hypothesis=claim). Is rule-based decomposition sufficient, or should we use an LLM for compound-claim splitting?

5. **Confidence tagging logic:**
   - Gating skips retrieval → "medium" (model was confident but answer is unverified)
   - No trusted passages after scoring → "low"
   - Average final_score ≥ 0.7 → "high", ≥ 0.45 → "medium", else "low"
   
   Is this heuristic reasonable?

6. **Mock-mode architecture:** Every component has `mock_mode=True` that uses seeded random values. This allows all 53 tests to pass without downloading any models (~2 sec). We also have a `@pytest.mark.integration` marker for GPU-requiring tests. Is this a good testing strategy?

---

### 5. Key Code: Orchestrator Pipeline

```python
# Option A: Functional API (creates components per call)
def run_pipeline(
    query: str,
    k: int = 10,
    gate: bool = True,
    score_threshold: float = 0.4,
    config_path: str = "configs/exp_sample.yaml",
    seed: int = 42,
    mock_mode: bool = False,
    # Pre-built components (avoids re-instantiation)
    probe: GatingProbe | None = None,
    retriever: HybridRetriever | None = None,
    scorer: PassageScorer | None = None,
    generator: Generator | None = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
    """Returns: (answer, trusted_passages, provenance, confidence_tag)"""
    ...

# Option B: Class API (load once, run many — preferred for experiments)
class Pipeline:
    def __init__(self, config_path, mock_mode=False, gate=True, k=10, score_threshold=0.4, seed=42):
        # Loads probe, retriever, scorer, generator ONCE
        # Generator + GatingProbe share model via model_registry
        ...
    
    def run(self, query: str) -> Tuple[str, List[Dict], Dict, str]:
        return run_pipeline(query, ..., probe=self.probe, retriever=self.retriever, ...)
```

---

### 6. Key Code: Gating Probe

```python
class GatingProbe:
    def should_retrieve(self, prompt, probe_tokens=1, entropy_thresh=1.2, logit_gap_thresh=2.0) -> bool:
        logits = self._get_next_token_logits(prompt, probe_tokens)
        probs = softmax(logits / self.temp)
        entropy = -sum(p * log(p) for p in probs if p > 0)
        sorted_logits = sorted(logits, reverse=True)
        logit_gap = sorted_logits[0] - sorted_logits[1]
        return (entropy > entropy_thresh) or (logit_gap < logit_gap_thresh)
    
    def calibrate_temperature(self, calibration_data, temps=[0.5, 0.75, 1.0, 1.5, 2.0]) -> float:
        """Grid search over temperature values to minimise ECE proxy."""
        # Returns best temperature for calibrated gating decisions
```

---

### 7. Key Code: Passage Scorer

```python
class PassageScorer:
    def score_passages(self, query, passages) -> List[Dict]:
        ret_scores = [p["combined_score"] for p in passages]
        ret_min, ret_max = min(ret_scores), max(ret_scores)
        
        for p in passages:
            # NLI: passage is premise (evidence), query is hypothesis (claim)
            p["nli_score"] = self._nli_entailment(premise=p["text"], hypothesis=query)
            p["overlap_score"] = self._compute_overlap(query, p["text"]) # token F1
            ret_norm = (p["combined_score"] - ret_min) / (ret_max - ret_min + 1e-9)
            p["final_score"] = (self.w_nli * p["nli_score"] +
                                self.w_overlap * p["overlap_score"] +
                                self.w_ret * ret_norm)
        return passages
```

---

### 8. Experiment Plan Summary

**Datasets:** NQ-Open, HotpotQA, FEVER, TriviaQA, TruthfulQA  
**Baselines:** Closed-book, Always-RAG, Gate-only, Score-only, (Self-RAG if time permits)  
**Metrics:** EM, Token F1, FactScore, Hallucination rate, Retrieval call %  
**Ablations:** α sweep, threshold sweep, scorer weight grid search, top-K sweep, encoder comparison, cross-encoder reranking  
**Statistical rigour:** 3 seeds, paired bootstrap test (n=1000)

---

### 9. Known Gaps / TODO Items

1. ~~Generator is a mock~~ → ✅ Real Mistral-7B via model_registry (v0.2)
2. ~~BM25 returns mock scores~~ → ✅ Real LuceneSearcher with fallback (v0.2)
3. ~~FactScore is a word-overlap proxy~~ → ✅ Real claim decomposition + NLI (v0.2)
4. No cross-encoder reranking stage
5. No learned gating or scorer weights
6. ~~Wikipedia ingestion is mock-only~~ → ✅ HF streaming via load_from_hf() (v0.2)
7. ~~No CI/CD pipeline~~ → ✅ GitHub Actions (pytest + ruff + mypy) (v0.2)

---

### 10. What I Need From You

Please provide:

1. **Architecture review:** Are there fundamental design flaws? Missing components? Better alternatives to our approach?

2. **Code quality feedback:** Anti-patterns, potential bugs, or improvements to the module structure.

3. **Scoring formula critique:** Is our linear fusion `0.5*NLI + 0.2*overlap + 0.3*ret` well-motivated? What alternatives should we consider?

4. **Gating probe critique:** Is entropy + logit gap sufficient? Should we use sequence-level entropy? Multiple token positions?

5. **Experiment plan feedback:** Are the baselines appropriate? Missing ablations? Statistical methodology concerns?

6. **Comparison to related work:** How does this compare to Self-RAG, CRAG, FLARE, and other recent factuality-focused RAG systems? What distinguishes our approach?

7. **Priority recommendations:** Given 4 months and 1× A100, what should we prioritise implementing first?

8. **Potential paper contributions:** What would be the strongest claims/contributions for an academic paper based on this system?

---

## PROMPT END

---

## How to Use This Prompt

1. Open a new Claude conversation (or any capable LLM).
2. Copy everything between **PROMPT START** and **PROMPT END**.
3. Paste it as a single message.
4. The external Claude will provide a structured review covering all 8 areas.

### Follow-up questions you can ask the external Claude:

- "Can you suggest specific prompt templates for the generator that would improve faithfulness?"
- "How should I implement claim decomposition for FactScore? Give me Python code."
- "What's the best way to create oracle labels for gating accuracy on NQ-Open?"
- "Can you help me design the human evaluation protocol (300 queries, 2 annotators)?"
- "Write a comparison table of our approach vs Self-RAG vs CRAG vs FLARE."
- "Suggest modifications to handle multi-hop questions (HotpotQA) better."
- "How should I fine-tune the NLI model on FEVER for better passage scoring?"
