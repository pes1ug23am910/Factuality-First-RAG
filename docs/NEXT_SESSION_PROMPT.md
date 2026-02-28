# Factuality-First RAG — Next Session Prompt & Status Tracker
> Hand this entire document to Claude as your first message in a new session.
> Last updated: 2026-02-28 (Session 2) | Current version: v0.1.0 | Tests: 53/53 passing

---

## ⚡ QUICK CONTEXT FOR CLAUDE

You are helping build **Factuality-First RAG** (`factuality_rag`), a Python NLP research package that implements a 4-stage pipeline:

1. **Gating** — entropy + logit-gap probe decides whether to retrieve at all
2. **Hybrid Retrieval** — FAISS (dense) + BM25 (sparse) with min-max score fusion
3. **Passage Scoring** — NLI entailment + token overlap + retriever score → `final_score`
4. **Generation** — Mistral-7B-Instruct conditioned only on trusted passages

**Everything currently works in `--mock-mode` (no models, no GPU, deterministic).** The next phase is wiring real components. All code lives at `E:\Lab\NLP\Faculty-first RAG\` on Windows 11, Python 3.13.5 (Anaconda), VS Code.

**Do not refactor working mock infrastructure.** Build real implementations alongside mocks using the existing `mock_mode: bool` flag pattern.

---

## 📦 CURRENT PACKAGE STATE

```
factuality_rag/
├── data/loader.py          ✅ DONE — HF dataset wrapper
├── data/wikipedia.py       ✅ DONE — WikiChunker (mock only, no real XML parser)
├── index/builder.py        ✅ DONE — FAISS HNSW/IVFPQ builder (mock mode works)
├── retriever/hybrid.py     ✅ DONE — HybridRetriever (dense real, BM25 MOCK)
├── gating/probe.py         ✅ DONE — GatingProbe entropy+logit-gap (MOCK logits)
├── scorer/passage.py       ✅ DONE — PassageScorer NLI+overlap (MOCK NLI scores)
├── generator/wrapper.py    ✅ DONE — Generator shell (MOCK: returns string placeholder)
├── pipeline/orchestrator.py ✅ DONE — run_pipeline() (works end-to-end in mock)
├── eval/metrics.py         ✅ DONE — EM, F1, FactScore stub (word overlap proxy)
├── cli/__main__.py         ✅ DONE — 4 CLI commands
└── experiment_runner.py    ✅ DONE — run metadata tracking, saves to runs/
```

**Known critical bugs / design issues (all fixed in Session 2):**
1. ~~`orchestrator.py` instantiates ALL models fresh on every `run_pipeline()` call~~ → ✅ Fixed: `Pipeline` class + component injection
2. ~~`scorer/passage.py` NLI argument order was reversed~~ → ✅ Fixed: `premise=passage, hypothesis=query`
3. ~~Gating-skipped queries tagged `confidence="high"`~~ → ✅ Fixed: returns `"medium"` now

---

## ✅ STATUS TRACKER

Mark each task as you complete it. Use: `[ ]` = not started, `[~]` = in progress, `[x]` = done, `[!]` = blocked

---

### PHASE 1 — Wire Real Components (Target: 2 weeks)

#### 1A. Generator Integration (HIGHEST PRIORITY)
- [x] **1A-1** Install `bitsandbytes` for 4-bit quantisation: `pip install bitsandbytes`
- [x] **1A-2** Update `generator/wrapper.py` — replace mock with real `AutoModelForCausalLM` + `BitsAndBytesConfig(load_in_4bit=True)`
- [x] **1A-3** Use model ID: `mistralai/Mistral-7B-Instruct-v0.3`
- [x] **1A-4** Implement proper RAG prompt template (Mistral `[INST]` wrapping)
- [x] **1A-5** Add `max_new_tokens=256`, `temperature=0.1`, `do_sample=False` to generation config
- [ ] **1A-6** Verify generator works standalone (requires GPU): `python -c "from factuality_rag.generator import Generator; g = Generator(); print(g.generate('What is DNA?', context='DNA is a molecule...'))"` 

#### 1B. Shared Model Instance (CRITICAL performance fix)
- [x] **1B-1** Create `factuality_rag/model_registry.py` — singleton with `_models` + `_tokenizers` dicts
- [x] **1B-2** Implemented: `get_model()`, `get_tokenizer()`, `clear_registry()`, `is_loaded()` with 4-bit quantization support
- [x] **1B-3** Updated `GatingProbe` to accept optional `model`/`tokenizer` and use registry
- [x] **1B-4** Updated `Generator` to accept optional `model`/`tokenizer` and use registry
- [x] **1B-5** Updated `run_pipeline()` + `Pipeline` class to load once + pass to all components
- [ ] **1B-6** Verify: a full pipeline run with `mock_mode=False` loads the 7B model exactly **once** (integration test)

#### 1C. Orchestrator Model Instantiation Fix
- [x] **1C-1** Refactored `run_pipeline()` to accept pre-built `probe`, `retriever`, `scorer`, `generator` kwargs
- [x] **1C-2** Added `Pipeline` class in `pipeline/orchestrator.py` — loads all components once at `__init__`
- [x] **1C-3** Updated CLI `_cmd_run()` to use `Pipeline` class
- [x] **1C-4** Updated `experiment_runner.py` to build `Pipeline` once before query loop

#### 1D. BM25 / Pyserini Wiring
- [x] **1D-1** Pyserini installed and verified
- [ ] **1D-2** Build a small Lucene test index (blocked on building real index)
- [x] **1D-3** In `retriever/hybrid.py` → `_bm25_search()`: wired real `LuceneSearcher` with import
- [x] **1D-4** Added triple graceful fallback: ImportError → path missing → general Exception → mock
- [ ] **1D-5** Integration test: verify BM25 returns non-random scores when real index present
#### 1E. Wikipedia Corpus (use HuggingFace, skip XML parser for now)
- [x] **1E-1** Added `load_from_hf()` method to `data/wikipedia.py` with HF streaming
- [x] **1E-2** Pipes HF articles through existing `WikiChunker.chunk_text()` and `process_articles()`
- [x] **1E-3** Stream output supported via `output_path` parameter
- [ ] **1E-4** Build FAISS + Lucene indexes on the 100K subset (next session — requires running data pipeline)
- [ ] **1E-5** Smoke test: run 5 NQ-Open queries end-to-end against real indexes

---

### PHASE 2 — Fix Critical Design Bugs (Target: alongside Phase 1)

#### 2A. NLI Argument Order Fix
- [x] **2A-1** Found and fixed `_nli_entailment()` call — was `(query, passage)`, now `(premise=passage, hypothesis=query)`
- [x] **2A-2** Verified NLI call uses `(premise=passage_text, hypothesis=query)` — docstring updated
- [x] **2A-3** Implementation uses HF `pipeline("text-classification")` which handles tokenization correctly
- [ ] **2A-4** Integration test with real NLI model (known entailment pair) — requires GPU
- [ ] **2A-5** Integration test with real NLI model (known contradiction pair) — requires GPU

#### 2B. Confidence Tag Logic Fix
- [x] **2B-1** Removed `"high"` auto-tag for gating-skipped queries
- [x] **2B-2** Applied simplest safe fix: `gating_skipped → "medium"` (self-check deferred to future)
- [x] **2B-3** `gating_skipped → tag = "medium"` implemented with explanatory comment
- [x] **2B-4** Added regression test in `test_pipeline.py` asserting gating-skip → `"medium"`

#### 2C. FactScore Real Implementation (can be done incrementally)
- [x] **2C-1** Implemented `decompose_claims(answer)` — regex sentence splitting with abbreviation handling
- [x] **2C-2** Implemented `compute_factscore(answer, passages, nli_fn, entailment_threshold=0.7)` — returns dict with `factscore`, `n_claims`, `n_supported`, `details`
- [x] **2C-3** `compute_factscore_stub()` kept for mock; `compute_factscore()` is the real version
- [x] **2C-4** Updated `evaluate_predictions()` to accept `nli_fn` param — uses real FactScore when provided, stub otherwise

---

### PHASE 3 — First Real Experiments (Target: Week 3-4)

#### 3A. Baseline B1 — Closed-Book (no retrieval)
- [x] **3A-1** Created `configs/exp_b1_closed_book.yaml` — gating disabled, `top_k: 0`
- [ ] **3A-2** Run on NQ-Open dev, 500 examples (requires real indexes + GPU)
- [ ] **3A-3** Record results in the Results Table at bottom of this document

#### 3B. Baseline B2 — Always-RAG
- [x] **3B-1** Created `configs/exp_b2_always_rag.yaml` — gating disabled, always retrieve
- [ ] **3B-2** Run on NQ-Open dev, 500 examples (requires real indexes + GPU)
- [ ] **3B-3** Record results in Results Table

#### 3C. Full Pipeline First Run
- [x] **3C-1** Created `configs/exp_full_pipeline.yaml` — full settings (α=0.6, threshold=0.4, gate=true)
- [ ] **3C-2** Run on NQ-Open dev, 500 examples (requires real indexes + GPU)
- [ ] **3C-3** Record results in Results Table
- [ ] **3C-4** Compare EM/F1/FactScore vs B1 and B2 — if full pipeline < B2, investigate why

---

### PHASE 4 — Validation & Quality Checks

#### 4A. Gating Oracle Analysis
- [ ] **4A-1** After Phase 3 runs, extract per-query: `{query, gated, em_score}`
- [ ] **4A-2** Compute gating accuracy: for queries where gating said "skip", what % had EM=1 (model was actually correct)?
- [ ] **4A-3** For queries where gating said "retrieve", what % had EM=0 without retrieval (model actually needed it)?
- [ ] **4A-4** Plot: precision-recall curve for gating decisions — this becomes a key figure in the paper

#### 4B. Scorer AUC Analysis  
- [ ] **4B-1** For NQ-Open, use gold passage annotations to label each retrieved passage as relevant (1) or not (0)
- [ ] **4B-2** Compute AUC-ROC of `final_score` for relevant vs irrelevant passages
- [ ] **4B-3** If AUC < 0.65, the scorer isn't working — investigate NLI argument order (see 2A) first

#### 4C. Error Analysis (manual, 50 queries)
- [ ] **4C-1** Sample 50 queries where full pipeline answer was wrong (EM=0)
- [ ] **4C-2** For each, identify failure mode from this taxonomy:
  - `GATE_MISS` — gating skipped retrieval but model didn't know
  - `RETRIEVAL_MISS` — correct passage not in top-K
  - `SCORER_MISS` — correct passage retrieved but scored below threshold
  - `GEN_HALLUCINATION` — correct passages provided but model ignored them
  - `SCORER_WRONG_FILTER` — correct passage filtered out by scorer
- [ ] **4C-3** Tally counts per failure mode — this drives which component to improve next

---

### PHASE 5 — CI/CD & Reproducibility

- [x] **5-1** Created `.github/workflows/ci.yml` — matrix Python 3.10/3.11/3.12 + ruff + mypy
- [x] **5-2** Added `@pytest.mark.integration` marker support in `pyproject.toml`
- [x] **5-3** CI runs `pytest -m "not integration"` to skip GPU-requiring tests
- [ ] **5-4** Verify CI passes on a clean clone with no GPU (needs GitHub push)

---

## 🐛 KNOWN BUGS (fix before running real experiments)

| ID | File | Bug | Severity | Status |
|----|------|-----|----------|--------|
| BUG-1 | `pipeline/orchestrator.py` | Models re-instantiated every `run_pipeline()` call | 🔴 Critical | [x] ✅ `Pipeline` class + component injection |
| BUG-2 | `scorer/passage.py` | NLI premise/hypothesis order reversed | 🔴 Critical | [x] ✅ Fixed: `premise=passage, hypothesis=query` |
| BUG-3 | `pipeline/orchestrator.py` | Gating-skip auto-tags as `"high"` confidence | 🟡 Medium | [x] ✅ Changed to `"medium"` |
| BUG-4 | `retriever/hybrid.py` | BM25 returns random scores (mock not real) | 🟡 Medium | [x] ✅ Wired `LuceneSearcher` with graceful fallback |
| BUG-5 | `eval/metrics.py` | FactScore is word-overlap stub, not claim-level | 🟡 Medium | [x] ✅ `decompose_claims()` + `compute_factscore()` |
| BUG-6 | `gating/probe.py` | ECE calibration uses entropy std-dev, not real ECE | 🟢 Low | [ ] |

---

## 📐 IMPLEMENTATION CONSTRAINTS

These are non-negotiable — do not change them:

1. **Every real implementation must keep `mock_mode=True` path working.** Never break the mock. Tests must still pass in < 2 seconds.
2. **All tunables stay in YAML config** — no hardcoded thresholds in code.
3. **Seed everything**: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)` at pipeline init.
4. **Every experiment run saves** `predictions.jsonl` + `metrics.json` + `metadata.json` to `runs/<run-id>/`.
5. **No proprietary APIs** — all models must be loadable from HuggingFace locally.
6. **Windows 11 compatibility** — avoid Linux-only shell features in scripts.

---

## 🔧 ENVIRONMENT REFERENCE

```
OS: Windows 11
Python: 3.13.5 (Anaconda)
Working dir: E:\Lab\NLP\Faculty-first RAG\
GPU: NVIDIA A100-80GB (when available)

Key installed versions:
  faiss-cpu          1.13.2
  transformers       5.2.0
  sentence-transformers 5.2.3
  datasets           4.6.1
  pyserini           1.5.0
  scikit-learn       1.6.1
  pytest             8.3.4

Models to use:
  Generator:    mistralai/Mistral-7B-Instruct-v0.3 (4-bit)
  NLI Scorer:   ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
  Embedder:     sentence-transformers/all-mpnet-base-v2
  Cross-encoder (optional): cross-encoder/ms-marco-MiniLM-L-12-v2
```

---

## 📊 RESULTS TABLE (fill in as experiments complete)

| Exp ID | Config | Dataset | N | EM | F1 | FactScore | Retrieval% | Latency | Date |
|--------|--------|---------|---|----|----|-----------|------------|---------|------|
| B1 | closed-book | NQ dev | 500 | — | — | — | 0% | — | |
| B2 | always-RAG | NQ dev | 500 | — | — | — | 100% | — | |
| B3 | gate-only | NQ dev | 500 | — | — | — | —% | — | |
| B4 | score-only | NQ dev | 500 | — | — | — | 100% | — | |
| **FULL** | α=0.6, thr=0.4 | NQ dev | 500 | — | — | — | —% | — | |
| FULL | α=0.6, thr=0.4 | HotpotQA dev | 500 | — | — | — | —% | — | |
| FULL | α=0.6, thr=0.4 | TruthfulQA | 817 | — | — | — | —% | — | |

---

## 🗂️ SESSION HANDOFF NOTES

**Session 1 (2026-02-28):**
- Full package bootstrap: all 9 modules in mock mode, 36 tests passing

**Session 2 (2026-02-28):**
- Fixed all 5 critical/medium bugs (BUG-1 through BUG-5)
- Wired real generator (Mistral `[INST]` template), model registry (4-bit quant), BM25 (Pyserini), Wikipedia HF loading
- Added `Pipeline` class for component reuse across queries
- Implemented real FactScore with claim decomposition + NLI verification
- Set up CI/CD (GitHub Actions) and 3 experiment configs
- 53 tests passing (17 new tests added)
- All documentation updated

**What remains for next session:**
- Build real FAISS + Lucene indexes from 100K Wikipedia chunks (1E-4)
- Run Phase 3 experiments (B1, B2, Full) on NQ-Open dev set
- Phase 4 validation: gating oracle analysis, scorer AUC, error analysis
- Integration tests with real NLI model (2A-4, 2A-5)
- Standalone generator verification (1A-6)
- Verify CI on clean clone (5-4)

**Suggested first task for next session:**
Run the Wikipedia data pipeline (`load_from_hf()` → chunk → build indexes) to create real FAISS + Lucene indexes. Then run the three baseline experiments (B1, B2, Full) on NQ-Open dev 500.

---

## 💬 HOW TO ASK FOR HELP IN THIS SESSION

When asking Claude to implement something, use this format for best results:

> "Implement task **[TASK-ID]** from the status tracker. Here is the current code for `[filename]`: [paste code]. Follow the implementation constraints in the tracker. After implementing, show me the updated test to add to `tests/[test_file].py`."

When something breaks:
> "I got this error running task **[TASK-ID]**: [error]. The relevant code is: [paste]. What's wrong and how do I fix it without breaking mock mode?"

---

*End of session prompt — version 1.0 — generated 2026-02-28*
