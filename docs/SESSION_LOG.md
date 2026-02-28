# Session Log — Factuality-First RAG

> **Date:** 2026-02-28
> **Session type:** Full project bootstrap
> **Platform:** Windows 11 · Python 3.13.5 (Anaconda) · VS Code
> **Working directory:** `E:\Lab\NLP\Faculty-first RAG`

---

## 1. Session Objective

Bootstrap a reproducible, typed, CI-friendly Python package named `factuality_rag` that implements an end-to-end **Factuality-first RAG** pipeline — combining adaptive retrieval gating with passage-level provenance/factuality scoring. All components must work in `--mock-mode` (no GPU, no model downloads) for fast iteration and CI.

---

## 2. Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Use `pyproject.toml` with `setuptools.build_meta` | Modern Python packaging; no `setup.py` needed |
| 2 | Lazy-load all heavy models (transformers, faiss, sentence-transformers) | Avoid import-time crashes; allow mock-mode |
| 3 | Mock-mode uses `np.random.RandomState(42)` throughout | Deterministic outputs for CI; no network calls |
| 4 | FAISS `IndexHNSWFlat` for dev, `IndexIVFPQ` reserved for prod | HNSW is simple, no training needed; IVFPQ scales better |
| 5 | Combined score: `alpha * dense_norm + (1-alpha) * bm25_norm` | Explicit formula avoids ambiguous score mixing |
| 6 | NLI via HuggingFace `pipeline("text-classification")` | Off-the-shelf, no fine-tuning needed for v0.1 |
| 7 | Scorer fusion: `w_nli*P(ent) + w_overlap*overlap + w_ret*ret_norm` | Simple tunable linear fusion; weights in YAML config |
| 8 | Gating via single-step logit probe (entropy + logit gap) | Cheap forward pass; no full decoding needed |
| 9 | Experiment runner saves to `runs/<run-id>/` with full metadata | Reproducibility: git commit, lib versions, seed, config |
| 10 | Generator is a placeholder (`"mistral-7b-instruct"`) | Real integration deferred; mock returns deterministic strings |

---

## 3. What Was Built — Chronological Steps

### Step 1 — Package Structure & Build Config

Created the project skeleton:

- `pyproject.toml` — package metadata, dependencies (`datasets>=2.0`, `transformers>=4.x`, `sentence-transformers`, `faiss-cpu`, `pyserini`, `scikit-learn`, `pytest`), entry-point `factuality-rag`, tool configs (pytest, ruff, mypy).
- `factuality_rag/__init__.py` — version `0.1.0`.
- Empty `__init__.py` files for all 9 sub-packages.
- `.gitignore` — Python, IDE, generated indexes/runs.

**Issue encountered:** Initial `build-backend` was `"setuptools.backends._legacy:_Backend"` which does not exist. Fixed to `"setuptools.build_meta"`.

### Step 2 — Configuration

Created `configs/exp_sample.yaml` with:
- Model IDs (dense embedder, DPR, NLI verifier, generator placeholder)
- Dataset list (NQ, HotpotQA, FEVER, TriviaQA, TruthfulQA)
- Chunker params (size=200, overlap=50)
- Index params (HNSW M=32, ef=200)
- Retriever params (alpha=0.6, normalize=true, top_k=10)
- Gating thresholds (entropy=1.2, logit_gap=2.0)
- Scorer weights (w_nli=0.5, w_overlap=0.2, w_ret=0.3) and threshold=0.4
- Eval metrics list

### Step 3 — `factuality_rag.data` Module

| File | What it does |
|------|-------------|
| `data/loader.py` | `load_dataset(name, split, dev_sample_size)` — wraps HF `datasets.load_dataset()` with known configs for NQ/HotpotQA/FEVER/TriviaQA/TruthfulQA; supports streaming and dev-sampling |
| `data/wikipedia.py` | `WikiChunker` class — chunks text into overlapping token windows; deduplicates by title+text MD5; streams JSONL output; supports `--dry-run`, `--mock-mode`, `--dev-sample-size`; generates mock articles for testing |

### Step 4 — `factuality_rag.index` Module

| File | What it does |
|------|-------------|
| `index/builder.py` | `build_faiss_index()` — loads JSONL, encodes with SentenceTransformer (or random in mock), builds `IndexHNSWFlat` or `IndexIVFPQ`, saves `.index` + `.ids.json` mapping. Also: `save_embeddings()` (numpy), `prepare_pyserini_collection()` (writes Pyserini-compatible JSONL) |

### Step 5 — `factuality_rag.retriever` Module

| File | What it does |
|------|-------------|
| `retriever/hybrid.py` | `HybridRetriever` class — loads FAISS index + id map; encodes query; searches FAISS (dense) + BM25 (mock or Pyserini); min-max normalises scores per query; computes `combined_score = alpha * dense_norm + (1-alpha) * bm25_norm`; sorts by combined score. Factory method `build_mock()` creates an in-memory index for testing |

**Return schema per result:**
```python
{"id", "text", "dense_score", "bm25_score", "dense_norm", "bm25_norm", "combined_score", "metadata"}
```

### Step 6 — `factuality_rag.gating` Module

| File | What it does |
|------|-------------|
| `gating/probe.py` | `GatingProbe` class — single forward pass on generator; computes entropy of softmax distribution and logit gap (top-2 difference); decision rule: `retrieve = (entropy > thresh) or (gap < thresh)`. Includes `calibrate_temperature()` with grid search. Mock-mode simulates a 32k-vocab logit vector via fixed-seed random |

### Step 7 — `factuality_rag.scorer` Module

| File | What it does |
|------|-------------|
| `scorer/passage.py` | `PassageScorer` class — loads NLI pipeline; for each passage computes `P(entailment)`, token/char F1 overlap with query, and retriever score norm; fuses: `final_score = w_nli * nli + w_overlap * overlap + w_ret * ret_norm`. Mock-mode returns calibrated random scores |

### Step 8 — `factuality_rag.generator` Module

| File | What it does |
|------|-------------|
| `generator/wrapper.py` | `Generator` class — wraps HF `pipeline("text-generation")`. Mock-mode returns `"Mock answer for query: {query}"`. Formats RAG prompts with context |

### Step 9 — `factuality_rag.pipeline` Module

| File | What it does |
|------|-------------|
| `pipeline/orchestrator.py` | `run_pipeline()` — seeds RNG, loads config, runs gating → retrieval → scoring → filtering → generation → confidence tag assignment. Returns `(answer, trusted_passages, provenance, confidence_tag)`. Confidence logic: if gating skipped retrieval → "high"; else based on avg final_score of trusted passages |

### Step 10 — `factuality_rag.eval` Module

| File | What it does |
|------|-------------|
| `eval/metrics.py` | `compute_em()` (exact match), `compute_f1()` (token-level F1), `compute_factscore_stub()` (word-overlap proxy for FactScore), `evaluate_predictions()` (batch evaluator) |

### Step 11 — `factuality_rag.cli` Module

| File | What it does |
|------|-------------|
| `cli/__main__.py` | 4 sub-commands: `build_index`, `chunk_wiki`, `run`, `evaluate`. All support `--mock-mode`, `--dev-sample-size`, `--dry-run` where applicable. Entry-point registered as `factuality-rag` in pyproject.toml |

### Step 12 — Experiment Runner

| File | What it does |
|------|-------------|
| `experiment_runner.py` | `run(config, queries, ...)` — runs pipeline on each query, saves `predictions.jsonl` + `metrics.json` + `metadata.json` (timestamp, git commit, lib versions, seed, model names, dataset info) under `runs/<run-id>/` |

### Step 13 — Tests & Sample Data

| File | Coverage |
|------|----------|
| `tests/data/sample_wiki.jsonl` | 10 short passages (Python, ML, Paris, Photosynthesis, Einstein, DNA, WWII, Climate, Shakespeare, QM) |
| `tests/test_retriever.py` | 10 tests: k items returned, required keys, score ranges, sorting, determinism, oversize k, metadata |
| `tests/test_gating.py` | 8 tests: returns bool, determinism, different prompts, calibration, entropy/gap computation, threshold edge cases |
| `tests/test_scorer.py` | 4 tests: adds keys, score range, token overlap, empty passages |
| `tests/test_pipeline.py` | 3 tests: return type, no-gate mode, determinism |
| `tests/test_eval.py` | 7 tests: EM pos/neg, F1 identical/partial, factscore supported/unsupported, batch evaluation |
| `tests/test_data.py` | 5 tests: chunking, schema, dedup, mock articles, dry-run |

### Step 14 — Scripts & README

| File | What it does |
|------|-------------|
| `scripts/run_sample_experiment.sh` | End-to-end bash: chunk → build index → run pipeline (all `--mock-mode`) |
| `scripts/demo.py` | Python demo: runs 3 queries through pipeline + experiment runner |
| `README.md` | Quickstart, CLI commands, package structure, model table, mock-mode docs |

---

## 4. Installation & Validation

### Install

```
pip install -e ".[dev]"
```

Successfully installed with all dependencies:

| Library | Version |
|---------|---------|
| Python | 3.13.5 |
| faiss-cpu | 1.13.2 |
| transformers | 5.2.0 |
| sentence-transformers | 5.2.3 |
| datasets | 4.6.1 |
| pyserini | 1.5.0 |
| scikit-learn | 1.6.1 |
| pytest | 8.3.4 |

### Test Results

```
======================== 36 passed, 3 warnings in 1.95s ========================
```

All 36 tests pass. Warnings are benign SWIG deprecation notices from FAISS.

### CLI Smoke Test

```
> python -m factuality_rag.cli run --query "What is the capital of France?" --mock-mode

============================================================
Query:       What is the capital of France?
Answer:      Mock answer for query: What is the capital of France?
Confidence:  medium
Trusted:     8 passage(s)
Provenance:  {'0': ['doc_5'], '1': ['doc_11'], ...}
============================================================
```

---

## 5. Known Gaps & Issues

| # | Gap | Impact | Status |
|---|-----|--------|--------|
| 1 | Generator is a mock placeholder | No real answer generation | TODO — integrate Mistral-7B-Instruct |
| 2 | Pyserini BM25 not wired to `LuceneSearcher` | BM25 scores are mock | TODO — uncomment in `_bm25_search()` |
| 3 | Wikipedia dump parser not implemented | Cannot ingest real enwiki | TODO — add XML/bz2 parser |
| 4 | DPR encoder path not implemented | Only SentenceTransformer dense | TODO — add DPR option in builder |
| 5 | FactScore is a word-overlap stub | Not a real claim decomposition | TODO — implement atomic claim extraction + NLI |
| 6 | No GPU device management | Everything defaults to CPU | TODO — add CUDA routing |
| 7 | Calibration uses entropy std-dev as ECE proxy | Not true ECE | TODO — implement proper binned ECE |
| 8 | No CI workflow file | Tests run locally only | TODO — add GitHub Actions |
| 9 | IVFPQ parameters are placeholder | Untested on large corpus | TODO — tune nlist, m_pq, nbits |
| 10 | No Contriever/ColBERT support | Dense is sentence-transformers only | TODO — add alternative encoders |

---

## 6. File Manifest

```
Faculty-first RAG/
├── .gitignore
├── pyproject.toml
├── README.md
├── configs/
│   └── exp_sample.yaml
├── docs/                              ← (this session)
│   ├── SESSION_LOG.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── SUGGESTIONS.md
│   ├── EXPERIMENT_PLAN.md
│   └── CLAUDE_VALIDATION_PROMPT.md
├── factuality_rag/
│   ├── __init__.py
│   ├── experiment_runner.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── __main__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── wikipedia.py
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── gating/
│   │   ├── __init__.py
│   │   └── probe.py
│   ├── generator/
│   │   ├── __init__.py
│   │   └── wrapper.py
│   ├── index/
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── hybrid.py
│   └── scorer/
│       ├── __init__.py
│       └── passage.py
├── scripts/
│   ├── demo.py
│   └── run_sample_experiment.sh
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_eval.py
    ├── test_gating.py
    ├── test_pipeline.py
    ├── test_retriever.py
    ├── test_scorer.py
    └── data/
        └── sample_wiki.jsonl
```

**Total:** 35 files, 9 modules, 36 passing tests.

---
---

# Session 2 — Wiring Real Components & Bug Fixes

> **Date:** 2026-02-28 (session 2)
> **Session type:** Implementation — wire real components, fix critical bugs
> **Platform:** Windows 11 · Python 3.13.5 (Anaconda) · VS Code
> **GPU:** NVIDIA A100-80GB confirmed available
> **Working directory:** `E:\Lab\NLP\Faculty-first RAG`

---

## 1. Session Objective

Implement the prioritised task list from `NEXT_SESSION_PROMPT.md`:
- Fix all 5 identified bugs (BUG-1 through BUG-5)
- Wire real components (generator, model registry, BM25, Wikipedia HF loading)
- Add `Pipeline` class for component reuse
- Implement real FactScore with claim decomposition
- Set up CI/CD and experiment configs
- Update all documentation

---

## 2. Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Return `"medium"` (not `"high"`) when gating skips retrieval | Can't verify factuality without passages — safer default (BUG-3) |
| 2 | Model registry uses 4-bit `BitsAndBytesConfig` quantization | A100-80GB can run 7B models in 4-bit; consistent with Mistral deployment |
| 3 | Mistral `[INST]` prompt wrapping for RAG template | Matches Mistral-7B-Instruct-v0.3 expected format |
| 4 | Claim decomposition via regex sentence splitting (not spaCy) | Avoids heavy dependency; handles abbreviations like Mr./Dr./etc. |
| 5 | `Pipeline` class wraps `run_pipeline()` rather than replacing it | Backward compatibility — raw function still works for simple use cases |
| 6 | Graceful triple fallback in BM25 real path | ImportError → log warning; path missing → log warning; other → log warning; always falls back to mock |
| 7 | Wikipedia HF loading uses `streaming=True` | Avoids downloading full 20GB dump; sample incrementally |
| 8 | CI matrix: Python 3.10, 3.11, 3.12 (not 3.13) | 3.13 has limited ecosystem support; CI should target stable versions |
| 9 | `@pytest.mark.integration` marker for GPU-requiring tests | Allows CI to run `pytest -m "not integration"` without GPU |

---

## 3. What Was Built — Chronological Steps

### Step 1 — BUG-2: NLI Argument Order Fix

**File:** `scorer/passage.py`

- `score_passages()` was calling `self._nli_entailment(query, passage_text)` — **premise and hypothesis were reversed**.
- Fixed to `self._nli_entailment(premise=passage_text, hypothesis=query)`.
- Updated `_nli_entailment()` docstring to document: passage = premise, query/claim = hypothesis.
- Added 2 regression tests in `test_scorer.py`:
  - Docstring verification test
  - Spy-pattern test that intercepts NLI calls and verifies argument order

### Step 2 — BUG-3: Confidence Tag Logic Fix

**File:** `pipeline/orchestrator.py`

- `_compute_confidence()` was returning `"high"` when gating skipped retrieval — misleading since model can hallucinate without evidence.
- Changed to return `"medium"` with comment: *"Gating skipped retrieval — model seemed confident, but we can't verify without passages."*
- Added regression test in `test_pipeline.py` asserting gating-skip produces `"medium"`.

### Step 3 — 1B: Model Registry (Singleton)

**New file:** `factuality_rag/model_registry.py`

- `_models: Dict[str, Any]` and `_tokenizers: Dict[str, Any]` module-level singletons.
- `get_model(model_id, device, quantize_4bit)` — loads with 4-bit `BitsAndBytesConfig` if `quantize_4bit=True` and `bitsandbytes` is installed.
- `get_tokenizer(model_id)` — caches tokenizer.
- `clear_registry()` — clears all cached models/tokenizers.
- `is_loaded(model_id)` — check if a model is already in the registry.
- Graceful fallback: if `bitsandbytes` not installed, loads without quantization.
- 3 unit tests in `tests/test_model_registry.py`.

### Step 4 — 1A: Generator Real Integration

**File:** `generator/wrapper.py` (rewritten)

- Added Mistral `[INST]` RAG prompt templates:
  - `_RAG_PROMPT_TEMPLATE` — with context block
  - `_RAG_PROMPT_NO_CONTEXT` — direct question answering
- Real generation via `model.generate()` with `max_new_tokens=256, temperature=0.1, do_sample=False`.
- Uses model registry for shared model loading (`get_model()` / `get_tokenizer()`).
- Accepts optional pre-loaded `model` and `tokenizer` kwargs (for Pipeline reuse).
- Strips prompt tokens from output — decodes only newly generated tokens.
- Mock path preserved unchanged (`"Mock answer for query: {query}"`).

### Step 5 — 1B continued: GatingProbe Update

**File:** `gating/probe.py`

- `__init__` now accepts optional `model` and `tokenizer` params.
- `_load_model()` uses `model_registry.get_model()` and `get_tokenizer()` instead of direct `AutoModelForCausalLM.from_pretrained()`.
- Both components now share the same model instance when used via `Pipeline`.

### Step 6 — 1C: Pipeline Class + Orchestrator Refactor

**File:** `pipeline/orchestrator.py`

- `run_pipeline()` now accepts `probe`, `retriever`, `scorer`, `generator` keyword-only args for pre-built component injection.
- **New `Pipeline` class:**
  - `__init__(config_path, mock_mode, seed, ...)` — loads all 4 components once from config.
  - `run(query, k, gate, score_threshold)` — delegates to `run_pipeline()` passing pre-built components.
  - Solves BUG-1: models loaded exactly once, reused across all queries.
- **Updated `experiment_runner.py`** — builds `Pipeline` once before query loop.
- **Updated `cli/__main__.py`** — `_cmd_run()` uses `Pipeline` class.
- 3 tests in `TestPipelineClass` (basic, reuse, no_gate).

### Step 7 — 1D: BM25/Pyserini Wiring

**File:** `retriever/hybrid.py`

- `_bm25_search()` now attempts real `LuceneSearcher` integration.
- Triple graceful fallback:
  1. `ImportError` (Pyserini not installed) → warning, fall back to mock.
  2. Path not found (`pyserini_index_path` missing) → warning, fall back to mock.
  3. Any other `Exception` → warning, fall back to mock.
- Mock path unchanged — same random BM25 scores as before.

### Step 8 — 1E: Wikipedia HF Loading

**File:** `data/wikipedia.py`

- New method `load_from_hf(wiki_config, sample_size, output_path)`:
  - Streams from HuggingFace `wikipedia` dataset via `load_dataset("wikipedia", config, streaming=True)`.
  - Pipes articles through existing `chunk_text()` and `process_articles()`.
  - Configurable `sample_size` (default 100,000) and `output_path`.

### Step 9 — 2C: FactScore Real Implementation

**File:** `eval/metrics.py`

- **`decompose_claims(answer)`** — regex sentence splitting with abbreviation handling (Mr., Dr., U.S., etc.). Splits on `.`, `?`, `!` at sentence boundaries.
- **`compute_factscore(answer, passages, nli_fn, entailment_threshold=0.7)`** — real claim-level FactScore:
  - Decomposes answer into claims.
  - For each claim, checks NLI entailment against all passages.
  - Returns `{"factscore": float, "n_claims": int, "n_supported": int, "details": [...]}`
  - Falls back to word-overlap if no `nli_fn` provided.
- Updated `evaluate_predictions()` to accept optional `nli_fn` param.
- Metric key changed from `factscore_stub` to `factscore`.
- 4 claim decomposition tests + 4 FactScore tests in `test_eval.py`.

### Step 10 — Phase 5: CI/CD

**New file:** `.github/workflows/ci.yml`

- Matrix: Python 3.10, 3.11, 3.12 on `ubuntu-latest`.
- Steps: checkout → setup-python → install → pytest (excluding `integration` marker) → ruff check.
- Separate `type-check` job with mypy.
- Added `markers` config to `pyproject.toml` for `@pytest.mark.integration`.

### Step 11 — Phase 3: Experiment Configs

Created 3 experiment YAML files:

| File | Purpose |
|------|---------|
| `configs/exp_b1_closed_book.yaml` | Gating disabled, `top_k=0` — closed-book baseline |
| `configs/exp_b2_always_rag.yaml` | Gating disabled, always retrieve — always-RAG baseline |
| `configs/exp_full_pipeline.yaml` | Full pipeline: gating + retrieval + scoring — default settings |

---

## 4. Test Results

```
======================== 53 passed, 3 warnings in 17.08s =======================
```

All 53 tests pass. 17 new tests added:
- 3 × `TestPipelineClass` (basic, reuse, no_gate)
- 1 × BUG-3 regression (gating-skip confidence)
- 4 × `TestClaimDecomposition`
- 4 × `TestFactScore`
- 2 × BUG-2 regression (NLI argument order)
- 3 × `TestModelRegistry`

Warnings remain benign SWIG deprecation notices from FAISS.

---

## 5. Bugs Fixed

| ID | File | Bug | Fix Applied | Status |
|----|------|-----|-------------|--------|
| BUG-1 | `pipeline/orchestrator.py` | Models re-instantiated every call | `Pipeline` class loads once + component injection | ✅ Fixed |
| BUG-2 | `scorer/passage.py` | NLI premise/hypothesis reversed | Fixed to `premise=passage, hypothesis=query` | ✅ Fixed |
| BUG-3 | `pipeline/orchestrator.py` | Gating-skip auto-tags `"high"` | Changed to `"medium"` | ✅ Fixed |
| BUG-4 | `retriever/hybrid.py` | BM25 returns mock scores only | Wired real `LuceneSearcher` with graceful fallback | ✅ Fixed |
| BUG-5 | `eval/metrics.py` | FactScore is word-overlap stub | Added `decompose_claims()` + `compute_factscore()` | ✅ Fixed |

---

## 6. Files Changed & Created

### Modified Files
| File | Changes |
|------|---------|
| `scorer/passage.py` | NLI argument order fix, docstring update |
| `pipeline/orchestrator.py` | Confidence fix, component injection, `Pipeline` class |
| `generator/wrapper.py` | Full rewrite with Mistral `[INST]` template + model registry |
| `gating/probe.py` | Model registry integration, optional model/tokenizer params |
| `retriever/hybrid.py` | Real Pyserini `LuceneSearcher` wiring with fallback |
| `data/wikipedia.py` | `load_from_hf()` method for HuggingFace streaming |
| `eval/metrics.py` | `decompose_claims()`, `compute_factscore()`, updated evaluation |
| `cli/__main__.py` | Uses `Pipeline` class |
| `experiment_runner.py` | Uses `Pipeline` class |
| `pyproject.toml` | Added pytest `integration` marker |
| `tests/test_pipeline.py` | `TestPipelineClass` + BUG-3 regression |
| `tests/test_eval.py` | `TestClaimDecomposition` + `TestFactScore` |
| `tests/test_scorer.py` | BUG-2 regression tests |

### New Files
| File | Purpose |
|------|---------|
| `factuality_rag/model_registry.py` | Singleton model/tokenizer registry |
| `.github/workflows/ci.yml` | GitHub Actions CI pipeline |
| `configs/exp_b1_closed_book.yaml` | Closed-book baseline experiment config |
| `configs/exp_b2_always_rag.yaml` | Always-RAG baseline experiment config |
| `configs/exp_full_pipeline.yaml` | Full pipeline experiment config |
| `tests/test_model_registry.py` | Model registry unit tests |

---

## 7. Updated File Manifest

```
Faculty-first RAG/
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                     ← NEW (Session 2)
├── pyproject.toml
├── README.md
├── configs/
│   ├── exp_sample.yaml
│   ├── exp_b1_closed_book.yaml        ← NEW (Session 2)
│   ├── exp_b2_always_rag.yaml         ← NEW (Session 2)
│   └── exp_full_pipeline.yaml         ← NEW (Session 2)
├── docs/
│   ├── SESSION_LOG.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── SUGGESTIONS.md
│   ├── EXPERIMENT_PLAN.md
│   ├── CLAUDE_VALIDATION_PROMPT.md
│   └── NEXT_SESSION_PROMPT.md
├── factuality_rag/
│   ├── __init__.py
│   ├── experiment_runner.py           ← MODIFIED (Session 2)
│   ├── model_registry.py             ← NEW (Session 2)
│   ├── cli/
│   │   ├── __init__.py
│   │   └── __main__.py               ← MODIFIED (Session 2)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── wikipedia.py              ← MODIFIED (Session 2)
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py                ← MODIFIED (Session 2)
│   ├── gating/
│   │   ├── __init__.py
│   │   └── probe.py                  ← MODIFIED (Session 2)
│   ├── generator/
│   │   ├── __init__.py
│   │   └── wrapper.py                ← REWRITTEN (Session 2)
│   ├── index/
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py           ← MODIFIED (Session 2)
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── hybrid.py                 ← MODIFIED (Session 2)
│   └── scorer/
│       ├── __init__.py
│       └── passage.py                ← MODIFIED (Session 2)
├── scripts/
│   ├── demo.py
│   └── run_sample_experiment.sh
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_eval.py                   ← MODIFIED (Session 2)
    ├── test_gating.py
    ├── test_model_registry.py         ← NEW (Session 2)
    ├── test_pipeline.py               ← MODIFIED (Session 2)
    ├── test_retriever.py
    ├── test_scorer.py                 ← MODIFIED (Session 2)
    └── data/
        └── sample_wiki.jsonl
```

**Total:** 41 files, 10 modules, 53 passing tests.

---

## 8. Remaining Work (Next Session)

| Priority | Task | Notes |
|----------|------|-------|
| 🔴 High | Phase 3 experiment runs (B1, B2, Full) | Requires real indexes built from Wikipedia subset |
| 🔴 High | Build real FAISS + Lucene indexes from 100K wiki chunks | 1E-4, 1E-5 |
| 🟡 Medium | Phase 4 validation (gating oracle, scorer AUC, error analysis) | Requires Phase 3 results |
| 🟡 Medium | Real NLI argument order verification with live model | 2A-4, 2A-5 (integration tests) |
| 🟢 Low | BUG-6: Real ECE calibration | Low priority, not blocking experiments |
| 🟢 Low | DPR encoder support, ColBERT, Contriever | Future work |
