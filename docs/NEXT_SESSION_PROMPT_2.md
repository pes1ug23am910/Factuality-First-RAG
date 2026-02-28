# Factuality-First RAG — Session 3 Prompt & Status Tracker
> Hand this entire document to Claude as your first message in a new session.
> Last updated: 2026-02-28 (Session 2 → 3) | Current version: v0.2.0 | Tests: 53/53 passing

---

## ⚡ QUICK CONTEXT FOR CLAUDE

You are helping build **Factuality-First RAG** (`factuality_rag`), a Python NLP research package. The pipeline has 4 stages:

1. **Gating** — entropy + logit-gap probe on generator's next-token distribution → retrieve or skip
2. **Hybrid Retrieval** — FAISS (dense) + Pyserini BM25 (sparse), min-max fusion
3. **Passage Scoring** — NLI P(entailment) + token overlap F1 + retriever score → `final_score`
4. **Generation** — Mistral-7B-Instruct-v0.3 (4-bit) conditioned on trusted passages only

**Current state:** All components are implemented with real logic AND mock mode. The codebase is structurally complete. Session 3 is about **running real experiments and improving the two weakest components** (gating probe and passage scorer).

**Project location:** `E:\Lab\NLP\Faculty-first RAG\` · Windows 11 · Python 3.13.5 (Anaconda) · NVIDIA A100-80GB

---

## 📦 COMPLETE MODULE STATUS

```
factuality_rag/
├── model_registry.py       ✅ v0.2 — Singleton 4-bit model cache (get_model, get_tokenizer)
├── data/loader.py          ✅ v0.1 — HF dataset wrapper for NQ/HotpotQA/FEVER/TriviaQA/TruthfulQA
├── data/wikipedia.py       ✅ v0.2 — WikiChunker + load_from_hf() HF streaming
├── index/builder.py        ✅ v0.1 — FAISS HNSW/IVFPQ + Pyserini collection prep
├── retriever/hybrid.py     ✅ v0.2 — Real LuceneSearcher + graceful fallback to mock BM25
├── gating/probe.py         ✅ v0.1 — Entropy + logit-gap probe (MOCK logits in CI)
├── scorer/passage.py       ✅ v0.2 — NLI premise=passage, hypothesis=query (MOCK NLI in CI)
├── generator/wrapper.py    ✅ v0.2 — Real Mistral-7B via model_registry (MOCK in CI)
├── pipeline/orchestrator.py ✅ v0.2 — Pipeline class (load-once) + run_pipeline() functional API
├── eval/metrics.py         ✅ v0.2 — EM, F1, decompose_claims(), compute_factscore() real NLI
├── cli/__main__.py         ✅ v0.1 — 4 CLI commands (build_index, chunk_wiki, run, evaluate)
└── experiment_runner.py    ✅ v0.2 — Uses Pipeline class, saves predictions/metrics/metadata
```

**Bugs fixed in Session 2:** Model re-instantiation (BUG-1) ✅, NLI arg order (BUG-2) ✅, Confidence tag logic (BUG-3) ✅, BM25 mock (BUG-4) ✅, FactScore stub (BUG-5) ✅

**Remaining low-priority bug:** ECE calibration uses entropy std-dev proxy instead of real binned ECE (BUG-6, gating/probe.py) 🟢

---

## ✅ STATUS TRACKER — SESSION 3

Mark tasks: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

### PHASE 3A — Build Real Indexes (FIRST PRIORITY — unblocks all experiments)

> These tasks must be done before any experiment in Phase 3B can run.

- [ ] **3A-1** Run Wikipedia HF ingestion for dev subset (100K articles):
  ```bash
  python -c "
  from factuality_rag.data.wikipedia import WikiChunker
  c = WikiChunker(chunk_size=256, chunk_overlap=32)
  c.load_from_hf(sample_size=100_000, output_path='data/wiki_100k_chunks.jsonl')
  print('Done')
  "
  ```
  Expected output: `data/wiki_100k_chunks.jsonl` (~1.5GB, ~400K passages)

- [ ] **3A-2** Build FAISS index on 100K chunk corpus:
  ```bash
  factuality-rag build-index \
      --corpus data/wiki_100k_chunks.jsonl \
      --faiss-out indexes/wiki100k.faiss \
      --embed-model sentence-transformers/all-mpnet-base-v2
  ```
  Expected: `indexes/wiki100k.faiss` + `indexes/wiki100k.ids.json`

- [ ] **3A-3** Prepare Pyserini collection and build Lucene index:
  ```bash
  factuality-rag build-index \
      --corpus data/wiki_100k_chunks.jsonl \
      --pyserini-out indexes/wiki100k_pyserini_collection

  python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input indexes/wiki100k_pyserini_collection \
      --index indexes/wiki100k_lucene \
      --generator DefaultLuceneDocumentGenerator \
      --threads 8
  ```
  Expected: `indexes/wiki100k_lucene/` directory with Lucene files

- [ ] **3A-4** Smoke test: 5 queries through real pipeline end-to-end:
  ```bash
  factuality-rag run \
      --query "What is the speed of light?" \
      --faiss-index indexes/wiki100k.faiss \
      --pyserini-index indexes/wiki100k_lucene \
      --k 10
  ```
  Verify: answer is non-mock, passages are real Wikipedia text, confidence tag is set

- [ ] **3A-5** Record indexing time and index sizes in notes below:
  ```
  Indexing time: _____ minutes
  FAISS index size: _____ MB
  Lucene index size: _____ MB
  Passages indexed: _____
  ```

---

### PHASE 3B — Baseline Experiments on NQ-Open Dev (500 examples)

> Run in order. Each experiment config is already created in `configs/`.

- [ ] **3B-1** Run **B1 (Closed-Book)** baseline:
  ```bash
  python -m factuality_rag.experiment_runner \
      --config configs/exp_b1_closed_book.yaml \
      --dataset natural_questions \
      --split validation \
      --sample 500 \
      --seed 42
  ```
  → Record EM, F1, FactScore in Results Table

- [ ] **3B-2** Run **B2 (Always-RAG)** baseline:
  ```bash
  python -m factuality_rag.experiment_runner \
      --config configs/exp_b2_always_rag.yaml \
      --dataset natural_questions \
      --split validation \
      --sample 500 \
      --seed 42
  ```
  → Record results. **Expected: B2 EM > B1 EM** (retrieval should help on NQ)

- [ ] **3B-3** Run **B3 (Gate-Only)** baseline — create config first:
  ```yaml
  # configs/exp_b3_gate_only.yaml
  # gate: true, scorer disabled (use all retrieved passages, no filtering)
  pipeline:
    gate: true
    score_threshold: 0.0   # accept all passages
  ```
  → Record results + retrieval call % 

- [ ] **3B-4** Run **B4 (Score-Only)** baseline — create config:
  ```yaml
  # configs/exp_b4_score_only.yaml
  # gate: false (always retrieve), scorer filters
  pipeline:
    gate: false
    score_threshold: 0.4
  ```
  → Record results + FactScore vs B2 (scoring should improve faithfulness)

- [ ] **3B-5** Run **Full Pipeline**:
  ```bash
  python -m factuality_rag.experiment_runner \
      --config configs/exp_full_pipeline.yaml \
      --dataset natural_questions \
      --split validation \
      --sample 500 \
      --seed 42
  ```
  → Record all metrics. **If Full < B2 on EM/F1, investigate immediately** (see Phase 4C)

- [ ] **3B-6** Replicate Full Pipeline on seeds 123 and 456:
  ```bash
  python -m factuality_rag.experiment_runner --config configs/exp_full_pipeline.yaml \
      --dataset natural_questions --split validation --sample 500 --seed 123
  python -m factuality_rag.experiment_runner --config configs/exp_full_pipeline.yaml \
      --dataset natural_questions --split validation --sample 500 --seed 456
  ```
  → Report mean ± std across 3 seeds in Results Table

---

### PHASE 3C — Additional Dataset Experiments

- [ ] **3C-1** Run Full Pipeline on **HotpotQA** dev (500 examples, seed 42)
  → Compare EM/F1 vs NQ-Open — expect lower (multi-hop is harder)
  → Note: scoring may filter out necessary supporting passages — document this

- [ ] **3C-2** Run Full Pipeline on **TruthfulQA** (all 817 MC examples)
  → Key metric: MC accuracy + hallucination rate
  → Expected: gating should skip retrieval often (TruthfulQA is adversarial parametric)

- [ ] **3C-3** Run Always-RAG (B2) on HotpotQA for comparison

---

### PHASE 4A — Gating Oracle Analysis (after Phase 3B)

- [ ] **4A-1** Write analysis script `scripts/analyze_gating.py`:
  ```python
  # Load predictions.jsonl from runs/
  # For each query: extract {query, gated_out (True/False), em_score}
  # Compute:
  #   - gating_skip_correct: queries where gate=skip AND em=1
  #   - gating_retrieve_needed: queries where gate=retrieve AND closed-book em=0
  #   - false_skip: gate=skip BUT em=0 (model was overconfident)
  #   - wasted_retrieve: gate=retrieve BUT closed-book em=1 (unnecessary retrieval)
  ```

- [ ] **4A-2** Report gating statistics:
  ```
  Total queries: 500
  Gating triggered retrieval: ___% (target: 40-70%)
  Of skip decisions: ___% were correct (model already knew)
  Of retrieve decisions: ___% were necessary (model wouldn't have known)
  False skip rate (overconfident hallucinations): ___%
  Wasted retrieve rate: ___%
  ```

- [ ] **4A-3** If false skip rate > 15%, lower `entropy_thresh` from 1.2 → 1.0
  If wasted retrieve rate > 40%, raise `entropy_thresh` from 1.2 → 1.5

- [ ] **4A-4** Generate precision-recall curve for gating decisions → save as `figures/gating_pr_curve.png`
  (x-axis: recall of queries needing retrieval; y-axis: precision; threshold sweep)

---

### PHASE 4B — Scorer AUC Analysis (after Phase 3B)

- [ ] **4B-1** Write `scripts/analyze_scorer.py`:
  - Load NQ-Open dev with gold passage annotations
  - For each retrieved passage in predictions, label: 1 if passage id matches gold, 0 otherwise
  - Compute AUC-ROC of `final_score` (relevant vs irrelevant passages)

- [ ] **4B-2** Report scorer AUC:
  ```
  Scorer AUC-ROC: _____
  Baseline (random): 0.50
  Target (good scorer): > 0.70
  ```

- [ ] **4B-3** If AUC < 0.65:
  - Re-check NLI model is loaded correctly (not in fallback mode)
  - Try dropping w_overlap to 0.0 and see if AUC improves
  - Consider switching to sentence-level NLI (see Phase 5A)

- [ ] **4B-4** Compute NLI score distribution for relevant vs irrelevant passages — histogram
  → Save as `figures/nli_score_distribution.png`

---

### PHASE 4C — Error Analysis (50 failure cases)

- [ ] **4C-1** Sample 50 queries from Full Pipeline run where EM=0
- [ ] **4C-2** For each failure, classify using taxonomy:
  | Code | Meaning |
  |------|---------|
  | `GATE_MISS` | Gate skipped retrieval but model hallucinated |
  | `RETRIEVAL_MISS` | Correct passage not in top-10 |
  | `SCORER_DROP` | Correct passage retrieved but `final_score < 0.4` (filtered out) |
  | `GEN_IGNORE` | Correct passage provided but Mistral ignored it |
  | `ANSWER_FORMAT` | Answer correct semantically but EM failed (normalisation issue) |
  | `CORPUS_GAP` | Answer not in 100K Wikipedia subset |

- [ ] **4C-3** Tally error codes:
  ```
  GATE_MISS:       ___ / 50
  RETRIEVAL_MISS:  ___ / 50
  SCORER_DROP:     ___ / 50
  GEN_IGNORE:      ___ / 50
  ANSWER_FORMAT:   ___ / 50
  CORPUS_GAP:      ___ / 50
  ```

- [ ] **4C-4** Act on findings:
  - If SCORER_DROP > 10: lower `score_threshold` from 0.4 → 0.3
  - If RETRIEVAL_MISS > 15: increase `top_k` from 10 → 20 or add cross-encoder (Phase 5B)
  - If GATE_MISS > 8: lower gating entropy threshold
  - If GEN_IGNORE > 10: improve prompt template (add instruction to cite passages)
  - If CORPUS_GAP > 10: expand corpus or note as limitation

---

### PHASE 5A — Scorer Improvements (P1)

#### 5A-1: Sentence-Level NLI
Current NLI scores entire passage as one unit. Long passages may dilute a supporting sentence.

- [ ] **5A-1a** Add `_sentence_level_nli(query, passage_text)` to `scorer/passage.py`:
  ```python
  def _sentence_level_nli(self, query: str, passage_text: str) -> float:
      sentences = self._split_sentences(passage_text)  # use regex or spaCy
      if not sentences:
          return self._nli_entailment(premise=passage_text, hypothesis=query)
      scores = [self._nli_entailment(premise=s, hypothesis=query) for s in sentences]
      return max(scores)  # passage score = best-supporting sentence
  ```
- [ ] **5A-1b** Add config option: `scorer.nli_mode: "passage" | "sentence"` in YAML
- [ ] **5A-1c** Add test: sentence-level NLI returns higher score than passage-level when supporting sentence is surrounded by irrelevant text
- [ ] **5A-1d** Run ablation: compare AUC-ROC with passage-level vs sentence-level NLI

#### 5A-2: Cross-Encoder Reranking (optional stage before NLI scoring)
- [ ] **5A-2a** Add optional cross-encoder step in `scorer/passage.py.__init__`:
  ```python
  self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2") if use_cross_encoder else None
  ```
- [ ] **5A-2b** If `cross_encoder` is not None, re-sort passages by cross-encoder score before NLI scoring
- [ ] **5A-2c** Add YAML config key: `scorer.use_cross_encoder: false` (default off)
- [ ] **5A-2d** Run ablation: Exp 4.7 from experiment plan — does cross-encoder help?

#### 5A-3: Learned Fusion Weights
- [ ] **5A-3a** After Phase 3B experiments, run scorer weight grid search on NQ-Open dev:
  - Grid: `w_nli ∈ {0.3, 0.4, 0.5, 0.6, 0.7}` × `w_ret ∈ {0.1, 0.2, 0.3}` (w_overlap = 1 - w_nli - w_ret)
  - Metric to optimise: FactScore on NQ dev
- [ ] **5A-3b** Add `scripts/tune_scorer_weights.py` for grid search
- [ ] **5A-3c** Update `configs/exp_full_pipeline.yaml` with best weights found

---

### PHASE 5B — Gating Probe Improvements (P1)

#### 5B-1: Real ECE Calibration (BUG-6)
- [ ] **5B-1a** Implement proper binned ECE in `gating/probe.py`:
  ```python
  def compute_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
      bin_edges = np.linspace(0, 1, n_bins + 1)
      ece = 0.0
      for i in range(n_bins):
          mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
          if mask.sum() > 0:
              ece += mask.sum() * abs(confidences[mask].mean() - accuracies[mask].mean())
      return ece / len(confidences)
  ```
- [ ] **5B-1b** Update `calibrate_temperature()` to use `compute_ece()` instead of entropy std-dev proxy
- [ ] **5B-1c** Add test for `compute_ece()` with known perfect calibration case → ECE = 0.0

#### 5B-2: Multi-Token Probe
- [ ] **5B-2a** Add `probe_tokens` parameter to `should_retrieve()` (already exists — verify it's used correctly)
- [ ] **5B-2b** When `probe_tokens > 1`, average entropy over first k token positions
- [ ] **5B-2c** Run ablation: probe_tokens ∈ {1, 2, 3, 5} vs gating accuracy

---

### PHASE 5C — Sensitivity Ablations (from Experiment Plan)

Run these only after Phase 3B baselines are complete. Each is a single YAML change + experiment run.

- [ ] **5C-1** Alpha sweep — vary `retriever.alpha ∈ {0.3, 0.5, 0.6, 0.7, 0.9}` on NQ dev 500
  → Plot EM/F1/FactScore vs alpha → save `figures/alpha_sweep.png`

- [ ] **5C-2** Score threshold sweep — vary `scorer.score_threshold ∈ {0.2, 0.3, 0.4, 0.5, 0.6}` on NQ dev 500
  → Plot precision/recall tradeoff → save `figures/threshold_sweep.png`

- [ ] **5C-3** Top-K sweep — vary `retriever.top_k ∈ {3, 5, 10, 15, 20}` on NQ dev 500
  → Plot retrieval recall@K + EM → save `figures/topk_sweep.png`

- [ ] **5C-4** Gating threshold sweep:
  - `gating.entropy_thresh ∈ {0.8, 1.0, 1.2, 1.5, 2.0}`
  - `gating.logit_gap_thresh ∈ {1.0, 1.5, 2.0, 3.0}`
  → 4×5 grid; plot retrieval% vs EM tradeoff

---

### PHASE 6 — Integration Tests (GPU required)

- [ ] **6-1** Integration test: generator produces non-mock output on real query (1A-6 from Session 2)
- [ ] **6-2** Integration test: NLI entailment > 0.7 for known support pair (2A-4)
- [ ] **6-3** Integration test: NLI entailment < 0.3 for known contradiction pair (2A-5)
- [ ] **6-4** Integration test: BM25 returns non-random scores with real Lucene index (1D-5)
- [ ] **6-5** Integration test: full Pipeline loads Mistral-7B exactly once (model_registry check)
- [ ] **6-6** Verify GitHub Actions CI passes on clean clone: push to remote and check Actions tab (5-4)

---

## 🐛 OPEN BUGS

| ID | File | Bug | Severity | Status |
|----|------|-----|----------|--------|
| BUG-6 | `gating/probe.py` | ECE calibration uses std-dev proxy, not real binned ECE | 🟢 Low | [ ] Fix in Phase 5B-1 |
| BUG-7 | `eval/metrics.py` | `decompose_claims()` misses compound claims with "and" (rule-based limit) | 🟢 Low | [ ] Document as limitation; LLM decomposition is P2 |
| BUG-8 | `pipeline/orchestrator.py` | `provenance` dict is populated with mock structure — real claim→passage mapping not implemented | 🟡 Medium | [ ] Fix when real FactScore is exercised |

---

## 📐 IMPLEMENTATION CONSTRAINTS (unchanged)

1. Every real implementation must keep `mock_mode=True` path working — all 53 tests must still pass in < 2 seconds
2. All tunables stay in YAML config — no hardcoded thresholds in source code
3. Seed everything at pipeline init: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`
4. Every experiment saves `predictions.jsonl` + `metrics.json` + `metadata.json` to `runs/<run-id>/`
5. No proprietary APIs — all models loadable from HuggingFace locally
6. Windows 11 compatibility — avoid Linux-only shell syntax

---

## 🔧 ENVIRONMENT REFERENCE

```
OS:           Windows 11
Python:       3.13.5 (Anaconda)
Working dir:  E:\Lab\NLP\Faculty-first RAG\
GPU:          NVIDIA A100-80GB

Installed:
  faiss-cpu              1.13.2
  transformers           5.2.0
  sentence-transformers  5.2.3
  datasets               4.6.1
  pyserini               1.5.0
  scikit-learn           1.6.1
  pytest                 8.3.4
  bitsandbytes           (installed in Session 2)

Models:
  Generator:     mistralai/Mistral-7B-Instruct-v0.3 (4-bit quant)
  NLI Scorer:    ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
  Embedder:      sentence-transformers/all-mpnet-base-v2
  Cross-encoder: cross-encoder/ms-marco-MiniLM-L-12-v2 (optional, Phase 5A)

Index paths (after Phase 3A):
  FAISS:   indexes/wiki100k.faiss
  IDs:     indexes/wiki100k.ids.json
  Lucene:  indexes/wiki100k_lucene/
  Chunks:  data/wiki_100k_chunks.jsonl
```

---

## 📊 RESULTS TABLE

| Exp ID | Config | Dataset | N | Seeds | EM (mean±std) | F1 (mean±std) | FactScore | Retrieval% | Latency ms/q | Date |
|--------|--------|---------|---|-------|--------------|--------------|-----------|------------|--------------|------|
| B1 | closed-book | NQ dev | 500 | 42 | — | — | — | 0% | — | |
| B2 | always-RAG | NQ dev | 500 | 42 | — | — | — | 100% | — | |
| B3 | gate-only | NQ dev | 500 | 42 | — | — | — | —% | — | |
| B4 | score-only | NQ dev | 500 | 42 | — | — | — | 100% | — | |
| FULL | α=0.6, thr=0.4 | NQ dev | 500 | 42,123,456 | — | — | — | —% | — | |
| FULL | α=0.6, thr=0.4 | HotpotQA dev | 500 | 42 | — | — | — | —% | — | |
| FULL | α=0.6, thr=0.4 | TruthfulQA | 817 | 42 | — | — | — | —% | — | |

**Gating analysis (fill after Phase 4A):**
```
Retrieval trigger rate:     ___%
Gate-skip accuracy:         ___%  (of skips, how many were correct)
False skip rate:            ___%  (overconfident hallucinations)
Wasted retrieval rate:      ___%
```

**Scorer analysis (fill after Phase 4B):**
```
Scorer AUC-ROC:            _____
NLI-only AUC:              _____
Retriever-score-only AUC:  _____
```

---

## 🗂️ SESSION HANDOFF NOTES

**Session 1 (2026-02-28):** Full mock package bootstrap — 36 tests, all 9 modules
**Session 2 (2026-02-28):** Real components wired — generator, BM25, Wikipedia, Pipeline class, FactScore, CI/CD — 53 tests

**Session 3 goal:** First real numbers. Build indexes → run baselines → run full pipeline → analyze where it breaks.

**Expected Session 3 duration:** 1 full day (index building ~2-3 hrs, experiment runs ~4-5 hrs, analysis ~2 hrs)

**Biggest risk for Session 3:** If B2 (always-RAG) significantly outperforms Full Pipeline on EM/F1, the scorer may be filtering too aggressively. First fix: lower `score_threshold` to 0.3 and re-run. Second fix: sentence-level NLI (Phase 5A-1).

**Second biggest risk:** Corpus gap — if 100K Wikipedia subset doesn't cover NQ answers, retrieval recall will be low regardless of scorer quality. Mitigation: check retrieval recall@10 first before blaming scorer.

---

## 💬 HOW TO ASK FOR HELP

```
"Implement task [TASK-ID]. Current code for [file]:
[paste code]
Constraints: keep mock_mode working, all config in YAML, Windows 11 paths.
Show updated test after implementing."

"Error on [TASK-ID]: [error message]
Code: [paste]
Fix without breaking mock mode."

"I got these experiment results: [paste Results Table row].
Full pipeline EM is [X]% vs Always-RAG [Y]%. Explain what's wrong and what to check."
```

---
*Session 3 prompt — v1.0 — 2026-02-28*
