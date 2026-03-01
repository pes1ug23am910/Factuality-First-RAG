# Status Tracker — Session 4
> Factuality-First RAG · v0.3.0 · First Real Experiments
> Update this file as tasks complete. Results table fills in live.

---

## Legend
`[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[s]` skipped

---

## PHASE A — BUILD INDEXES
> Prerequisite for everything. Do this before any experiment.

| # | Task | Command | Done? | Notes |
|---|------|---------|-------|-------|
| A1 | Ingest 100K Wikipedia articles | `python scripts/build_corpus.py --sample-size 100000` | [ ] | Target: 300K–450K passages |
| A2 | Verify JSONL output | Count lines in `data/wiki_100k_chunks.jsonl` | [ ] | |
| A3 | Build FAISS HNSW index | `factuality-rag build-index --faiss-out indexes/wiki100k.faiss` | [ ] | |
| A4 | Verify FAISS index | `idx.ntotal == len(ids)`, `idx.d == 768` | [ ] | |
| A5 | Prepare Pyserini collection | `factuality-rag build-index --pyserini-out ...` | [ ] | |
| A6 | Build Lucene BM25 index | `python -m pyserini.index.lucene ...` | [ ] | |
| A7 | Verify BM25 | `LuceneSearcher` returns real passage text | [ ] | |
| A8 | End-to-end smoke test | `factuality-rag run --query "Who invented the telephone?"` | [ ] | Answer must not be "Mock..." |

**Index build stats (fill in):**
```
Passages indexed:     _______
FAISS build time:     _______ min
Lucene build time:    _______ min
FAISS index size:     _______ MB
Lucene index size:    _______ MB
```

---

## PHASE B — BASELINE EXPERIMENTS (NQ-Open dev, n=500)

### B1 — Closed-Book

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B1-a | Create run: `--run-id b1_nq_500_s42` | [ ] | |
| B1-b | Record metrics in Results Table | [ ] | |

### B2 — Always-RAG

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B2-a | Create run: `--run-id b2_nq_500_s42` | [ ] | |
| B2-b | Verify B2.EM > B1.EM (if not — STOP and diagnose) | [ ] | |
| B2-c | Record metrics | [ ] | |

### B3 — Gate-Only

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B3-a | Create run: `--run-id b3_nq_500_s42` | [ ] | |
| B3-b | Check `retrieval_rate` ∈ 40–70% | [ ] | If <30%: lower entropy_thresh; if >85%: raise it |
| B3-c | Record metrics | [ ] | |

### B4 — Score-Only

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B4-a | Create run: `--run-id b4_nq_500_s42` | [ ] | |
| B4-b | Verify B4.FactScore > B2.FactScore | [ ] | Scorer should improve faithfulness |
| B4-c | Verify B4.EM ≥ B2.EM − 3pp | [ ] | If bigger gap: lower threshold to 0.3 |
| B4-d | Record metrics | [ ] | |

### B5 — Full Pipeline (3 seeds)

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B5-a | Seed 42 run: `--run-id full_nq_500_s42` | [ ] | |
| B5-b | Seed 123 run: `--run-id full_nq_500_s123` | [ ] | |
| B5-c | Seed 456 run: `--run-id full_nq_500_s456` | [ ] | |
| B5-d | Aggregate across seeds: `python scripts/aggregate_results.py` | [ ] | |
| B5-e | Zero-passage check: confirm <20% queries have no trusted passages | [ ] | |
| B5-f | Record mean ± std in Results Table | [ ] | |

### B6 — Additional Datasets (after NQ-Open stable)

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B6-a | HotpotQA dev 500, seed 42 | [ ] | Expect lower EM (multi-hop is harder) |
| B6-b | TruthfulQA all 817, seed 42 | [ ] | Expect high retrieval skip rate |
| B6-c | Always-RAG on HotpotQA (for comparison) | [ ] | |

---

## PHASE C — ANALYSIS

| # | Task | Script | Done? | Result |
|---|------|--------|-------|--------|
| C1 | Aggregate cross-seed metrics | `aggregate_results.py` | [ ] | |
| C2 | Gating oracle analysis | `analyze_gating.py` | [ ] | See gating stats box below |
| C3 | Scorer AUC analysis | `analyze_scorer.py` | [ ] | See scorer stats box below |
| C4 | Error taxonomy (50 failures) | `analyze_errors.py` | [ ] | See error table below |
| C5 | Bootstrap: Full vs B2 FactScore | `bootstrap_test.py` | [ ] | p = _____ |
| C6 | Bootstrap: Full vs B4 EM | `bootstrap_test.py` | [ ] | p = _____ |
| C7 | Bootstrap: Full vs B3 retrieval_rate | `bootstrap_test.py` | [ ] | p = _____ |

**Gating analysis results (fill after C2):**
```
Retrieval trigger rate:     _____%
Gate-skip accuracy:         _____%   (skips where closed-book EM=1)
False-skip rate:            _____%   (target: <15%; if exceeded, lower entropy_thresh)
Wasted-retrieval rate:      _____%   (target: <40%)
```

**Scorer analysis results (fill after C3):**
```
Full scorer AUC-ROC:        _____   (target: >0.70)
NLI-only AUC:               _____
Overlap-only AUC:           _____
Retriever-score-only AUC:   _____
```

**Error taxonomy (fill after C4, n=50 failures):**
```
GATE_MISS:       __ / 50   (gated skip but model hallucinated)
RETRIEVAL_MISS:  __ / 50   (correct passage not in top-10)
SCORER_DROP:     __ / 50   (correct passage filtered by scorer)
GEN_IGNORE:      __ / 50   (Mistral ignored correct passage)
ANSWER_FORMAT:   __ / 50   (EM normalisation failure)
CORPUS_GAP:      __ / 50   (answer not in 100K subset)
```

**Actions triggered by error analysis:**
- [ ] If GATE_MISS > 10: lower `entropy_threshold` from 1.2 → 1.0, add to ablation D4
- [ ] If RETRIEVAL_MISS > 15: increase `top_k` from 10 → 20, add to ablation D3
- [ ] If SCORER_DROP > 10: lower `score_threshold` from 0.4 → 0.3, run D1; also run D6 (sentence NLI)
- [ ] If GEN_IGNORE > 10: run Phase E2 prompt variants immediately
- [ ] If ANSWER_FORMAT > 5: fix EM normalisation in `eval/metrics.py`
- [ ] If CORPUS_GAP > 10: note as limitation; consider expanding to 250K articles

---

## PHASE D — SENSITIVITY ABLATIONS

> Run one at a time. Each takes ~90 min. Prioritise based on C4 error analysis.

| # | Ablation | Values | Status | Best value |
|---|----------|--------|--------|------------|
| D1 | Score threshold sweep | 0.2, 0.3, 0.4, 0.5, 0.6 | [ ] | |
| D2 | Alpha sweep (dense/sparse fusion) | 0.3, 0.5, 0.6, 0.7, 0.9 | [ ] | |
| D3 | Top-K sweep | 3, 5, 10, 15, 20 | [ ] | |
| D4 | Entropy threshold sweep | 0.8, 1.0, 1.2, 1.5, 2.0 | [ ] | |
| D5 | Scorer weight tuning (on FEVER dev) | grid (w_nli, w_ret) | [ ] | |
| D6 | NLI mode: passage vs sentence | passage, sentence | [ ] | |
| D7 | Cross-encoder ablation | off vs ms-marco-MiniLM | [ ] | |

**Ablation priority order** (set based on error analysis in C4):
1. _____________  (most impactful finding from C4)
2. _____________
3. _____________

---

## PHASE E — PARALLEL WORK

| # | Task | Status | Target date |
|---|------|--------|-------------|
| E1 | Sample 300 queries for human eval (stratified by confidence tag) | [ ] | After B5 |
| E2 | Set up annotation tool (Label Studio or Google Forms) | [ ] | |
| E3 | Create annotation guidelines document | [ ] | |
| E4 | Complete first 50 annotations + compute Cohen's κ | [ ] | κ target: >0.7 |
| E5 | Complete remaining 250 annotations | [ ] | |
| E6 | Prompt Variant A vs B vs C test (100 NQ queries) | [ ] | After B5 |
| E7 | Choose best prompt variant, update `exp_full_pipeline.yaml` | [ ] | After E6 |
| E8 | Self-RAG baseline: load `selfrag/selfrag_llama2_7b`, run on NQ dev 500 | [ ] | |
| E9 | Contriever encoder comparison (requires FAISS rebuild with new encoder) | [ ] | Optional |

---

## RESULTS TABLE

Fill in immediately after each experiment completes.

### NQ-Open Dev (n=500)

| Exp ID | Config | EM | F1 | FactScore | Ret% | Latency ms/q | p vs Full |
|--------|--------|:--:|:--:|:---------:|:----:|:------------:|:---------:|
| B1 | closed-book | | | | 0% | | — |
| B2 | always-RAG | | | | 100% | | |
| B3 | gate-only | | | | | | |
| B4 | score-only | | | | 100% | | |
| **Full** | α=0.6, thr=0.4 (seed 42) | | | | | | — |
| **Full** | α=0.6, thr=0.4 (seed 123) | | | | | | — |
| **Full** | α=0.6, thr=0.4 (seed 456) | | | | | | — |
| **Full** | **mean ± std** | | | | | | — |
| Self-RAG† | selfrag_llama2_7b | | | | | | |

### Cross-Dataset (Full Pipeline only, seed 42)

| Dataset | EM | F1 | FactScore | Ret% |
|---------|:--:|:--:|:---------:|:----:|
| NQ-Open dev 500 | | | | |
| HotpotQA dev 500 | | | | |
| TruthfulQA all 817 | | | | |

### Ablation Results (NQ-Open dev 500, seed 42, vs Full baseline)

| Ablation | Value | EM | FactScore | Ret% | Note |
|----------|-------|:--:|:---------:|:----:|------|
| Threshold | 0.2 | | | | |
| Threshold | 0.3 | | | | |
| **Threshold** | **0.4 (default)** | | | | baseline |
| Threshold | 0.5 | | | | |
| Threshold | 0.6 | | | | |
| Alpha | 0.3 | | | | |
| Alpha | 0.5 | | | | |
| **Alpha** | **0.6 (default)** | | | | baseline |
| Alpha | 0.7 | | | | |
| Alpha | 0.9 | | | | |
| Top-K | 3 | | | | |
| Top-K | 5 | | | | |
| **Top-K** | **10 (default)** | | | | baseline |
| Top-K | 15 | | | | |
| Top-K | 20 | | | | |
| NLI mode | passage | | | | baseline |
| NLI mode | sentence | | | | |
| Cross-encoder | off | | | | baseline |
| Cross-encoder | ms-marco-MiniLM | | | | |

---

## OPEN BUGS

| ID | File | Issue | Priority | Status |
|----|------|-------|----------|--------|
| BUG-7 | `eval/metrics.py` | `decompose_claims()` misses compound "and" claims | 🟢 Low | Fix only if error analysis shows it matters |

---

## DECISION LOG

> Record key decisions made during this session here so future sessions have context.

| # | Decision | Reason | Outcome |
|---|----------|--------|---------|
| | | | |
| | | | |

---

## NOTES / OBSERVATIONS

> Free-form notes as experiments run. Record surprises, unexpected results, ideas.

```
[date/time] _______________________
[date/time] _______________________
[date/time] _______________________
```

---

*Status Tracker · Session 4 · v1.0 · 2026-02-28*
