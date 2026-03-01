# Status Tracker — Session 4
> Factuality-First RAG · v0.3.0+ · First Real Experiments + Conference Expansion
> Update this file as tasks complete. Results table fills in live.
> **Last updated:** 2026-03-01 (Session 4, turn 2)

---

## Legend
`[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[s]` skipped

---

## SESSION 4 — PROGRESS SUMMARY

| Category | Items done | Items remaining | Notes |
|----------|:----------:|:---------------:|-------|
| Phase A: Indexes | 8/8 | 0 | FAISS + Lucene built, 544,953 passages |
| Infrastructure | 10/10 | 0 | 8 bug fixes, 5 Pipeline bugs, model download |
| New Features | 4/4 | 0 | Learned scorer, 3 datasets, 5 configs |
| Phase B: Experiments | 0/11 | 11 | Blocked on first real run |
| Phase C: Analysis | 0/7 | 7 | Depends on Phase B |
| Phase D: Ablations | 0/7 | 7 | Depends on Phase C |
| Phase E: Parallel | 0/9 | 9 | Can start after B5 |
| Tests | 94 pass | 7 integration deselected | All unit tests green |

---

## PHASE A — BUILD INDEXES  [x] COMPLETE
> All tasks verified and done.

| # | Task | Done? | Notes |
|---|------|-------|-------|
| A1 | Ingest 100K Wikipedia articles | [x] | `data/wiki_100k_chunks.jsonl` |
| A2 | Verify JSONL output | [x] | 544,953 passages |
| A3 | Build FAISS HNSW index | [x] | 3:25:32 build time |
| A4 | Verify FAISS index | [x] | 544,953 vectors, dim=768 |
| A5 | Prepare Pyserini collection | [x] | JSONL → Lucene input |
| A6 | Build Lucene BM25 index | [x] | 20 segments |
| A7 | Verify BM25 | [x] | Real passage text returned |
| A8 | End-to-end smoke test | [x] | "Who invented the telephone?" → Bell passages |

**Index build stats:**
```
Passages indexed:     544,953
FAISS build time:     205 min (3:25:32)
Lucene build time:    ~5 min
FAISS index size:     1,562 MB (1.56 GB)
Lucene index size:    ~800 MB (20 segment files)
```

---

## INFRASTRUCTURE FIXES  [x] COMPLETE
> 8 code audit bugs + 5 Pipeline architecture bugs, all fixed.

| # | Fix | Commit | Notes |
|---|-----|--------|-------|
| 1 | BUG-1: JSONL→FAISS index path | d1243b4 | Fixed path resolution |
| 2 | BUG-2: NLI argument order | d1243b4 | premise=passage, hypothesis=query |
| 3 | BUG-3: Retriever caching | d1243b4 | Fixed cache key collision |
| 4 | BUG-4: Wikipedia dataset name | d1243b4 | Correct HF dataset ID |
| 5 | BUG-5: NQ dataset config | d1243b4 | Use nq_open directly |
| 6 | BUG-6: JAVA_HOME detection | d1243b4 | Auto-detect JDK path |
| 7 | Pipeline Bug A: config propagation | d1243b4 | `run_pipeline()` accepts config dict |
| 8 | Pipeline Bug B: closed-book crash | d1243b4 | Skip retriever when top_k=0 |
| 9 | Pipeline Bug E: override loss | d1243b4 | Pipeline.__init__ accepts config dict |
| 10 | Pipeline Bug F: wrong paths | d1243b4 | exp_sample.yaml fixed |
| 11 | Pipeline Bug G: falsy 0.0 | d1243b4 | `is not None` checks |

---

## NEW FEATURES (Conference Expansion)  [x] COMPLETE
> Added in commit f1f0f2a.

| # | Feature | File | Status | Tests |
|---|---------|------|--------|-------|
| 1 | Learned scorer (LogReg + MLP) | `factuality_rag/scorer/learned_scorer.py` | [x] | 15 tests |
| 2 | Scorer training script | `scripts/train_scorer.py` | [x] | Manual |
| 3 | Pipeline integration (use_learned) | `factuality_rag/pipeline/orchestrator.py` | [x] | Covered by existing |
| 4 | PopQA dataset | `factuality_rag/data/loader.py` | [x] | Manual verified |
| 5 | HAGRID dataset | `factuality_rag/data/loader.py` | [x] | Manual verified |
| 6 | 2WikiMultiHopQA dataset | `factuality_rag/data/loader.py` | [x] | Manual verified |
| 7 | FEVER loader fix | `factuality_rag/data/loader.py` | [x] | — |
| 8 | Experiment runner multi-dataset | `factuality_rag/experiment_runner.py` | [x] | — |

**New experiment configs:**

| Config | Dataset | Purpose |
|--------|---------|---------|
| `exp_popqa.yaml` | PopQA (14,267 test) | Tail-entity single-hop QA |
| `exp_hagrid.yaml` | HAGRID (716 val) | Attribution/grounded QA |
| `exp_2wiki.yaml` | 2WikiMultiHopQA (12,576 val) | Multi-hop reasoning |
| `exp_fever.yaml` | FEVER (78,947 val) | Claim verification |
| `exp_b5_learned_scorer.yaml` | NQ-Open (validation) | Learned vs hand-tuned scorer |

---

## MODEL DOWNLOADS  [x] COMPLETE

| Model | Size | Location | Status |
|-------|------|----------|--------|
| Mistral-7B-Instruct-v0.3 | 13.5 GB | HF cache | [x] Downloaded |
| all-mpnet-base-v2 | 420 MB | HF cache | [x] Cached |
| RoBERTa-large NLI | ~1.4 GB | HF cache | [ ] Not cached yet (first use downloads) |

---

## PHASE B — BASELINE EXPERIMENTS

### Phase B-I: NQ-Open Dev (n=500) — Core Baselines

#### B1 — Closed-Book

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B1-a | Create run: `--run-id b1_nq_500_s42` | [ ] | ~25 min |
| B1-b | Record metrics in Results Table | [ ] | Expected EM: 0.20–0.35 |

#### B2 — Always-RAG

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B2-a | Create run: `--run-id b2_nq_500_s42` | [ ] | ~90 min |
| B2-b | Verify B2.EM > B1.EM (if not — STOP and diagnose) | [ ] | |
| B2-c | Record metrics | [ ] | |

#### B3 — Gate-Only

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B3-a | Create run: `--run-id b3_nq_500_s42` | [ ] | |
| B3-b | Check `retrieval_rate` in 40–70% | [ ] | If <30%: lower entropy_thresh; if >85%: raise it |
| B3-c | Record metrics | [ ] | |

#### B4 — Score-Only

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B4-a | Create run: `--run-id b4_nq_500_s42` | [ ] | |
| B4-b | Verify B4.FactScore > B2.FactScore | [ ] | Scorer should improve faithfulness |
| B4-c | Verify B4.EM >= B2.EM - 3pp | [ ] | If bigger gap: lower threshold to 0.3 |
| B4-d | Record metrics | [ ] | |

#### B5 — Full Pipeline (3 seeds)

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B5-a | Seed 42 run: `--run-id full_nq_500_s42` | [ ] | ~90 min |
| B5-b | Seed 123 run: `--run-id full_nq_500_s123` | [ ] | ~90 min |
| B5-c | Seed 456 run: `--run-id full_nq_500_s456` | [ ] | ~90 min |
| B5-d | Aggregate across seeds: `python scripts/aggregate_results.py` | [ ] | |
| B5-e | Zero-passage check: confirm <20% queries have no trusted passages | [ ] | |
| B5-f | Record mean +/- std in Results Table | [ ] | |

### Phase B-II: Cross-Dataset Experiments (after NQ-Open stable)

#### B6 — Original Datasets

| # | Task | Done? | Notes |
|---|------|-------|-------|
| B6-a | HotpotQA dev 500, seed 42 | [ ] | Multi-hop, expect lower EM |
| B6-b | TruthfulQA all 817, seed 42 | [ ] | Anti-hallucination |
| B6-c | Always-RAG on HotpotQA (for comparison) | [ ] | |

#### B7 — New Datasets (Conference Expansion)

| # | Task | Config | Done? | Notes |
|---|------|--------|-------|-------|
| B7-a | PopQA test 500, seed 42 | `exp_popqa.yaml` | [ ] | Tail-entity QA, 14,267 total |
| B7-b | HAGRID val all (716), seed 42 | `exp_hagrid.yaml` | [ ] | Attribution/grounded QA |
| B7-c | 2WikiMultiHopQA val 500, seed 42 | `exp_2wiki.yaml` | [ ] | Multi-hop reasoning |
| B7-d | FEVER val 500, seed 42 | `exp_fever.yaml` | [ ] | Claim verification |

#### B8 — Learned Scorer Comparison

| # | Task | Config | Done? | Notes |
|---|------|--------|-------|-------|
| B8-a | Train learned scorer on FEVER | `scripts/train_scorer.py` | [ ] | Must run before B8-b |
| B8-b | NQ-Open 500 with learned scorer | `exp_b5_learned_scorer.yaml` | [ ] | Compare vs B5 hand-tuned |
| B8-c | PopQA 500 with learned scorer | override `use_learned: true` | [ ] | Cross-dataset generalization |

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
| **Full** | seed 42 | | | | | | — |
| **Full** | seed 123 | | | | | | — |
| **Full** | seed 456 | | | | | | — |
| **Full** | **mean +/- std** | | | | | | — |
| **B5-learned** | learned scorer (logreg) | | | | | | vs Full |
| Self-RAG | selfrag_llama2_7b | | | | | | |

### Cross-Dataset (Full Pipeline, seed 42)

| Dataset | n | EM | F1 | FactScore | Ret% | Notes |
|---------|:-:|:--:|:--:|:---------:|:----:|-------|
| NQ-Open dev | 500 | | | | | Single-hop factoid |
| HotpotQA dev | 500 | | | | | Multi-hop |
| TruthfulQA val | 817 | | | | | Anti-hallucination |
| PopQA test | 500 | | | | | Tail-entity |
| HAGRID val | 716 | | | | | Attribution |
| 2WikiMultiHopQA val | 500 | | | | | Multi-hop reasoning |
| FEVER val | 500 | | | | | Claim verification |

### Learned Scorer vs Hand-Tuned (seed 42)

| Dataset | Scorer | EM | F1 | FactScore | Ret% |
|---------|--------|:--:|:--:|:---------:|:----:|
| NQ-Open | hand-tuned (0.5/0.2/0.3) | | | | |
| NQ-Open | learned (logreg) | | | | |
| NQ-Open | learned (mlp) | | | | |
| PopQA | hand-tuned (0.5/0.2/0.3) | | | | |
| PopQA | learned (logreg) | | | | |

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
| 1 | Drop TriviaQA, add PopQA/HAGRID/2WikiMultiHopQA | Redundant with NQ-Open; new datasets add breadth for conf paper | 7 datasets total |
| 2 | Implement learned scorer (LogReg + MLP) | Hand-tuned 0.5/0.2/0.3 is not a learned contribution | New module + training script |
| 3 | Use FEVER train for scorer training | 311K labelled claim-evidence pairs, natural supervision | Synthetic feature mode for speed |
| 4 | Mistral-7B with XET disabled | XET protocol caused zero-byte downloads | 14.5GB download completed |
| 5 | Disable MLP early_stopping | 6-sample test data too small for validation split | early_stopping=False |
| 6 | Use datasets parquet revision | datasets v4.6.1 removed loading scripts for FEVER/HAGRID/2Wiki | revision='refs/convert/parquet' |
| 7 | Skip ci.yml in push | PAT lacks `workflow` scope | ci.yml staged locally, not pushed |

---

## NOTES / OBSERVATIONS

> Free-form notes as experiments run. Record surprises, unexpected results, ideas.

```
[2026-02-28] Session 4 start. Found 8 code bugs during audit, all fixed.
[2026-02-28] Pipeline had 5 architecture bugs (A/B/E/F/G), all fixed.
[2026-02-28] FAISS index: 544,953 vectors built in 3:25:32.
[2026-02-28] Lucene index: 544,953 docs, 20 segments.
[2026-02-28] Mistral download stuck with XET → disabled XET, completed 14.5GB.
[2026-02-28] GPU: RTX 4060 Laptop 8GB (not A100 as prompt assumed). Will need 4-bit.
[2026-02-28] Python 3.13.5 with PyTorch 2.10.0+cu126 works.
[2026-03-01] Added PopQA (14,267), HAGRID (2,638), 2WikiMultiHopQA (12,576).
[2026-03-01] Learned scorer module created: LogReg + MLP over (nli, overlap, ret_norm).
[2026-03-01] 94 tests passing. Commit f1f0f2a pushed.
[2026-03-01] RoBERTa NLI model not yet cached — first real experiment will download it.
```

---

## GIT HISTORY (Session 4)

| Commit | Message | Files |
|--------|---------|-------|
| `d1243b4` | fix Pipeline bugs, build indexes, prepare for experiments | 16 |
| `f1f0f2a` | add learned scorer, 3 new datasets, 5 new experiment configs | 13 |

---

*Status Tracker · Session 4 · v2.0 · 2026-03-01*
