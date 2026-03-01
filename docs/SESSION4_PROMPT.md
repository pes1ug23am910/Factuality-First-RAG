# Factuality-First RAG — Session 4 Handoff Prompt
> Paste this entire document as your first message.
> Version: v0.3.0 | Tests: 79 passing (+ 7 integration deselected) | Sessions completed: 3

---

## WHAT THIS SESSION IS

**Session 4 is the first real-data session.** The code is done. All components are implemented, tested, and documented across 3 prior sessions. The only thing missing is actual numbers.

Session 4 has one job: **build indexes → run experiments → analyse results**. Do not build new features unless experiments expose a concrete bug. Do not refactor working code.

---

## PROJECT LOCATION

```
E:\Lab\NLP\Faculty-first RAG\
Windows 11 · Python 3.13.5 (Anaconda) · NVIDIA A100-80GB
```

---

## COMPLETE MODULE STATE (v0.3.0 — all done, do not modify unless a real bug is found)

```
factuality_rag/
├── model_registry.py        ✅ Singleton 4-bit model cache (get_model, get_tokenizer)
├── data/loader.py           ✅ HF dataset wrapper (NQ, HotpotQA, FEVER, TriviaQA, TruthfulQA)
├── data/wikipedia.py        ✅ WikiChunker + load_from_hf() streaming
├── index/builder.py         ✅ FAISS HNSW/IVFPQ + Pyserini collection prep
├── retriever/hybrid.py      ✅ Real LuceneSearcher + fallback
├── gating/probe.py          ✅ Entropy+logit-gap + multi-token + real ECE calibration
├── scorer/passage.py        ✅ NLI (passage or sentence mode) + cross-encoder + fusion
├── generator/wrapper.py     ✅ Mistral-7B-Instruct-v0.3 via model_registry (4-bit)
├── pipeline/orchestrator.py ✅ Pipeline class (load-once) + real provenance mapping
├── eval/metrics.py          ✅ EM, F1, decompose_claims(), compute_factscore()
├── cli/__main__.py          ✅ 4 CLI commands
└── experiment_runner.py     ✅ Pipeline class, saves predictions/metrics/metadata

scripts/
├── build_corpus.py          ✅ Wikipedia ingestion + index building
├── analyze_gating.py        ✅ Phase 4A gating oracle analysis
├── analyze_scorer.py        ✅ Phase 4B scorer AUC analysis
├── analyze_errors.py        ✅ Phase 4C error taxonomy
├── tune_scorer_weights.py   ✅ Grid search over (w_nli, w_overlap, w_ret)
├── aggregate_results.py     ✅ Cross-seed metric aggregation
└── bootstrap_test.py        ✅ Paired bootstrap significance test

configs/
├── exp_b1_closed_book.yaml  ✅
├── exp_b2_always_rag.yaml   ✅
├── exp_b3_gate_only.yaml    ✅
├── exp_b4_score_only.yaml   ✅
└── exp_full_pipeline.yaml   ✅
```

---

## KEY DESIGN DECISIONS (do not re-litigate these — they are implemented and tested)

| Decision | Value | File |
|----------|-------|------|
| Gating rule | `retrieve = (entropy > 1.2) OR (logit_gap < 2.0)` | `gating/probe.py` |
| Gating probe depth | `probe_tokens=1` (default), multi-token available | `gating/probe.py` |
| Retrieval fusion | `α=0.6 · dense_norm + 0.4 · bm25_norm` | `retriever/hybrid.py` |
| Scorer formula | `0.5·NLI + 0.2·overlap + 0.3·ret_norm` | `scorer/passage.py` |
| NLI direction | `premise=passage, hypothesis=query` | `scorer/passage.py` |
| NLI mode | `"passage"` (default) or `"sentence"` (via config) | `scorer/passage.py` |
| Score threshold | `0.4` (configurable) | `configs/` |
| Confidence tag | gating-skip → `"medium"` · avg≥0.7 → `"high"` · avg<0.45 → `"low"` | `pipeline/orchestrator.py` |
| Generator | Mistral-7B-Instruct-v0.3, 4-bit, shared via model_registry | `generator/wrapper.py` |

---

## THE ONLY OPEN BUG (low priority — do not fix in this session)

| ID | File | Issue | Action |
|----|------|-------|--------|
| BUG-7 | `eval/metrics.py` | `decompose_claims()` misses compound "and" claims (rule-based limit) | Document as limitation; fix only if GEN_IGNORE analysis shows it matters |

---

## SESSION 4 TASK FLOW

```
PHASE A: Build Indexes  (prerequisite — do this first, ~3 hrs)
    │
    ▼
PHASE B: Run Baselines  (B1 → B2 → B3 → B4 → Full × 3 seeds, ~6 hrs)
    │
    ▼
PHASE C: Analyse Results  (gating oracle, scorer AUC, error taxonomy, ~2 hrs)
    │
    ▼
PHASE D: Ablations  (sensitivity sweeps on best config, ~4 hrs)
    │
    ▼
PHASE E: Parallel work  (human eval setup, prompt variants, Self-RAG, ~ongoing)
```

---

## PHASE A — BUILD INDEXES

### A1. Ingest Wikipedia (100K articles)

```python
# Run from project root
python scripts/build_corpus.py --sample-size 100000 --output data/wiki_100k_chunks.jsonl
```

**Verify:**
```python
python -c "
lines = sum(1 for _ in open('data/wiki_100k_chunks.jsonl'))
print(f'Passages: {lines}')
# Expect: 300,000–450,000
"
```

**If OOM:** reduce `--sample-size` to `50000` for dev run. Experiments still valid.

---

### A2. Build FAISS Dense Index

```bash
factuality-rag build-index \
    --corpus data/wiki_100k_chunks.jsonl \
    --faiss-out indexes/wiki100k.faiss \
    --embed-model sentence-transformers/all-mpnet-base-v2 \
    --faiss-type hnsw_flat
```

**Verify:**
```python
import faiss, json
idx = faiss.read_index("indexes/wiki100k.faiss")
ids = json.load(open("indexes/wiki100k.ids.json"))
print(f"Vectors: {idx.ntotal}  IDs: {len(ids)}  Dim: {idx.d}")
# Expect: ntotal == len(ids), d == 768
```

---

### A3. Build Lucene BM25 Index

```bash
# Step 1: prepare collection format
factuality-rag build-index \
    --corpus data/wiki_100k_chunks.jsonl \
    --pyserini-out indexes/wiki100k_pyserini_col

# Step 2: build Lucene index
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input indexes/wiki100k_pyserini_col \
    --index indexes/wiki100k_lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions --storeDocvectors --storeRaw
```

**Verify:**
```python
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("indexes/wiki100k_lucene")
hits = s.search("speed of light", k=3)
for h in hits: print(h.docid, f"{h.score:.2f}", h.raw[:60])
# Expect: real passages about speed of light, not empty
```

---

### A4. End-to-End Smoke Test

```bash
factuality-rag run \
    --query "Who invented the telephone?" \
    --faiss-index indexes/wiki100k.faiss \
    --pyserini-index indexes/wiki100k_lucene \
    --k 10
```

**Pass criteria — all must be true:**
- Answer does NOT start with `"Mock answer"`
- Trusted passages contain text about telephony/Bell/Gray
- Confidence tag is `"high"`, `"medium"`, or `"low"` (not blank)
- No stack traces

**Record:** indexing time, index sizes, passage count.

---

## PHASE B — RUN BASELINES

Run in this exact order. Record every result in the Results Table in STATUS_TRACKER.md immediately after each run.

### B1. Closed-Book Baseline

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b1_closed_book.yaml \
    --dataset natural_questions --split validation \
    --sample 500 --seed 42 --run-id b1_nq_500_s42
```

Expected runtime: ~25 min. Expected EM range: 0.20–0.35.

---

### B2. Always-RAG Baseline

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b2_always_rag.yaml \
    --dataset natural_questions --split validation \
    --sample 500 --seed 42 --run-id b2_nq_500_s42
```

Expected runtime: ~90 min. **Expected: B2.EM > B1.EM. If not, stop and diagnose before continuing.**

---

### B3. Gate-Only Baseline

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b3_gate_only.yaml \
    --dataset natural_questions --split validation \
    --sample 500 --seed 42 --run-id b3_nq_500_s42
```

Key metric to watch: `retrieval_rate`. Target: 40–70%.
- If `retrieval_rate < 30%`: gating too aggressive → lower `entropy_thresh` to 1.0
- If `retrieval_rate > 85%`: gating adds no efficiency → raise `entropy_thresh` to 1.5

---

### B4. Score-Only Baseline

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b4_score_only.yaml \
    --dataset natural_questions --split validation \
    --sample 500 --seed 42 --run-id b4_nq_500_s42
```

Key comparison: B4.FactScore vs B2.FactScore.
- If B4.FactScore > B2.FactScore → scorer is filtering bad evidence correctly ✅
- If B4.EM << B2.EM (>3 points lower) → scorer over-filtering → lower threshold to 0.3

---

### B5. Full Pipeline (3 seeds)

```powershell
# Windows PowerShell
foreach ($seed in 42, 123, 456) {
    python -m factuality_rag.experiment_runner `
        --config configs/exp_full_pipeline.yaml `
        --dataset natural_questions --split validation `
        --sample 500 --seed $seed `
        --run-id "full_nq_500_s$seed"
}
```

Expected runtime per seed: ~90 min. Total: ~4.5 hrs.

**STOP criteria — if Full.EM < B2.EM by more than 3 points:**
Run this diagnostic before continuing:
```python
# How many queries had zero trusted passages?
import json
preds = [json.loads(l) for l in open("runs/full_nq_500_s42_<ts>/predictions.jsonl")]
zero_psg = sum(1 for p in preds if len(p.get("trusted_passages", [])) == 0)
print(f"Zero-passage queries: {zero_psg}/500 ({zero_psg/5:.1f}%)")
# If >20%: lower score_threshold to 0.3 and rerun
```

---

### B6. Additional Datasets (after NQ-Open is stable)

```bash
# HotpotQA
python -m factuality_rag.experiment_runner \
    --config configs/exp_full_pipeline.yaml \
    --dataset hotpot_qa --split validation \
    --sample 500 --seed 42 --run-id full_hotpot_500_s42

# TruthfulQA (all 817 examples)
python -m factuality_rag.experiment_runner \
    --config configs/exp_full_pipeline.yaml \
    --dataset truthful_qa --split validation \
    --seed 42 --run-id full_tqa_all_s42
```

---

## PHASE C — ANALYSE RESULTS

Run these scripts once Phase B NQ-Open runs are complete.

### C1. Aggregate Cross-Seed Metrics

```bash
python scripts/aggregate_results.py --pattern "full_nq_500_s*"
```

Expected output: mean ± std table for EM, F1, FactScore, retrieval_rate across seeds 42/123/456.

---

### C2. Gating Oracle Analysis

```bash
python scripts/analyze_gating.py \
    --full-run runs/full_nq_500_s42_<timestamp>/ \
    --closedbook-run runs/b1_nq_500_s42_<timestamp>/
```

**Record these four numbers:**
```
Retrieval trigger rate:        ___%   (if <30% or >85% → adjust thresholds)
Gate-skip accuracy:            ___%   (of skips, how many had EM=1 on closed-book)
False-skip rate:               ___%   (gate skipped but closed-book EM=0 — overconfident errors)
Wasted-retrieval rate:         ___%   (gate retrieved but closed-book EM=1 — unnecessary)
```

**Action thresholds:**
- False-skip rate > 15% → lower `gating.entropy_threshold` from 1.2 to 1.0, rerun B5
- Wasted-retrieval rate > 40% → raise `gating.entropy_threshold` from 1.2 to 1.5, rerun B5

---

### C3. Scorer AUC Analysis

```bash
python scripts/analyze_scorer.py \
    --predictions runs/full_nq_500_s42_<timestamp>/predictions.jsonl \
    --dataset natural_questions --split validation --sample 500
```

**Record:**
```
Scorer AUC-ROC (full final_score):  _____   (target: >0.70)
NLI-only AUC:                       _____
Overlap-only AUC:                   _____
Retriever-score-only AUC:           _____
```

**Action thresholds:**
- AUC < 0.65 → check NLI model loaded correctly (not in fallback mode); check logs
- NLI-only AUC << full AUC → overlap/retriever scores are adding noise; consider removing

---

### C4. Error Taxonomy

```bash
python scripts/analyze_errors.py \
    --full-run runs/full_nq_500_s42_<timestamp>/ \
    --b2-run runs/b2_nq_500_s42_<timestamp>/ \
    --n-sample 50
```

**Record error counts (target: 50 failures manually classified):**

| Code | Meaning | Count | Action if >10 |
|------|---------|-------|---------------|
| `GATE_MISS` | Gated out retrieval, model hallucinated | — | Lower entropy_thresh |
| `RETRIEVAL_MISS` | Correct passage not in top-10 | — | Increase top_k to 20 |
| `SCORER_DROP` | Correct passage filtered by scorer | — | Lower score_threshold or use sentence NLI |
| `GEN_IGNORE` | Correct passage provided, Mistral ignored it | — | Try Prompt Variant B (Phase E2) |
| `ANSWER_FORMAT` | Correct answer but EM normalisation failed | — | Fix EM normalisation |
| `CORPUS_GAP` | Answer simply not in 100K subset | — | Note as limitation |

---

### C5. Bootstrap Significance Tests

```bash
python scripts/bootstrap_test.py \
    --run-a runs/full_nq_500_s42_<timestamp>/ \
    --run-b runs/b2_nq_500_s42_<timestamp>/ \
    --metric factscore \
    --n-resamples 1000
```

Run for: Full vs B2 (FactScore), Full vs B4 (EM), Full vs B3 (retrieval_rate).
Record p-values in Results Table.

---

## PHASE D — SENSITIVITY ABLATIONS

Run only after Phase C analysis is complete and Phase B results are stable. Each is a single YAML parameter override.

### D1. Score Threshold Sweep

```powershell
foreach ($thr in 0.2, 0.3, 0.4, 0.5, 0.6) {
    python -m factuality_rag.experiment_runner `
        --config configs/exp_full_pipeline.yaml `
        --override "scorer.score_threshold=$thr" `
        --dataset natural_questions --split validation `
        --sample 500 --seed 42 `
        --run-id "ablation_threshold_$thr"
}
```

Plot EM vs FactScore vs Retrieval% against threshold → `figures/threshold_sweep.png`

---

### D2. Alpha (Dense/Sparse Fusion) Sweep

```powershell
foreach ($a in 0.3, 0.5, 0.6, 0.7, 0.9) {
    python -m factuality_rag.experiment_runner `
        --config configs/exp_full_pipeline.yaml `
        --override "retriever.alpha=$a" `
        --dataset natural_questions --split validation `
        --sample 500 --seed 42 `
        --run-id "ablation_alpha_$a"
}
```

---

### D3. Top-K Sweep

```powershell
foreach ($k in 3, 5, 10, 15, 20) {
    python -m factuality_rag.experiment_runner `
        --config configs/exp_full_pipeline.yaml `
        --override "retriever.top_k=$k" `
        --dataset natural_questions --split validation `
        --sample 500 --seed 42 `
        --run-id "ablation_topk_$k"
}
```

---

### D4. Gating Threshold Sweep

Key ablation: at what entropy threshold does efficiency-accuracy tradeoff optimise?

```powershell
foreach ($eth in 0.8, 1.0, 1.2, 1.5, 2.0) {
    python -m factuality_rag.experiment_runner `
        --config configs/exp_full_pipeline.yaml `
        --override "gating.entropy_threshold=$eth" `
        --dataset natural_questions --split validation `
        --sample 500 --seed 42 `
        --run-id "ablation_entropy_$eth"
}
```

---

### D5. Scorer Weight Tuning (on FEVER dev)

```bash
python scripts/tune_scorer_weights.py \
    --dataset fever --split validation --sample 1000
```

If better weights found (AUC improves > 0.03 over current 0.5/0.2/0.3), update `exp_full_pipeline.yaml` and rerun B5.

---

### D6. NLI Mode Comparison (passage vs sentence)

```bash
# Run sentence-level NLI variant
python -m factuality_rag.experiment_runner \
    --config configs/exp_full_pipeline.yaml \
    --override "scorer.nli_mode=sentence" \
    --dataset natural_questions --split validation \
    --sample 500 --seed 42 \
    --run-id "ablation_nli_sentence"
```

Compare FactScore and SCORER_DROP rate vs default `nli_mode=passage`.

---

### D7. Cross-Encoder Ablation

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_full_pipeline.yaml \
    --override "scorer.cross_encoder_model=cross-encoder/ms-marco-MiniLM-L-12-v2" \
    --dataset natural_questions --split validation \
    --sample 500 --seed 42 \
    --run-id "ablation_crossencoder"
```

Compare EM, FactScore, latency vs no cross-encoder.

---

## PHASE E — PARALLEL WORK (run alongside D)

### E1. Human Evaluation Setup

**Goal:** 300 annotated query-answer pairs comparing B2 (always-RAG) vs Full Pipeline.

**Sample queries:**
```python
# scripts/sample_human_eval_queries.py
import json, random
random.seed(42)

full_preds = [json.loads(l) for l in open("runs/full_nq_500_s42_<ts>/predictions.jsonl")]
b2_preds   = [json.loads(l) for l in open("runs/b2_nq_500_s42_<ts>/predictions.jsonl")]

# Stratify by confidence tag: 100 high, 100 medium, 100 low
high   = [p for p in full_preds if p["confidence_tag"] == "high"][:100]
medium = [p for p in full_preds if p["confidence_tag"] == "medium"][:100]
low    = [p for p in full_preds if p["confidence_tag"] == "low"][:100]

sample = high + medium + low
# Match to B2 predictions for same queries
# Export to annotation format (CSV or Label Studio JSON)
```

**Annotation dimensions per query (binary):**
1. Correctness — Is the answer factually correct?
2. Groundedness — Is every claim supported by a cited passage?
3. Relevance — Are the cited passages relevant?
4. Hallucination — Does the answer contain any unsupported claim?

**Setup:** Label Studio (free, self-hosted) or Google Forms.
**Annotators:** 2 per query. Compute Cohen's κ after 50 queries — target κ > 0.7.

---

### E2. Prompt Engineering Study

Test 3 prompt variants on NQ-Open dev (100 queries, seed 42).

**Variant A (current — baseline):**
```
[INST] Answer the question using ONLY the provided context.
Context: {context}
Question: {query} [/INST]
```

**Variant B (explicit citation):**
```
[INST] Answer the question using ONLY the passages below.
If the passages don't answer the question, say: "I cannot determine this from the provided context."

Passages:
[1] {passage_1}
[2] {passage_2}
...

Question: {query}
Answer (cite passage numbers, e.g. "According to [1]..."): [/INST]
```

**Variant C (chain-of-thought):**
```
[INST] Read the passages carefully, then answer step by step.
Passages: {context}
Question: {query}
Let me think step by step: [/INST]
```

Create `configs/exp_prompt_a.yaml`, `configs/exp_prompt_b.yaml`, `configs/exp_prompt_c.yaml`. Run each on 100 NQ queries. Choose best by FactScore + lowest GEN_IGNORE rate.

---

### E3. Self-RAG Baseline (high-value if time allows)

```bash
# Load Self-RAG checkpoint
pip install vllm  # optional, for faster inference
# HuggingFace model: selfrag/selfrag_llama2_7b

python scripts/run_selfrag_baseline.py \
    --dataset natural_questions \
    --split validation \
    --sample 500 \
    --seed 42 \
    --run-id selfrag_nq_500_s42
```

Create `scripts/run_selfrag_baseline.py` that:
1. Loads `selfrag/selfrag_llama2_7b`
2. Uses same Wikipedia 100K index
3. Outputs same `predictions.jsonl` format
4. Reports same metrics

This is the strongest comparison for the paper.

---

## TROUBLESHOOTING GUIDE

### "Full pipeline EM is lower than B2"
1. Check zero-passage rate (should be <20%)
2. Lower `score_threshold` from 0.4 → 0.3 and rerun
3. If still lower: run D6 (sentence NLI mode) before re-investigating

### "Retrieval rate is 95%+"
Gating is not activating. Check:
1. `gate: true` is set in config
2. Model is loading correctly (not in fallback)
3. Try raising `entropy_thresh` to 1.5

### "FactScore is near zero"
NLI model likely in fallback mode. Check:
```python
from transformers import pipeline
nli = pipeline("text-classification",
    model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device=0)
result = nli("The sky is blue. [SEP] The sky has a blue colour.")
print(result)
# Must return entailment with score > 0.9
```

### "CUDA out of memory"
- Verify model_registry is working (model loaded once, not multiple times)
- Move NLI scorer to CPU: add `scorer.device: "cpu"` to config
- Reduce batch size in scorer: `scorer.batch_size: 1`

### "Pyserini / Java error"
- Verify Java 11+ is installed: `java -version`
- Delete `indexes/wiki100k_lucene/` and rebuild from scratch
- Check Pyserini version: `pip show pyserini`

---

## HOW TO COMMUNICATE WITH CLAUDE IN THIS SESSION

**Running experiments:** Just paste the command and output. No need to explain context.

**When something fails:**
> "Error running [PHASE X, task Y]:
> Command: [paste]
> Error: [paste full traceback]
> What failed and how do I fix it without touching mock-mode tests?"

**When results look wrong:**
> "Got these results: [paste metrics].
> Full pipeline EM=[X] vs Always-RAG EM=[Y].
> Explain what could cause this gap and what to check first."

**When an experiment finishes:**
> "Completed [run-id]. Results: EM=[X], F1=[Y], FactScore=[Z], Retrieval%=[W].
> What should I do next based on the STATUS_TRACKER?"

---

*Session 4 Prompt · v1.0 · 2026-02-28*
