# Experiment Runbook — Factuality-First RAG
> Step-by-step operational guide for running experiments reproducibly.
> Version 1.0 · Session 3 onward

---

## Purpose

This runbook complements the Status Tracker. While the tracker tells you *what* to do, this document tells you *exactly how* to do it: the precise commands, expected outputs, how to know if something went wrong, and what to do about it.

---

## Prerequisites Checklist

Before running any experiment, verify:

```bash
# 1. GPU is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA A100-SXM4-80GB

# 2. Package is installed and importable
python -c "import factuality_rag; print(factuality_rag.__version__)"
# Expected: 0.2.0

# 3. Mock tests still pass (catch regressions before long runs)
pytest tests/ -m "not integration" -q
# Expected: 53 passed in ~2.0s

# 4. Disk space check (need ~50GB for full pipeline)
# Windows: dir E:\Lab\NLP
# Need: ~5GB for 100K Wikipedia chunks, ~3GB indexes, ~2GB model cache per experiment run
```

---

## Part 1: Building Real Indexes

### Step 1.1 — Ingest Wikipedia (100K articles)

**Expected duration:** 45–90 minutes on A100

```python
# Run from project root: python scripts/build_corpus.py
from factuality_rag.data.wikipedia import WikiChunker

chunker = WikiChunker(chunk_size=256, chunk_overlap=32)
result = chunker.load_from_hf(
    sample_size=100_000,
    output_path="data/wiki_100k_chunks.jsonl"
)
print(f"Passages written: {result['n_passages']}")
print(f"Estimated size: {result['size_mb']:.1f} MB")
```

**Success check:**
```bash
# Count lines in output file
python -c "
import json
with open('data/wiki_100k_chunks.jsonl') as f:
    n = sum(1 for _ in f)
print(f'Passages: {n}')
# Expected: ~300,000 - 450,000 passages
"
```

**If it fails:**
- `ConnectionError` — HuggingFace Hub unreachable. Check network. Try: `HF_DATASETS_OFFLINE=1` if cached.
- `MemoryError` — Reduce `sample_size` to 50_000 for dev run.
- `KeyError: 'text'` — Wikipedia HF dataset schema changed. Check `datasets.load_dataset("wikipedia", "20220301.en")` column names.

---

### Step 1.2 — Build FAISS Dense Index

**Expected duration:** 30–60 minutes (depends on passages count)

```bash
factuality-rag build-index \
    --corpus data/wiki_100k_chunks.jsonl \
    --faiss-out indexes/wiki100k.faiss \
    --embed-model sentence-transformers/all-mpnet-base-v2 \
    --faiss-type hnsw_flat \
    --hnsw-m 32 \
    --hnsw-ef-construction 200
```

**Success check:**
```python
import faiss
index = faiss.read_index("indexes/wiki100k.faiss")
print(f"Index size: {index.ntotal} vectors")
print(f"Dimension: {index.d}")
# Expected: ntotal matches passage count, d=768
```

**If it fails:**
- `RuntimeError: faiss` — Install `faiss-gpu` instead of `faiss-cpu` for A100 acceleration
- Out of memory — Use `--dev-sample-size 50000` to index fewer passages

---

### Step 1.3 — Build BM25 Lucene Index

**Expected duration:** 20–40 minutes

```bash
# Step A: prepare Pyserini collection format
factuality-rag build-index \
    --corpus data/wiki_100k_chunks.jsonl \
    --pyserini-out indexes/wiki100k_pyserini_collection

# Step B: build Lucene index
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input indexes/wiki100k_pyserini_collection \
    --index indexes/wiki100k_lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions \
    --storeDocvectors \
    --storeRaw
```

**Success check:**
```bash
python -c "
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('indexes/wiki100k_lucene')
hits = searcher.search('capital of France', k=3)
for h in hits:
    print(h.docid, h.score, h.raw[:80])
# Expected: passages mentioning Paris with BM25 scores
"
```

**If Pyserini fails:**
- Java not installed — Pyserini requires Java 11+. `java -version` to check.
- `ImportError` — Ensure `pip install pyserini` completed without errors in Anaconda env.
- Index directory must not already exist — delete and rebuild if rerunning.

---

### Step 1.4 — End-to-End Smoke Test

Before running full experiments, verify the real pipeline works:

```bash
factuality-rag run \
    --query "Who developed the theory of relativity?" \
    --faiss-index indexes/wiki100k.faiss \
    --pyserini-index indexes/wiki100k_lucene \
    --k 10 \
    --no-mock-mode

# Expected output:
# Query:      Who developed the theory of relativity?
# Answer:     Albert Einstein developed the theory of relativity in the early 20th century.
# Confidence: high | medium | low
# Trusted:    N passage(s)
# Retrieval:  True | False
```

**Sanity checks:**
1. Answer is not `"Mock answer for query: ..."` → real generation working ✅
2. Confidence tag is not unconditionally "high" → confidence logic working ✅
3. Trusted passages contain text about Einstein/relativity → scoring working ✅
4. If retrieval=False for this query → gating may need threshold adjustment (Einstein = very likely in model's parametric knowledge, skip is acceptable)

---

## Part 2: Running Baseline Experiments

### Experiment Naming Convention

All experiment runs follow: `{exp_id}_{dataset}_{n_samples}_seed{seed}`
Example: `b2_nq_500_seed42`

### Step 2.1 — Run B1 (Closed-Book)

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b1_closed_book.yaml \
    --dataset natural_questions \
    --split validation \
    --sample 500 \
    --seed 42 \
    --run-id b1_nq_500_seed42
```

**Expected runtime:** ~20-30 minutes (500 × ~3s per query, no retrieval)

**Expected output file:** `runs/b1_nq_500_seed42_<timestamp>/metrics.json`
```json
{
    "exact_match": 0.25,    // Expect 20-35% for 7B model on NQ
    "f1": 0.31,
    "factscore": 0.55,     // Lower without retrieved context
    "retrieval_rate": 0.0,
    "n_queries": 500
}
```

**Red flags:**
- EM > 0.5 → something is wrong (closed-book 7B shouldn't exceed this on NQ)
- EM < 0.10 → model may not be generating properly; check prompt template
- `factscore: 0.0` → FactScore implementation issue; check NLI model loaded

---

### Step 2.2 — Run B2 (Always-RAG)

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b2_always_rag.yaml \
    --dataset natural_questions \
    --split validation \
    --sample 500 \
    --seed 42 \
    --run-id b2_nq_500_seed42
```

**Expected runtime:** ~90-120 minutes (retrieval + NLI scoring + generation per query)

**Expected output:**
```json
{
    "exact_match": 0.38,    // Should be > B1 EM
    "f1": 0.45,
    "factscore": 0.62,
    "retrieval_rate": 1.0,  // Always retrieves
    "n_queries": 500
}
```

**Key comparison:** `B2.EM > B1.EM` — if retrieval doesn't help, investigate:
- Is the correct passage actually in the retrieved top-10? (check retrieval recall)
- Is Mistral-7B using the context? (check if answers reference retrieved text)

---

### Step 2.3 — Create and Run B3 (Gate-Only)

Create config:
```yaml
# configs/exp_b3_gate_only.yaml
models:
  embedder: "sentence-transformers/all-mpnet-base-v2"
  nli: "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
  generator: "mistralai/Mistral-7B-Instruct-v0.3"

retriever:
  faiss_path: "indexes/wiki100k.faiss"
  lucene_path: "indexes/wiki100k_lucene"
  top_k: 10
  alpha: 0.6

gating:
  entropy_threshold: 1.2
  logit_gap_threshold: 2.0

scorer:
  score_threshold: 0.0    # Accept ALL passages (no filtering)
  w_nli: 0.5
  w_overlap: 0.2
  w_retrieval: 0.3

pipeline:
  gate: true
  seed: 42
```

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b3_gate_only.yaml \
    --dataset natural_questions \
    --split validation \
    --sample 500 \
    --seed 42 \
    --run-id b3_nq_500_seed42
```

**Key metric:** `retrieval_rate` (what % of queries triggered retrieval)
- Target: 40-70% (if < 30%, gating is too aggressive; if > 85%, gating adds no efficiency)

---

### Step 2.4 — Create and Run B4 (Score-Only)

```yaml
# configs/exp_b4_score_only.yaml
# Same as full pipeline but gate: false
pipeline:
  gate: false            # Always retrieve
  score_threshold: 0.4   # Filter by scorer
  seed: 42
```

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_b4_score_only.yaml \
    --dataset natural_questions \
    --split validation \
    --sample 500 \
    --seed 42 \
    --run-id b4_nq_500_seed42
```

**Key comparison:** B4 vs B2
- `B4.factscore > B2.factscore` — scoring improves faithfulness ✅
- `B4.EM ≈ B2.EM` — scoring doesn't hurt answer quality ✅ (if B4.EM << B2.EM, scorer is over-filtering)

---

### Step 2.5 — Run Full Pipeline (3 Seeds)

```bash
for seed in 42 123 456; do
    python -m factuality_rag.experiment_runner \
        --config configs/exp_full_pipeline.yaml \
        --dataset natural_questions \
        --split validation \
        --sample 500 \
        --seed $seed \
        --run-id "full_nq_500_seed${seed}"
done
```

**Windows equivalent:**
```powershell
foreach ($seed in 42, 123, 456) {
    python -m factuality_rag.experiment_runner `
        --config configs/exp_full_pipeline.yaml `
        --dataset natural_questions `
        --split validation `
        --sample 500 `
        --seed $seed `
        --run-id "full_nq_500_seed$seed"
}
```

**Expected total runtime:** 3 × 90 min = ~4.5 hours

---

## Part 3: Reading and Comparing Results

### Aggregating Metrics Across Seeds

```python
# scripts/aggregate_results.py
import json
import glob
import numpy as np
from pathlib import Path

def load_metrics(pattern: str) -> list:
    """Load metrics.json from all matching run directories."""
    results = []
    for path in glob.glob(f"runs/{pattern}/metrics.json"):
        with open(path) as f:
            m = json.load(f)
            m["run_dir"] = str(Path(path).parent)
            results.append(m)
    return results

# Example: compare all baselines
for exp_id, pattern in [
    ("B1 Closed-book", "b1_nq_500_*"),
    ("B2 Always-RAG",  "b2_nq_500_*"),
    ("B3 Gate-only",   "b3_nq_500_*"),
    ("B4 Score-only",  "b4_nq_500_*"),
    ("Full Pipeline",  "full_nq_500_*"),
]:
    runs = load_metrics(pattern)
    if not runs:
        print(f"{exp_id}: No results yet")
        continue
    ems = [r["exact_match"] for r in runs]
    f1s = [r["f1"] for r in runs]
    fss = [r.get("factscore", 0) for r in runs]
    rrs = [r.get("retrieval_rate", 0) for r in runs]
    print(f"{exp_id:20s}  EM={np.mean(ems):.3f}±{np.std(ems):.3f}  "
          f"F1={np.mean(f1s):.3f}  FS={np.mean(fss):.3f}  "
          f"Ret={np.mean(rrs)*100:.0f}%  n={len(runs)}")
```

### Paired Bootstrap Significance Test

```python
# scripts/bootstrap_test.py
import json
import numpy as np
from pathlib import Path

def bootstrap_test(scores_a: list, scores_b: list, n_resamples: int = 1000) -> float:
    """
    Paired bootstrap test: returns p-value for H0: mean(A) = mean(B).
    scores_a, scores_b: per-query metric values (same query order).
    """
    diff_obs = np.mean(scores_a) - np.mean(scores_b)
    diffs = []
    n = len(scores_a)
    for _ in range(n_resamples):
        idx = np.random.randint(0, n, n)
        diffs.append(np.mean(np.array(scores_a)[idx]) - np.mean(np.array(scores_b)[idx]))
    # Two-sided p-value
    return np.mean(np.abs(diffs) >= np.abs(diff_obs))

# Usage: compare Full Pipeline vs B2 on EM
full_preds = load_predictions("runs/full_nq_500_seed42/predictions.jsonl")
b2_preds   = load_predictions("runs/b2_nq_500_seed42/predictions.jsonl")

# Extract per-query EM (must be same query order!)
full_em = [p["em"] for p in full_preds]
b2_em   = [p["em"] for p in b2_preds]

p_value = bootstrap_test(full_em, b2_em, n_resamples=1000)
print(f"Full vs B2 EM: p={p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
```

---

## Part 4: Failure Triage Guide

When results are unexpected, use this guide to diagnose the cause.

### Symptom: Full Pipeline EM << B2 (Always-RAG)

The scorer is over-filtering — too many good passages are being dropped.

**Diagnosis steps:**
```python
# Check what fraction of queries have 0 trusted passages
preds = load_predictions("runs/full_nq_500_seed42/predictions.jsonl")
n_zero_passages = sum(1 for p in preds if p["n_trusted_passages"] == 0)
print(f"Queries with no trusted passages: {n_zero_passages}/500 ({n_zero_passages/5:.1f}%)")
# If > 20%, scorer is too aggressive
```

**Fixes (in order of ease):**
1. Lower `score_threshold` from 0.4 → 0.3 in config
2. Switch NLI mode to sentence-level (Phase 5A-1 in tracker)
3. Verify NLI model is not in fallback mode (check log for warnings)

---

### Symptom: FactScore of Full Pipeline ≈ B2 (scoring not improving faithfulness)

The scorer is either not filtering effectively or the generator ignores context.

**Diagnosis steps:**
```python
# Compare per-query FactScore for queries where full pipeline DID filter passages
# vs queries where no filtering happened (all passages trusted)
for p in preds:
    n_retrieved = p.get("n_retrieved", 10)
    n_trusted   = p.get("n_trusted_passages", 0)
    filtering_happened = n_trusted < n_retrieved
    # Compute mean FactScore for each group
```

**Fixes:**
- If filtering happened but FactScore unchanged → generator ignores context → improve prompt (Suggestions §11)
- If no filtering happened → lower threshold or check NLI model is working

---

### Symptom: Retrieval rate < 30% (gating too conservative)

The gating probe is skipping retrieval too often. This means most queries have low entropy, which may or may not be correct.

**Diagnosis:**
```python
# Check gating decisions against closed-book EM
full_preds = load_predictions(...)
b1_preds   = load_predictions(...)  # closed-book

skipped = [(f, b) for f, b in zip(full_preds, b1_preds) if not f["retrieved"]]
skip_accuracy = np.mean([b["em"] for _, b in skipped])
print(f"Of {len(skipped)} skips, closed-book EM was {skip_accuracy:.2%}")
# If > 0.7 → gating is correct (model knows these)
# If < 0.5 → gating is overconfident, lower entropy_thresh
```

**Fix:** Lower `gating.entropy_threshold` from 1.2 → 1.0 or 0.9.

---

### Symptom: RuntimeError during generation (OOM)

Model + index + NLI scorer doesn't fit in 80GB A100 memory.

**Diagnosis:**
```bash
nvidia-smi  # Check memory usage
```

**Fixes:**
- Ensure model_registry is working (model loaded once, not 3×)
- Offload NLI scorer to CPU: set `scorer.device: "cpu"` in config
- Reduce batch size for NLI scoring: process 1 passage at a time

---

## Part 5: Ablation Run Protocol

For all ablation experiments (Phase 5C), use this protocol to ensure fair comparison:

1. **Fix everything except the variable being swept.** Use base config `configs/exp_full_pipeline.yaml` and override only the target parameter.

2. **Use the same 500 queries in the same order.** Set `seed: 42` and `sample: 500`.

3. **Run each configuration once** (seed 42 only for ablations — 3-seed replication only for main results).

4. **Save to structured output:**
```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_full_pipeline.yaml \
    --override "retriever.alpha=0.3" \
    --run-id "ablation_alpha_0.3"
```

5. **Aggregate ablation results:**
```python
# scripts/plot_ablation.py
import matplotlib.pyplot as plt

alphas  = [0.3, 0.5, 0.6, 0.7, 0.9]
em_vals = [load_metrics(f"ablation_alpha_{a}")[0]["exact_match"] for a in alphas]
fs_vals = [load_metrics(f"ablation_alpha_{a}")[0]["factscore"]   for a in alphas]

fig, ax1 = plt.subplots()
ax1.plot(alphas, em_vals, 'b-o', label='EM')
ax2 = ax1.twinx()
ax2.plot(alphas, fs_vals, 'r-s', label='FactScore')
plt.savefig("figures/alpha_sweep.png", dpi=150)
```

---

## Part 6: Common Commands Reference

```bash
# Run full pipeline on single query (quick test)
factuality-rag run --query "What is photosynthesis?" --k 5

# Evaluate saved predictions
factuality-rag evaluate \
    --predictions runs/full_nq_500_seed42_*/predictions.jsonl \
    --references data/nq_dev_references.jsonl

# Check what's in a run directory
python -c "
import json
with open('runs/full_nq_500_seed42_<timestamp>/metadata.json') as f:
    print(json.dumps(json.load(f), indent=2))
"

# Quick NLI test (verify NLI model loaded correctly)
python -c "
from transformers import pipeline
nli = pipeline('text-classification', 
    model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
    device=0)
result = nli('The capital of France is Paris. [SEP] Paris is the capital city of France.')
print(result)
# Expected: [{'label': 'entailment', 'score': ~0.95}]
"

# Check GPU memory usage during experiment
watch -n 5 nvidia-smi  # Linux/WSL
# Windows: nvidia-smi in loop or use GPU-Z
```

---

## Appendix: Expected Results Reference

Based on published literature for similar systems on NQ-Open dev split with 100K Wikipedia:

| System | EM | F1 | Retrieval% | Notes |
|--------|----|----|------------|-------|
| Closed-book 7B | ~25-35% | ~30-40% | 0% | Varies heavily by model |
| Always-RAG 7B | ~35-45% | ~42-52% | 100% | Depends on retrieval quality |
| Self-RAG (fine-tuned) | ~48-56% | ~56-64% | ~60% | Fine-tuned; stronger baseline |
| FLARE (inference only) | ~38-46% | ~46-54% | ~45% | Closer comparison to our system |

**Our target:** FactScore improvement of +5-10 points over Always-RAG, with ≤5% EM degradation and 30-50% retrieval reduction vs Always-RAG.

If your numbers fall significantly outside these ranges, either:
- The model isn't using retrieved context (check prompt)
- The corpus doesn't cover the answers (check retrieval recall@10)
- The NLI model isn't scoring correctly (check NLI smoke test)
