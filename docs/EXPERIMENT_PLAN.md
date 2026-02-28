# Experiment Plan — Factuality-First RAG

> Structured experiment schedule for a team of 3, 1× A100-80GB, 4-month timeline

---

## 1. Research Questions

| ID | Research Question |
|----|-------------------|
| **RQ1** | Does adaptive gating reduce unnecessary retrieval calls while maintaining or improving answer quality? |
| **RQ2** | Does passage-level factuality scoring (NLI + overlap + retrieval confidence) improve faithfulness over vanilla top-K retrieval? |
| **RQ3** | How does the combined pipeline (gating + scoring) compare against Always-RAG, Gate-only, Score-only, and Self-RAG baselines? |
| **RQ4** | How sensitive is performance to the gating thresholds (entropy, logit gap) and scorer fusion weights? |

---

## 2. Datasets

| Dataset | Task | Split | # Dev | # Test | Use Case |
|---------|------|-------|-------|--------|----------|
| **NQ-Open** | Open-domain QA | train / dev / test | 8,757 | 3,610 | Primary evaluation; single-hop factoid |
| **HotpotQA** | Multi-hop QA | train / dev / test | 7,405 | 7,405 | Multi-hop reasoning; bridge-type questions |
| **FEVER** | Fact verification | train / dev / test | 19,998 | 19,998 | Binary claim verification; high-confidence NLI eval |
| **TriviaQA** | Trivia QA | train / validation | 11,313 | 11,313 | Longer answers; reading comprehension |
| **TruthfulQA** | Truthfulness | — / — / mc | — | 817 | Hallucination stress-test; adversarial probing |

### Dev sampling strategy
For rapid iteration, use `dev_sample_size` in the YAML config to create small, reproducible dev slices:
- **Quick dev:** 500 examples (< 1 hr on A100)
- **Full dev:** entire dev split (~3-5 hrs on A100)

---

## 3. Knowledge Corpus

| Source | Size | Format |
|--------|------|--------|
| English Wikipedia (Dec 2023 dump) | ~6.5M articles, ~37M passages after chunking | JSONL |
| Chunking parameters | `chunk_size=256`, `overlap=32` tokens | — |
| FAISS index | IndexHNSWFlat (`M=32, ef_construction=40`) for dev; IndexIVFPQ (`nlist=4096, m=64, bits=8`) for prod | `.faiss` + `.ids.json` |
| Lucene index | Built via Pyserini `JsonCollection` | Lucene directory |

### Corpus preparation steps
1. Download `enwiki-20231201-pages-articles.xml.bz2` (~22 GB)
2. Parse + clean with `mwparserfromhell` → JSONL
3. Chunk with `WikiChunker(chunk_size=256, overlap=32)`
4. Build FAISS index: `factuality-rag build-index --passages chunks.jsonl --index-dir indexes/`
5. Build Lucene index for BM25: `python -m pyserini.index.lucene --collection JsonCollection --input indexes/pyserini_dir --index indexes/lucene_index`

**Estimated time:** ~12 hrs indexing (A100) for full Wikipedia. Use `--dev-sample-size 100000` for rapid experiments.

---

## 4. Models

| Component | Model | Parameters | Memory |
|-----------|-------|------------|--------|
| **Embedder** | `sentence-transformers/all-mpnet-base-v2` | 110M | ~0.5 GB |
| **DPR encoder** (alt) | `facebook/dpr-question_encoder-single-nq-base` | 110M | ~0.5 GB |
| **NLI scorer** | `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` | 355M | ~1.5 GB |
| **Generator** | `mistralai/Mistral-7B-Instruct-v0.3` (4-bit) | 7B | ~5 GB |
| **Generator** (alt) | `meta-llama/Llama-3.1-8B-Instruct` (4-bit) | 8B | ~6 GB |
| **Cross-encoder** (opt) | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | ~0.2 GB |

**Total memory:** ≈ 8–9 GB (fits easily on A100-80GB with room for batch processing).

---

## 5. Baselines

| ID | Baseline | Description |
|----|----------|-------------|
| **B1** | Closed-book | LLM only, no retrieval — measures parametric knowledge |
| **B2** | Always-RAG | Retrieve top-K for every query, no gating |
| **B3** | Gate-only | Use gating probe to decide retrieval, but no factuality scoring (use all retrieved passages) |
| **B4** | Score-only | Always retrieve, apply scorer to filter, no gating |
| **B5** | Self-RAG† | Reproduce Self-RAG with reflection tokens on same corpus/model |
| **B6** | CRAG† | Corrective RAG with web search fallback (adapted for offline use) |

† = external baselines; implement if time permits.

---

## 6. Metrics

### Primary metrics

| Metric | Scope | How measured |
|--------|-------|-------------|
| **Exact Match (EM)** | Answer quality | Strict string match after normalisation |
| **Token F1** | Answer quality | Token-level precision/recall |
| **FactScore** | Faithfulness | Claim-level verification against retrieved passages |
| **Hallucination rate** | Faithfulness | 1 − FactScore (fraction of unsupported claims) |
| **Retrieval calls** | Efficiency | % of queries where retrieval was triggered |

### Secondary metrics

| Metric | Scope | How measured |
|--------|-------|-------------|
| **Retrieval recall@K** | Retriever | % of annotated gold passages in top-K |
| **Gating accuracy** | Gating | % of queries correctly gated (using EM as oracle) |
| **Score AUC-ROC** | Scorer | Separation between relevant/irrelevant passages |
| **Provenance precision** | Pipeline | % of trusted passages that are actually relevant |
| **Latency** | Efficiency | End-to-end ms/query |

---

## 7. Experiment Schedule

### Phase 1: Infrastructure (Weeks 1–2)
| Task | Owner | Duration |
|------|-------|----------|
| Wire real generator (Mistral-7B-Instruct-v0.3, 4-bit) | Member A | 3 days |
| Wire Pyserini BM25 (build Lucene index) | Member B | 2 days |
| Ingest Wikipedia subset (100K articles for dev) | Member C | 2 days |
| Build FAISS + Lucene indexes on dev subset | Member B | 1 day |
| End-to-end dry run on NQ-Open dev (500 examples) | All | 1 day |

### Phase 2: NQ-Open Main Experiment (Weeks 3–5)
| Experiment | Config | Expected output |
|------------|--------|-----------------|
| **Exp 2.1** | Full pipeline (α=0.6, threshold=0.4) on NQ-Open dev | EM, F1, FactScore, % retrieval calls |
| **Exp 2.2** | B1 (closed-book) on NQ-Open dev | EM, F1 (parametric baseline) |
| **Exp 2.3** | B2 (always-RAG) on NQ-Open dev | EM, F1, FactScore |
| **Exp 2.4** | B3 (gate-only) on NQ-Open dev | EM, F1, % retrieval calls |
| **Exp 2.5** | B4 (score-only) on NQ-Open dev | EM, F1, FactScore |
| **Exp 2.6** | Sensitivity: vary α ∈ {0.3, 0.5, 0.6, 0.7, 0.9} | EM/F1/FactScore vs α curve |
| **Exp 2.7** | Sensitivity: vary score_threshold ∈ {0.2, 0.3, 0.4, 0.5, 0.6} | EM/F1/FactScore vs threshold curve |

### Phase 3: Multi-hop & Verification (Weeks 6–8)
| Experiment | Config | Expected output |
|------------|--------|-----------------|
| **Exp 3.1** | Full pipeline on HotpotQA dev | EM, F1, FactScore |
| **Exp 3.2** | B2 (always-RAG) on HotpotQA dev | EM, F1 (compare with Exp 3.1) |
| **Exp 3.3** | Full pipeline on FEVER dev | Accuracy, FactScore |
| **Exp 3.4** | Full pipeline on TruthfulQA | MC accuracy, hallucination rate |
| **Exp 3.5** | Compare gating behaviour across datasets | % retrieval calls per dataset |

### Phase 4: Ablations & Analysis (Weeks 9–11)
| Experiment | What varies | Purpose |
|------------|-------------|---------|
| **Exp 4.1** | Scorer weights: grid search (w_nli, w_overlap, w_ret) | Find optimal fusion |
| **Exp 4.2** | Gating thresholds: entropy ∈ {0.8, 1.0, 1.2, 1.5, 2.0}, logit_gap ∈ {1.0, 1.5, 2.0, 3.0} | Precision-recall trade-off |
| **Exp 4.3** | Remove NLI from scorer | Contribution of NLI component |
| **Exp 4.4** | Remove gating (always retrieve) + keep scorer | Contribution of gating component |
| **Exp 4.5** | Embedder comparison: mpnet vs DPR vs Contriever | Best dense encoder |
| **Exp 4.6** | top_k ∈ {3, 5, 10, 15, 20} | Retrieval depth vs quality |
| **Exp 4.7** | Cross-encoder reranking before scoring | Does reranking help? |

### Phase 5: Final Evaluation & Writing (Weeks 12–16)
| Task | Duration |
|------|----------|
| Full Wikipedia indexing (6.5M articles) | 2 days |
| Run best config on NQ-Open test split | 1 day |
| Run best config on HotpotQA test split | 1 day |
| Human evaluation (300 queries, 2 annotators) | 1 week |
| Error analysis + qualitative examples | 3 days |
| Paper draft + figures | 2 weeks |
| Revisions based on mentor feedback | 1 week |

---

## 8. Experiment Configuration Template

Each experiment should be run through the experiment runner:

```bash
python -m factuality_rag.experiment_runner \
    --config configs/exp_nq_main.yaml \
    --run-id "exp2.1_nq_full_pipeline" \
    --seed 42
```

### YAML config template for Exp 2.1:

```yaml
# configs/exp_nq_main.yaml
models:
  embedder: "sentence-transformers/all-mpnet-base-v2"
  nli: "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
  generator: "mistralai/Mistral-7B-Instruct-v0.3"

datasets:
  - "natural_questions"

chunker:
  chunk_size: 256
  overlap: 32

index:
  type: "hnsw"
  M: 32
  ef_construction: 40

retriever:
  top_k: 10
  alpha: 0.6

gating:
  entropy_threshold: 1.2
  logit_gap_threshold: 2.0

scorer:
  w_nli: 0.5
  w_overlap: 0.2
  w_retrieval: 0.3
  score_threshold: 0.4

pipeline:
  confidence_thresholds:
    high: 0.75
    low: 0.4
  seed: 42

eval:
  metrics:
    - "em"
    - "f1"
    - "factscore"
```

---

## 9. Run Output Structure

Each experiment run produces:
```
runs/
└── exp2.1_nq_full_pipeline_20250715_143022/
    ├── metadata.json       # Config, git commit, library versions, seed
    ├── predictions.jsonl   # Per-query: query, answer, passages, scores
    └── metrics.json        # Aggregated EM, F1, FactScore, retrieval_rate
```

### Expected results table skeleton

| Experiment | Config | EM | F1 | FactScore | Retrieval % | Latency (ms/q) |
|------------|--------|:--:|:--:|:---------:|:-----------:|:---------------:|
| B1 Closed-book | No retrieval | — | — | — | 0% | — |
| B2 Always-RAG | top_k=10 | — | — | — | 100% | — |
| B3 Gate-only | entropy=1.2 | — | — | — | —% | — |
| B4 Score-only | threshold=0.4 | — | — | — | 100% | — |
| **Full pipeline** | α=0.6, thr=0.4 | — | — | — | —% | — |

---

## 10. Statistical Significance

- Report mean ± std over 3 seeds (42, 123, 456).
- Use paired bootstrap test (n=1000 resamples) for EM/F1 comparisons.
- Report p-values for Full pipeline vs each baseline.

---

## 11. Compute Budget Estimate

| Task | GPU hours |
|------|-----------|
| Wikipedia indexing (FAISS + Lucene) | ~12 hrs |
| Phase 2 experiments (7 × NQ dev) | ~35 hrs |
| Phase 3 experiments (5 × multi-dataset) | ~25 hrs |
| Phase 4 ablations (~10 runs) | ~50 hrs |
| Phase 5 final evaluation | ~20 hrs |
| **Total** | **~142 hrs** |

At 1× A100, this is roughly **6 days of continuous compute**, easily fitting the 4-month timeline.

---

## 12. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Generator hallucinations dominate results | High | Stronger prompt engineering; try multiple models |
| BM25 index too slow on full Wikipedia | Medium | Use Pyserini pre-built indexes from official releases |
| NLI model weak on domain-specific claims | Medium | Fine-tune on FEVER train split; try DeBERTa-v3-large |
| Gating probe unreliable | Medium | Fall back to always-retrieve; report as ablation |
| A100 unavailable | High | All dev experiments work on smaller GPU (4090/3090) with 100K subset |
| Team member availability | Medium | Each module is independently testable; clear ownership |
