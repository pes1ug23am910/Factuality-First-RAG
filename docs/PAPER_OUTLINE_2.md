# Paper Outline & Contribution Framing
> Factuality-First RAG · Draft v0.1 · Session 3

---

## Purpose

This document helps the team write the paper in parallel with running experiments. It defines the story, contribution framing, and section-by-section structure so writing can proceed as results arrive — rather than starting from blank page at month 4.

---

## 1. Contribution Statement

### What we are claiming (honest version)

We propose a **training-free, inference-time framework** for factuality-aware RAG that jointly addresses two failure modes of standard RAG:

1. **Unnecessary retrieval:** Models are often already confident in factual answers. Standard RAG retrieves context for every query, which (a) wastes computation and (b) risks conditioning on noisy or incorrect context.

2. **Uncritical evidence ingestion:** Retrieved passages vary in relevance and factual support. Standard RAG passes all top-K passages to the generator regardless of their quality.

Our system addresses both with:
- **Adaptive gating:** A single-forward-pass logit probe (entropy + logit gap) decides whether retrieval is needed, without fine-tuning.
- **Pre-generation passage scoring:** A linear fusion of NLI entailment, token overlap, and retrieval confidence filters unreliable passages before the generator sees them.

**What makes this different from prior work:**
- Self-RAG: requires supervised fine-tuning of the generator with reflection tokens. Our system works with any off-the-shelf instruction-tuned LLM.
- CRAG: corrects after retrieval (post-hoc web search fallback). We filter before generation.
- FLARE: triggers retrieval based on low-probability tokens during generation. We decide before generation in a single forward pass.
- TARG: gating only (for efficiency), no passage-level quality scoring.

**Honest limitations to acknowledge:**
- Gating probe uses a heuristic threshold, not learned from data.
- Passage scorer is not trained on our specific task distribution.
- 100K Wikipedia subset is smaller than DPR's full Wikipedia index.
- Multi-hop questions require joint evidence; our scorer scores passages independently.

---

## 2. Target Venue

### Primary target: EMNLP 2026 (main conference)
- Deadline: ~April/May 2026 (check CFP)
- Appropriate for: empirical NLP systems paper with clear evaluation

### Fallback targets:
- ACL 2026 Findings
- NAACL 2026
- EMNLP 2026 System Demonstrations track (if implementation is the contribution)
- AKBC / FEVER workshop (if focusing on factuality verification aspect)

### What it takes for main conference:
- Clear improvement on established benchmarks (NQ-Open, HotpotQA)
- Meaningful comparison to Self-RAG or FLARE
- Human evaluation supporting automated metrics
- Ablations showing each component contributes

---

## 3. Paper Structure

### Abstract (150 words target)

```
Retrieval-Augmented Generation (RAG) systems typically retrieve evidence for 
every query and pass all retrieved passages to the generator — two decisions 
that can introduce noise and increase hallucinations. We propose Factuality-
First RAG, a training-free pipeline that makes retrieval conditional on the 
generator's own uncertainty and filters evidence by factual alignment before 
generation. Our approach combines (1) an adaptive gating probe based on 
next-token entropy and logit gap, which skips retrieval when the model is 
already confident, and (2) a passage-level factuality scorer that fuses NLI 
entailment probability, lexical overlap, and retrieval confidence to filter 
unreliable evidence. Unlike Self-RAG, our method requires no fine-tuning and 
works with any instruction-tuned LLM. On Natural Questions and HotpotQA, our 
system improves FactScore by X.X points over always-retrieve baselines while 
reducing retrieval calls by Y%, with no degradation in exact match.
```

*Fill in X.X and Y% from experimental results.*

---

### Section 1: Introduction (1.5 pages)

**Story arc:**
1. RAG has improved LLM factuality, but hallucinations persist even with retrieved context. Why?
2. Two underappreciated failure modes: unnecessary retrieval (adds noise) and uncritical evidence ingestion (bad passages mislead generator).
3. Prior work addresses these separately: gating approaches (TARG, FLARE) focus on efficiency; verification approaches (CRAG, FActScore) focus on post-hoc correction.
4. We propose a tight coupling of both in a training-free framework: gate first, score before generation.
5. Key results (3 bullet points with numbers from experiments).

**Figures:**
- Fig 1: 4-stage pipeline diagram (use ASCII from ARCHITECTURE.md, rendered properly)
- Fig 2: Motivation example — same query through Always-RAG (hallucinates) vs our system (filters bad passage, correct answer)

---

### Section 2: Related Work (1 page)

**Subsections:**
- 2.1 Retrieval-Augmented Generation (Lewis et al. 2020, RAG; Izacard et al. 2021, FiD)
- 2.2 Adaptive Retrieval (TARG — Mallen et al. 2022; FLARE — Jiang et al. 2023)
- 2.3 Retrieval Quality and Correction (CRAG — Yan et al. 2024; RECOMP — Xu et al. 2023)
- 2.4 Self-RAG (Asai et al. 2024) — most important comparison: requires fine-tuning, we don't
- 2.5 Factuality Evaluation (FActScore — Min et al. 2023; FactCheck)

**Key differentiator table (for Related Work section):**

| System | Requires fine-tuning | Gating | Pre-generation filtering | Post-generation correction |
|--------|---------------------|--------|------------------------|--------------------------|
| Self-RAG | ✓ | ✓ (reflection tokens) | — | — |
| CRAG | — | — | — | ✓ (web search fallback) |
| FLARE | — | ✓ (low-prob tokens) | — | — |
| TARG | — | ✓ (popularity-based) | — | — |
| **Ours** | **✗** | **✓ (entropy+gap)** | **✓ (NLI+overlap+ret)** | **—** |

---

### Section 3: Method (2.5 pages)

**3.1 Problem Formulation**
Given query q and knowledge corpus C, produce answer a with high factual accuracy. Formally define the 4-stage pipeline and its inputs/outputs.

**3.2 Adaptive Retrieval Gating**
- Forward pass mechanism: single token, no decoding
- Entropy H and logit gap Δ definitions
- Decision rule and thresholds
- Temperature calibration (ECE)
- Cost analysis: ~50ms on A100 vs ~2s for full generation

**3.3 Hybrid Retrieval**
- Dense: SentenceTransformer + FAISS HNSW
- Sparse: BM25 via Pyserini/Lucene
- Score normalisation and linear fusion (α parameter)
- Complexity: O(log N) HNSW + BM25 query

**3.4 Passage-Level Factuality Scoring**
- NLI entailment: passage as premise, query as hypothesis; RoBERTa-large
- Token overlap F1
- Retrieval confidence (normalised combined score)
- Linear fusion formula and threshold

**3.5 Conditioned Generation and Confidence Tagging**
- Prompt template
- Confidence tag logic
- Provenance output

---

### Section 4: Experimental Setup (0.75 pages)

- Datasets table (NQ-Open, HotpotQA, FEVER, TriviaQA, TruthfulQA)
- Models used (generator, embedder, NLI, cross-encoder)
- Baselines (B1-B4 + Self-RAG if reproduced)
- Metrics (EM, F1, FactScore, Retrieval%, Latency)
- Reproducibility statement (all open-source, code released)

---

### Section 5: Results (2 pages)

**Table 1: Main Results on NQ-Open (Table must fill)**

| System | EM | F1 | FactScore | Ret% | Latency |
|--------|----|----|-----------|------|---------|
| B1 Closed-book | | | | 0% | |
| B2 Always-RAG | | | | 100% | |
| B3 Gate-only | | | | —% | |
| B4 Score-only | | | | 100% | |
| Self-RAG†  | | | | —% | |
| **Ours (Full)** | | | | —% | |

† = reproduced; ‡ = reported from paper

**Table 2: Cross-dataset Results**
Full Pipeline across NQ-Open, HotpotQA, TruthfulQA.

**Table 3: Ablation Results (NQ-Open dev)**
α sweep, threshold sweep, scorer component ablation (NLI-only vs overlap-only vs fusion).

**Figure 3: Gating precision-recall curve**
Precision-recall of gating decisions (using EM as oracle label).

**Figure 4: Score distribution**
NLI score distribution for relevant vs irrelevant passages (from AUC analysis).

---

### Section 6: Analysis (1 page)

**6.1 Gating Probe Behaviour**
- What fraction of queries trigger retrieval by dataset?
- Is entropy a reliable proxy for factual accuracy? (Gating accuracy table)
- False skip analysis: how often is the model overconfident?

**6.2 Passage Scorer Effectiveness**
- AUC-ROC by component (NLI alone, overlap alone, retrieval alone, fusion)
- Sentence-level vs passage-level NLI comparison (if ablation is done)

**6.3 Error Analysis**
- Error taxonomy distribution (GATE_MISS, RETRIEVAL_MISS, SCORER_DROP, GEN_IGNORE, CORPUS_GAP)
- Qualitative examples: one success case + one failure case for each category

**6.4 Human Evaluation**
- Correctness, Groundedness, Relevance, Hallucination for Full vs B2
- Cohen's κ for annotator agreement
- Statistical significance (McNemar's test)

---

### Section 7: Conclusion (0.5 pages)

- Summary of contributions
- Key findings (1-2 sentences each)
- Limitations (multi-hop, corpus coverage, heuristic gating thresholds)
- Future work (learned gating, iterative retrieval, larger corpus)

---

## 4. Writing Timeline

| Milestone | Target Date | Owner |
|-----------|-------------|-------|
| Experiment results available (B1-B4 + Full) | Month 2, week 3 | All |
| Draft Section 3 (Method) — write from current design | Month 2, week 4 | Member B |
| Draft Section 2 (Related Work) | Month 2, week 4 | Member C |
| Ablation results + figures | Month 3, week 2 | Member A |
| Human eval complete | Month 3, week 3 | Member C |
| Draft Section 5 (Results) + Section 6 (Analysis) | Month 3, week 3 | Member A |
| Draft Intro + Abstract | Month 3, week 4 | Member B |
| First full draft circulated | Month 4, week 1 | All |
| Mentor feedback round | Month 4, week 2 | Mentor |
| Camera-ready or submission | Month 4, week 3 | All |

---

## 5. Key Claims to Verify Before Submission

These are the claims the paper will make. Each needs a corresponding experiment result.

| Claim | Evidence needed | Experiment |
|-------|----------------|------------|
| "Reduces retrieval calls without hurting quality" | Full.EM ≥ B2.EM - 2pp AND Full.Ret% < B2.Ret% | Phase 3B |
| "Improves faithfulness over always-RAG" | Full.FactScore > B2.FactScore (p<0.05) | Phase 3B |
| "Both components contribute independently" | Full > B3 on FactScore AND Full > B4 on Ret% | Phase 3B |
| "Training-free advantage over Self-RAG" | Full.FactScore comparable to Self-RAG despite no fine-tuning | Phase 3B (if Self-RAG reproduced) |
| "Human annotators confirm reduced hallucination" | Groundedness(Full) > Groundedness(B2), κ > 0.7 | Phase 4 human eval |

---

## 6. Anticipating Reviewer Objections

| Objection | Our response |
|-----------|-------------|
| "Why not just fine-tune?" | We explicitly study the no-fine-tuning setting. This is valuable for practitioners who cannot afford fine-tuning or want a plug-and-play solution. |
| "Gating probe thresholds are heuristic" | We provide threshold ablations (Fig/Table X) and calibration analysis. The system degrades gracefully as thresholds vary. |
| "NLI model may not generalise to QA domain" | We show AUC-ROC on annotated passages and compare NLI-only vs ensemble. FEVER training gives reasonable in-domain signal. |
| "100K Wikipedia is too small" | We show results are consistent with smaller and larger corpus sizes (ablation). The method, not the corpus, is the contribution. |
| "No comparison to Self-RAG" | [If not reproduced]: Self-RAG requires fine-tuning on proprietary data; comparison would be unfair. We instead compare on efficiency (no fine-tuning required). |
| "Multi-hop results are worse" | We acknowledge this as a limitation and explain why joint evidence modelling is future work. Our method still outperforms closed-book. |
