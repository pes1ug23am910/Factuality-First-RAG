# External Claude Validation Prompt — v0.2

> Copy everything between PROMPT START and PROMPT END into a new Claude session for an independent expert review.
> This is the Session 3 version, reflecting the real-components implementation completed in Session 2.

---

## PROMPT START — COPY FROM HERE

You are an expert NLP/IR researcher and senior Python engineer. I'm sharing my **Factuality-First RAG** project for a detailed technical review. The system is now past the mock-prototype stage — real components are wired, bugs are fixed, and we are about to run our first real experiments. I need your honest assessment of what will and won't work.

---

### 1. Project Overview

**Title:** Factuality-First Retrieval-Augmented Generation with Adaptive Gating and Passage-Level Provenance Scoring
**Version:** 0.2.0
**Team:** 3 graduate students | **Compute:** 1× A100-80GB | **Timeline:** 4 months (months 2–4 remaining)

**Core claims:**
1. Adaptive gating (entropy + logit-gap probe) reduces unnecessary retrieval while preserving answer quality
2. Passage-level factuality scoring (NLI entailment + overlap + retrieval score) filters bad evidence before generation, reducing hallucinations vs always-RAG
3. The combination (gating + scoring) achieves better faithfulness-efficiency tradeoff than either component alone

---

### 2. Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Gating Probe                       │
│  - Single forward pass through Mistral-7B    │
│  - Compute entropy H and logit gap Δ         │
│  - retrieve = (H > 1.2) OR (Δ < 2.0)        │
│  - Cost: ~50ms on A100, no decoding          │
└──────────────┬────────────────┬──────────────┘
               │ retrieve=True  │ retrieve=False
               ▼                ▼
┌────────────────────┐   ┌─────────────────────┐
│ Stage 2: Retrieval │   │ Direct Generation   │
│ FAISS (dense) +    │   │ confidence="medium" │
│ Pyserini (BM25)    │   └─────────────────────┘
│ α=0.6 fusion       │
│ top-K=10 passages  │
└──────────┬─────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Passage Scorer                     │
│  NLI: P(entailment | premise=passage,        │
│                       hypothesis=query)       │
│  overlap: token F1(query, passage)           │
│  final = 0.5·NLI + 0.2·overlap + 0.3·ret    │
│  filter: drop if final_score < 0.4           │
└──────────────┬───────────────────────────────┘
               │ trusted passages
               ▼
┌──────────────────────────────────────────────┐
│  Stage 4: Generator                          │
│  Mistral-7B-Instruct-v0.3 (4-bit)           │
│  [INST] RAG prompt with context [/INST]      │
│  Shared weights with gating probe            │
│  (model_registry singleton, loaded once)     │
└──────────────────────────────────────────────┘
```

---

### 3. Implementation Details

#### Model Registry (NEW in v0.2)
```python
# factuality_rag/model_registry.py
_models: Dict[str, AutoModelForCausalLM] = {}
_tokenizers: Dict[str, AutoTokenizer] = {}

def get_model(model_id: str, device: str = "cuda", load_in_4bit: bool = True):
    if model_id not in _models:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        _models[model_id] = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto"
        )
    return _models[model_id]
```

#### Pipeline Class (NEW in v0.2)
```python
class Pipeline:
    def __init__(self, config_path, mock_mode=False, ...):
        # Components built ONCE
        self.probe = GatingProbe(model=get_model(...), ...)
        self.retriever = HybridRetriever(faiss_path, lucene_path, ...)
        self.scorer = PassageScorer(nli_model=..., ...)
        self.generator = Generator(model=get_model(...), ...)  # same model instance
    
    def run(self, query: str) -> Tuple[str, List[Dict], Dict, str]:
        return run_pipeline(query, probe=self.probe, retriever=self.retriever, ...)
```

#### Gating Probe
```python
def should_retrieve(self, prompt, entropy_thresh=1.2, logit_gap_thresh=2.0) -> bool:
    logits = self._get_next_token_logits(prompt)   # single forward pass
    probs = softmax(logits / self.temp)
    H = -sum(p * log(p) for p in probs if p > 0)
    Δ = sorted(logits)[-1] - sorted(logits)[-2]
    return (H > entropy_thresh) or (Δ < logit_gap_thresh)
```

#### Passage Scorer (NLI argument order fixed in v0.2)
```python
def score_passages(self, query, passages) -> List[Dict]:
    for p in passages:
        p["nli_score"] = self._nli_entailment(premise=p["text"], hypothesis=query)
        p["overlap_score"] = token_f1(query, p["text"])
        p["final_score"] = 0.5 * p["nli_score"] + 0.2 * p["overlap_score"] + 0.3 * ret_norm
    return passages
```

#### FactScore (real implementation, v0.2)
```python
def decompose_claims(answer: str) -> List[str]:
    # Regex sentence splitting with abbreviation handling
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer)

def compute_factscore(answer, passages, nli_fn, threshold=0.7) -> Dict:
    claims = decompose_claims(answer)
    supported = sum(
        1 for claim in claims
        if any(nli_fn(premise=p["text"], hypothesis=claim) > threshold for p in passages)
    )
    return {"factscore": supported / len(claims), "n_claims": len(claims), ...}
```

---

### 4. Baselines to Compare Against

| ID | System | Description |
|----|--------|-------------|
| B1 | Closed-book | Mistral-7B with no retrieval at all |
| B2 | Always-RAG | Retrieve top-10 always, pass all to generator |
| B3 | Gate-only | Gate on/off retrieval, but no passage scoring |
| B4 | Score-only | Always retrieve, score+filter, no gating |
| **Full** | Our system | Gate + retrieve + score + generate |

---

### 5. Evaluation Setup

- **Primary datasets:** NQ-Open (NQ dev, n=500 for dev runs), HotpotQA dev, TruthfulQA (all 817)
- **Primary metrics:** Exact Match, Token F1, FactScore (claim-level NLI), Retrieval call %, Latency
- **Corpus:** English Wikipedia 100K article subset → ~400K passages → FAISS HNSW + Lucene BM25
- **Statistical testing:** Mean ± std over 3 seeds, paired bootstrap (n=1000 resamples)

---

### 6. Known Weaknesses We Are Aware Of

1. **Single-token gating probe** — entropy from just the first next-token position may be insufficient for complex queries
2. **Passage-level NLI** — long passages may dilute a single supporting sentence; sentence-level max-pooling may help
3. **Rule-based claim decomposition** — compound claims ("X is A and B") may not split correctly
4. **Linear fusion weights** — fixed at 0.5/0.2/0.3, not learned from data
5. **100K Wikipedia subset** — may miss answers to some NQ questions (corpus gap issue)
6. **HotpotQA multi-hop** — passage scoring is per-passage; multi-hop requires joint evidence which our scorer doesn't model
7. **Confidence tagging** — gating-skipped queries tagged "medium" regardless of actual answer quality

---

### 7. What We Need Reviewed

Please respond to each numbered point with specific, actionable feedback:

**7.1 Architecture soundness**
Are there fundamental flaws in the 4-stage design that will prevent our core claims from holding empirically? Specifically: (a) can a single-token logit probe reliably distinguish "I know this" from "I don't know"? (b) is NLI-based passage scoring reliable enough at 355M parameters (RoBERTa-large) to filter evidence for a 7B generator?

**7.2 The NLI hypothesis direction**
We use `P(entailment | premise=passage, hypothesis=query)`. But NLI direction matters: we want passages that *support* the query, not passages that the query supports. Is this the right direction? Should we instead frame it as: for each generated sentence/claim, check `(premise=passage, hypothesis=generated_claim)`?

**7.3 Scoring formula**
Is `final_score = 0.5·NLI + 0.2·overlap + 0.3·ret_norm` well-motivated? What's the expected correlation between these three signals? Could overlap and retrieval score be redundant? Would multiplicative combination be better for filtering (passage needs ALL three signals to be good, not just one)?

**7.4 Gating probe reliability**
For Mistral-7B, what is the expected correlation between next-token entropy and actual factual accuracy on NQ-Open? Is there published evidence that this proxy works? What thresholds have been used in published work (TARG, FLARE)?

**7.5 Experiment plan gaps**
Given our 4-month timeline with only months 2-4 remaining: which experiments are essential for the paper vs which are nice-to-have? Are our baseline choices appropriate for the EMNLP/ACL submission we're targeting?

**7.6 Differentiation from prior work**
Specifically vs Self-RAG (Asai et al. 2023): Self-RAG uses reflection tokens trained via supervised fine-tuning. Our gating is inference-time only. What does this mean for our claims? Is "no fine-tuning required" a meaningful contribution or does it just mean we're weaker? Vs FLARE (Jiang et al. 2023): FLARE also uses generation confidence to trigger retrieval. What is our exact differentiator?

**7.7 Potential failure modes for Session 3 experiments**
What specific outcomes should we be worried about when we get first real numbers? What would "the system is fundamentally broken" look like in the results vs "needs tuning"?

**7.8 Paper framing**
Given what we have (gating + scoring, open-source, no fine-tuning), what is the strongest honest contribution statement for a workshop or main conference paper? What would be overselling vs fair characterisation?

---

## PROMPT END

---

## How to Use This Prompt

1. Start a fresh Claude conversation
2. Copy everything between **PROMPT START** and **PROMPT END**
3. Paste as a single message
4. The response will be structured around all 8 review points

### High-value follow-up questions:

**On NLI direction:**
> "Give me a worked example with a real NQ question showing whether (premise=passage, hypothesis=query) or (premise=passage, hypothesis=generated_claim) is the better framing. Use a 2-3 sentence Wikipedia passage and a factoid question."

**On scoring alternatives:**
> "Write Python code for a multiplicative passage scorer that replaces our linear fusion. Show both formulas and explain when each would produce different rankings."

**On gating reliability:**
> "Point me to specific papers that measure correlation between LLM next-token entropy and factual accuracy. What AUC values have been reported?"

**On paper framing:**
> "Write a 150-word abstract for our system assuming Full Pipeline beats Always-RAG by 5 points on FactScore with 30% fewer retrieval calls. What venue would this target?"

**On HotpotQA failure prediction:**
> "Our passage scorer scores passages independently. For a bridge question in HotpotQA requiring two passages jointly, describe exactly how our system would fail and propose a minimal fix that doesn't require architectural changes."
