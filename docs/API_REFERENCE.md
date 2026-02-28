# API Reference — Factuality-First RAG

> Version 0.3.0 · Updated 2026-03-01

---

## Table of Contents

1. [factuality_rag.data](#1-factuality_ragdata)
2. [factuality_rag.index](#2-factuality_ragindex)
3. [factuality_rag.retriever](#3-factuality_ragretriever)
4. [factuality_rag.gating](#4-factuality_raggating)
5. [factuality_rag.scorer](#5-factuality_ragscorer)
6. [factuality_rag.generator](#6-factuality_raggenerator)
7. [factuality_rag.pipeline](#7-factuality_ragpipeline)
8. [factuality_rag.eval](#8-factuality_rageval)
9. [factuality_rag.model_registry](#9-factuality_ragmodel_registry)
10. [factuality_rag.cli](#10-factuality_ragcli)
11. [factuality_rag.experiment_runner](#11-factuality_ragexperiment_runner)
12. [scripts](#12-scripts)

---

## 1. `factuality_rag.data`

### `load_dataset()`

```python
def load_dataset(
    name: str,
    split: str = "train",
    dev_sample_size: Optional[int] = None,
    *,
    streaming: bool = False,
    trust_remote_code: bool = True,
) -> datasets.Dataset
```

Unified dataset loading wrapper around HuggingFace `datasets`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | required | Dataset identifier — one of `"natural_questions"`, `"hotpot_qa"`, `"fever"`, `"trivia_qa"`, `"truthful_qa"`, `"EleutherAI/truthful_qa_mc"`, or any HF dataset path |
| `split` | `str` | `"train"` | Dataset split |
| `dev_sample_size` | `Optional[int]` | `None` | Random sample size (seed=42) for dev iteration |
| `streaming` | `bool` | `False` | Use HF streaming mode |
| `trust_remote_code` | `bool` | `True` | Passed to `datasets.load_dataset` |

**Returns:** `datasets.Dataset` (or `IterableDataset` when streaming).

**Example:**
```python
from factuality_rag.data import load_dataset
ds = load_dataset("hotpot_qa", split="validation", dev_sample_size=50)
```

---

### `WikiChunker`

```python
class WikiChunker:
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        dry_run: bool = False,
        mock_mode: bool = False,
        dev_sample_size: Optional[int] = None,
    ) -> None
```

Chunk Wikipedia articles into fixed-token-window passages with deduplication.

**Methods:**

#### `chunk_text(title, text)`

```python
def chunk_text(self, title: str, text: str) -> Generator[Dict[str, Any], None, None]
```

Yields dicts matching the JSONL schema: `{"id", "title", "text", "tokens", "source"}`.

#### `process_articles(articles, output_path=None)`

```python
def process_articles(
    self,
    articles: Iterable[Dict[str, str]],
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]
```

Chunks an iterable of `{"title", "text"}` dicts and streams to JSONL.

#### `generate_mock_articles(n=10)`

```python
def generate_mock_articles(self, n: int = 10) -> List[Dict[str, str]]
```

Generate *n* synthetic articles for testing.

#### `load_from_hf(sample_size=None, output_path=None, wiki_config="20220301.en")` *(new in v0.2)*

```python
def load_from_hf(
    self,
    sample_size: Optional[int] = None,
    output_path: Optional[str] = None,
    wiki_config: str = "20220301.en",
) -> List[Dict[str, Any]]
```

Load Wikipedia articles from HuggingFace (streaming) and chunk them. Returns list of chunk dicts, writes JSONL.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sample_size` | `Optional[int]` | `None` | Max articles to process; `None` → all |
| `output_path` | `Optional[str]` | `None` | Output JSONL path (auto-generated if omitted) |
| `wiki_config` | `str` | `"20220301.en"` | HuggingFace Wikipedia snapshot config |

**JSONL Output Schema:**
```json
{"id": "uuid5", "title": "Article Title", "text": "chunk text...", "tokens": 200, "source": "enwiki"}
```

---

## 2. `factuality_rag.index`

### `build_faiss_index()`

```python
def build_faiss_index(
    jsonl_path: str,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    out_path: str = "faiss.index",
    mock_mode: bool = False,
    faiss_type: str = "hnsw_flat",
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    dim: int = 768,
    dev_sample_size: Optional[int] = None,
) -> str
```

Build a FAISS index from a JSONL corpus. Returns absolute path to the saved index.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `jsonl_path` | `str` | required | Path to chunked JSONL corpus |
| `embed_model` | `str` | `"sentence-transformers/all-mpnet-base-v2"` | HF model name |
| `out_path` | `str` | `"faiss.index"` | Output path for index |
| `mock_mode` | `bool` | `False` | Random embeddings (fixed seed) |
| `faiss_type` | `str` | `"hnsw_flat"` | `"hnsw_flat"` or `"ivfpq"` |
| `dim` | `int` | `768` | Embedding dimension |
| `dev_sample_size` | `Optional[int]` | `None` | Limit passages |

**Side effects:** Also saves `.ids.json` alongside the index.

### `save_embeddings()`

```python
def save_embeddings(
    path: str,
    embeddings: Optional[np.ndarray] = None,
    jsonl_path: Optional[str] = None,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    mock_mode: bool = False,
    dim: int = 768,
) -> str
```

Persist passage embeddings as a `.npy` file.

### `prepare_pyserini_collection()`

```python
def prepare_pyserini_collection(
    jsonl_path: str,
    out_dir: str,
    dev_sample_size: Optional[int] = None,
) -> str
```

Write Pyserini-compatible JSONL collection (`{"id", "contents"}`).

---

## 3. `factuality_rag.retriever`

### `HybridRetriever`

```python
class HybridRetriever:
    def __init__(
        self,
        faiss_index_path: str,
        pyserini_index_path: str,
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        alpha: float = 0.6,
        normalize: bool = True,
    ) -> None
```

Hybrid dense + sparse retriever with per-query score normalisation.

#### `retrieve(query, k=10, rerank=True)`

```python
def retrieve(
    self, query: str, k: int = 10, rerank: bool = True
) -> List[Dict[str, Any]]
```

**Return schema:**

```python
{
    "id": str,              # Document ID
    "text": str,            # Passage text
    "dense_score": float,   # Raw dense similarity (negated L2)
    "bm25_score": float,    # Raw BM25 score
    "dense_norm": float,    # Min-max normalised dense score [0,1]
    "bm25_norm": float,     # Min-max normalised BM25 score [0,1]
    "combined_score": float, # alpha * dense_norm + (1-alpha) * bm25_norm
    "metadata": dict,       # {"rank": int, ...}
}
```

**Normalisation:** Per-query min-max to [0,1]. Combined: `α * dense_norm + (1-α) * bm25_norm`.

#### `build_mock(dim=768, n_docs=20, seed=42, alpha=0.6)` (classmethod)

```python
@classmethod
def build_mock(cls, dim=768, n_docs=20, seed=42, alpha=0.6) -> HybridRetriever
```

Create an in-memory mock retriever for testing. No files needed.

---

## 4. `factuality_rag.gating`

### `GatingProbe`

```python
class GatingProbe:
    def __init__(
        self,
        generator_model_hf: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cuda",
        temp: float = 1.0,
        mock_mode: bool = False,
        model: Any = None,       # (new) pre-loaded model instance
        tokenizer: Any = None,   # (new) pre-loaded tokenizer
    ) -> None
```

Single-step logit probe for adaptive retrieval gating. Now loads models through the shared `model_registry` to avoid double-loading with the generator.

#### `should_retrieve(prompt, probe_tokens=1, entropy_thresh=1.2, logit_gap_thresh=2.0)`

```python
def should_retrieve(
    self,
    prompt: str,
    probe_tokens: int = 1,
    entropy_thresh: float = 1.2,
    logit_gap_thresh: float = 2.0,
) -> bool
```

**Decision rule:** `retrieve = (entropy > entropy_thresh) OR (logit_gap < logit_gap_thresh)`

**Multi-token mode (v0.3):** When `probe_tokens > 1`, uses `_get_multi_token_logits()` to run an autoregressive loop over `k` positions (greedy argmax per step). Entropy and logit gap are computed at each position and *averaged* across all `k` positions before applying thresholds. This produces a more stable gating signal than single-token probing.

**Returns:** `True` if retrieval should happen, `False` to skip.

#### `_get_multi_token_logits(prompt, k)` *(new in v0.3)*

```python
def _get_multi_token_logits(
    self,
    prompt: str,
    k: int = 3,
) -> List[torch.Tensor]
```

Autoregressive loop: for each of `k` positions, forward the current input through the model, extract the last-position logits, append the greedy argmax token, and continue. Returns a list of `k` logit tensors.

**Mock mode:** Returns `k` copies of the single mock logit vector.

#### `calibrate_temperature(dev_prompts, targets=None)`

```python
def calibrate_temperature(
    self,
    dev_prompts: List[str],
    targets: Optional[List[str]] = None,
) -> float
```

Grid search over T ∈ [0.5, 3.0] to minimise ECE. Now uses real binned ECE (v0.3) instead of entropy std-dev proxy.

---

### `compute_ece()` *(new in v0.3)*

```python
def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float
```

Module-level function. Computes Expected Calibration Error (Guo et al., 2017) using the standard binned approach.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `confidences` | `np.ndarray` | required | Max softmax probability per query |
| `accuracies` | `np.ndarray` | required | 1.0 if correct (EM=1), 0.0 otherwise |
| `n_bins` | `int` | `15` | Number of equal-width bins in [0, 1] |

**Returns:** ECE value in [0, 1] (lower is better).

---

## 5. `factuality_rag.scorer`

### `PassageScorer`

```python
class PassageScorer:
    def __init__(
        self,
        nli_model_hf: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        overlap_metric: str = "token",
        device: str = "cpu",
        mock_mode: bool = False,
        w_nli: float = 0.5,
        w_overlap: float = 0.2,
        w_ret: float = 0.3,
        nli_mode: str = "passage",                     # (new v0.3)
        cross_encoder_model: Optional[str] = None,     # (new v0.3)
    ) -> None
```

Passage-level factuality scorer with optional sentence-level NLI and cross-encoder reranking.

**New parameters (v0.3):**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `nli_mode` | `str` | `"passage"` | `"passage"` for full-passage NLI; `"sentence"` for sentence-level max scoring |
| `cross_encoder_model` | `Optional[str]` | `None` | HF cross-encoder model ID for reranking; `None` disables reranking |

#### `score_passages(query, passages)`

```python
def score_passages(
    self,
    query: str,
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]
```

Adds to each passage dict in-place:

| Key | Type | Description |
|-----|------|-------------|
| `nli_score` | `float` | P(entailment) from NLI model — **premise=passage, hypothesis=query** (fixed in v0.2) |
| `overlap_score` | `float` | Token or char overlap F1 |
| `final_score` | `float` | Weighted fusion: `w_nli*nli + w_overlap*overlap + w_ret*ret_norm` |
| `cross_encoder_score` | `float` | *(optional)* Cross-encoder relevance score (only when `cross_encoder_model` is set) |

**Sentence-level NLI (v0.3):** When `nli_mode="sentence"`, each passage is split into sentences and each sentence is scored independently via NLI. The passage receives the *maximum* sentence-level entailment score, addressing the problem where passages with one relevant sentence surrounded by noise scored low.

**Cross-encoder reranking (v0.3):** When `cross_encoder_model` is set, a reranking step runs before NLI scoring. The cross-encoder attends jointly to query+passage, producing more accurate relevance scores than bi-encoder retrieval. Passages are re-sorted and only top-k are passed to NLI.

#### `_split_sentences(text)` *(new in v0.3, static method)*

```python
@staticmethod
def _split_sentences(text: str) -> List[str]
```

Regex-based sentence splitting with abbreviation handling (Mr., Dr., U.S., etc.). Returns list of sentences.

#### `_sentence_level_nli(query, passage_text)` *(new in v0.3)*

```python
def _sentence_level_nli(self, query: str, passage_text: str) -> float
```

Score a passage by its best-scoring individual sentence. Falls back to passage-level NLI for single-sentence passages.

#### `_cross_encoder_rerank(query, passages, top_k=10)` *(new in v0.3)*

```python
def _cross_encoder_rerank(
    self,
    query: str,
    passages: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]
```

Rerank passages using the cross-encoder model. Adds `cross_encoder_score` to each passage dict. Returns passages sorted by cross-encoder score, limited to `top_k`.

#### `_load_cross_encoder()` *(new in v0.3)*

Lazy loader for the `sentence_transformers.CrossEncoder` model. Called on first use of `_cross_encoder_rerank()`.

---

## 6. `factuality_rag.generator`

### `Generator`

```python
class Generator:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        mock_mode: bool = False,
        model: Any = None,       # (new) pre-loaded model
        tokenizer: Any = None,   # (new) pre-loaded tokenizer
    ) -> None
```

LLM generator with lazy loading via `model_registry`. Shares weights with the gating probe (no double-loading). Uses Mistral `[INST]` chat template for RAG prompts.

#### `generate(query, context=None, passages=None)`

```python
def generate(
    self,
    query: str,
    context: Optional[str] = None,
    passages: Optional[List[str]] = None,
) -> str
```

**Mock return:** `"Mock answer for query: {query}"`

**Real mode:** Tokenises with Mistral `[INST]` template, generates via `model.generate()`, strips prompt prefix.

---

## 7. `factuality_rag.pipeline`

### `run_pipeline()`

```python
def run_pipeline(
    query: str,
    k: int = 10,
    gate: bool = True,
    score_threshold: float = 0.4,
    config_path: str = "configs/exp_sample.yaml",
    seed: int = 42,
    mock_mode: bool = False,
    *,
    probe: Optional[GatingProbe] = None,       # (new) pre-built component
    retriever: Optional[HybridRetriever] = None,
    scorer: Optional[PassageScorer] = None,
    generator: Optional[Generator] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]
```

Stateless convenience function. Accepts pre-built components via keyword args to avoid re-instantiation.

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `answer` | `str` | Generated answer |
| 1 | `trusted_passages` | `List[Dict]` | Passages with `final_score ≥ threshold` |
| 2 | `provenance` | `Dict[str, Any]` | `{claim_idx: [passage_ids]}` |
| 3 | `confidence_tag` | `str` | `"high"`, `"medium"`, or `"low"` |

**Confidence logic (fixed in v0.2):** Gating-skipped queries now return `"medium"` (not `"high"`) since there are no passages to verify the answer against.

**Provenance (v0.3):** The returned `provenance` dict is now built from `compute_factscore()` details via `_build_provenance()`. For each supported claim, it maps claim index to the passage ID(s) that provided entailment. Unsupported claims are absent from the dict.

**Config wiring (v0.3):** `nli_mode` and `cross_encoder_model` from YAML config are passed to `PassageScorer` constructor.

---

### `Pipeline` *(new in v0.2)*

```python
class Pipeline:
    def __init__(
        self,
        config_path: str = "configs/exp_sample.yaml",
        mock_mode: bool = False,
        seed: int = 42,
    ) -> None
```

Reusable pipeline that loads all components once at init, then reuses them across calls. Fixes the performance bug of re-instantiating every model on every query.

**Attributes:** `probe`, `retriever`, `scorer`, `generator`, `score_threshold`, `k`.

#### `run(query, *, k=None, gate=True, score_threshold=None, seed=None)`

```python
def run(
    self,
    query: str,
    *,
    k: Optional[int] = None,
    gate: bool = True,
    score_threshold: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]
```

Returns same 4-tuple as `run_pipeline()`.

---

## 8. `factuality_rag.eval`

### `compute_em(prediction, reference)`

Exact match (case-insensitive, stripped). Returns `1.0` or `0.0`.

### `compute_f1(prediction, reference)`

Token-level F1 score in [0, 1].

### `compute_factscore_stub(claims, passages)`

Stub: fraction of claims with >50% token overlap in any passage. Returns float in [0, 1]. Kept for backward compatibility; `compute_factscore()` is preferred.

### `decompose_claims(answer)` *(new in v0.2)*

```python
def decompose_claims(answer: str) -> List[str]
```

Split an answer into atomic claim sentences using sentence-boundary heuristics. Handles abbreviations. Returns list of claim strings (filtered: len > 3).

### `compute_factscore(answer, passages, nli_fn=None, entailment_threshold=0.7)` *(new in v0.2)*

```python
def compute_factscore(
    answer: str,
    passages: List[Dict[str, Any]],
    nli_fn: Optional[Callable[[str, str], float]] = None,
    entailment_threshold: float = 0.7,
) -> Dict[str, Any]
```

Real claim-level FactScore: decompose → per-claim NLI → aggregate.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `answer` | `str` | required | Generated answer |
| `passages` | `List[Dict]` | required | Passage dicts with `"text"` key |
| `nli_fn` | `Callable` | `None` | `(premise, hypothesis) → P(ent)`. Falls back to overlap if `None` |
| `entailment_threshold` | `float` | `0.7` | Min P(ent) for "supported" |

**Return schema:**

```python
{
    "factscore": float,       # supported / total claims
    "n_claims": int,
    "n_supported": int,
    "details": [
        {"claim": str, "supported": bool, "best_score": float, "best_passage_id": str},
        ...
    ],
}
```

### `evaluate_predictions(predictions, references=None, nli_fn=None)`

```python
def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    references: Optional[List[str]] = None,
    nli_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, float]
```

Batch evaluator. Returns `{"exact_match", "f1", "factscore", "n_predictions"}`. When `nli_fn` is provided, uses real claim-level FactScore; otherwise falls back to stub.

---

## 9. `factuality_rag.model_registry` *(new in v0.2)*

Singleton registry to avoid loading the same 7B model twice (e.g. once for gating, once for generation).

### `get_model(model_id, device="cuda", quantize_4bit=True, trust_remote_code=False)`

```python
def get_model(model_id: str, device: str = "cuda", quantize_4bit: bool = True, ...) -> Any
```

Returns a cached `AutoModelForCausalLM`. On first call, loads with optional 4-bit quantisation (bitsandbytes).

### `get_tokenizer(model_id, trust_remote_code=False)`

```python
def get_tokenizer(model_id: str, ...) -> Any
```

Returns a cached `AutoTokenizer`.

### `clear_registry()`

Remove all cached models and tokenizers.

### `is_loaded(model_id)`

Check whether a model is already cached. Returns `bool`.

---

## 10. `factuality_rag.cli`

Entry-point: `python -m factuality_rag.cli` or `factuality-rag` (if installed).

### Commands

| Command | Description | Key flags |
|---------|-------------|----------|
| `build_index` | Build FAISS + Pyserini indexes | `--corpus`, `--faiss-out`, `--mock-mode` |
| `chunk_wiki` | Chunk Wikipedia into JSONL | `--chunk-size`, `--chunk-overlap`, `--mock-mode` |
| `run` | Run full pipeline on a query | `--query`, `--k`, `--mock-mode`, `--no-gate` |
| `evaluate` | Evaluate predictions JSONL | `--predictions`, `--references` |

All commands support `--mock-mode` and `--dev-sample-size`.

The `run` command now uses `Pipeline` class internally (components loaded once).

---

## 11. `factuality_rag.experiment_runner`

### `run()`

```python
def run(
    config: Dict[str, Any],
    queries: Optional[List[str]] = None,
    config_path: str = "configs/exp_sample.yaml",
    mock_mode: bool = False,
    runs_dir: str = "runs",
) -> Dict[str, Any]
```

**Returns:**

```python
{
    "run_id": str,           # "20260228_145000_a1b2c3d4"
    "predictions": List,     # Per-query result dicts
    "metrics": Dict,         # Aggregated metrics
    "metadata": Dict,        # Full reproducibility info
    "run_dir": str,          # Path to saved run
}
```

### `build_metadata(config, config_path="", extra=None)`

Returns metadata dict with: `timestamp`, `git_commit`, `config_path`, `seed`, `models`, `datasets`, `library_versions`.

**Library versions tracked:** `faiss`, `datasets`, `transformers`, `sentence_transformers`.

The experiment runner now uses `Pipeline` class internally — components are loaded once and reused across all queries in a run (fixed re-instantiation performance bug).

---

## 12. `scripts` *(new in v0.3)*

Standalone analysis and experiment scripts. Not part of the `factuality_rag` library; run directly with `python scripts/<name>.py`.

### `build_corpus.py`

Build a Wikipedia chunk corpus with FAISS + Lucene indexes.

```bash
python scripts/build_corpus.py --sample-size 100000 --output data/wiki_100k_chunks.jsonl
```

### `analyze_gating.py`

Phase 4A — gating oracle analysis. Compares gating decisions against oracle (did retrieval actually help?) and computes precision/recall.

```bash
python scripts/analyze_gating.py --run-dir runs/<run-id>/
```

### `analyze_scorer.py`

Phase 4B — scorer quality analysis. Computes ROC-AUC, PR-AUC, and optimal threshold from passage-level labels.

```bash
python scripts/analyze_scorer.py --run-dir runs/<run-id>/
```

### `analyze_errors.py`

Phase 4C — error taxonomy. Classifies failures into `gating_miss`, `scoring_miss`, and `generation_miss` categories.

```bash
python scripts/analyze_errors.py --run-dir runs/<run-id>/
```

### `tune_scorer_weights.py`

Phase 5A-3 — grid search over scorer weight combinations `(w_nli, w_overlap, w_ret)` on a dev set to find the combination that maximises passage-level AUC.

```bash
python scripts/tune_scorer_weights.py --dev-predictions runs/<run-id>/predictions.jsonl
```

### `aggregate_results.py`

Cross-seed metric aggregation. Reads multiple run directories and produces a mean ± std table.

```bash
python scripts/aggregate_results.py --run-dirs runs/seed42/ runs/seed43/ runs/seed44/
```

### `bootstrap_test.py`

Paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). Compares two systems on the same test set.

```bash
python scripts/bootstrap_test.py --system-a runs/baseline/ --system-b runs/full/ --n-bootstrap 1000
```
