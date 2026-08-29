# Selected API Reference — Factuality-First RAG

> Version 0.4.0 · Updated 2026-08-29

This is a selected reference, not an exhaustive export list. Section names
group subpackages; symbols not re-exported by a package initializer must be
imported from their defining module. Underscore-prefixed methods are
implementation details.

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
    seed: int = 42,
) -> datasets.Dataset
```

Unified dataset loading wrapper around HuggingFace `datasets`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | required | Dataset identifier. Known enabled mappings are `"natural_questions"`/`"nq_open"`, `"hotpot_qa"`, and `"2wikimultihopqa"`; other names are passed through as HF paths unless explicitly disabled |
| `split` | `str` | `"train"` | Dataset split |
| `dev_sample_size` | `Optional[int]` | `None` | Deterministic development sample size; ignored in streaming mode |
| `streaming` | `bool` | `False` | Use HF streaming mode |
| `seed` | `int` | `42` | Seed used for the non-streaming development sample |

**Returns:** `datasets.Dataset` (or `IterableDataset` when streaming).

FEVER, TruthfulQA, PopQA, and HAGRID identifiers fail closed because the
project does not yet implement their task-specific prompting, metadata, and
evaluation adapters. Passing an arbitrary HF path through this low-level
loader does not imply that the experiment runner can evaluate its task schema.

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

Chunk Wikipedia articles into fixed whitespace-token-window passages with deduplication.

**Methods:**

#### `chunk_text(title, text)`

```python
def chunk_text(self, title: str, text: str) -> Generator[Dict[str, Any], None, None]
```

Yields dicts matching the JSONL schema:
`{"id", "title", "text", "tokens", "source", "mock_mode"}`.

#### `process_articles(articles, output_path=None)`

```python
def process_articles(
    self,
    articles: Iterable[Dict[str, str]],
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]
```

Chunks an iterable of `{"title", "text"}` dicts. JSONL writes are incremental,
but the function also retains and returns every chunk in memory; it is not a
large-corpus streaming builder.

#### `generate_mock_articles(n=10)`

```python
def generate_mock_articles(self, n: int = 10) -> List[Dict[str, str]]
```

Generate *n* synthetic articles for testing.

#### `load_from_hf(sample_size=None, output_path=None, wiki_config="20231101.en")` *(new in v0.2)*

```python
def load_from_hf(
    self,
    sample_size: Optional[int] = None,
    output_path: Optional[str] = None,
    wiki_config: str = "20231101.en",
) -> List[Dict[str, Any]]
```

Load the `wikimedia/wikipedia` Hugging Face stream and chunk its articles.
JSONL writes are incremental, but all chunks are retained and returned in memory.
This is not a snapshot-pinned benchmark ingester or a large-corpus streaming
builder.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sample_size` | `Optional[int]` | `None` | Method-level article cap; `None` applies no method-level cap, though the chunker's `dev_sample_size` may still limit processing |
| `output_path` | `Optional[str]` | `None` | Output JSONL path (auto-generated if omitted) |
| `wiki_config` | `str` | `"20231101.en"` | HuggingFace Wikipedia snapshot config |

**JSONL Output Schema:**
```json
{"id": "uuid5", "title": "Article Title", "text": "chunk text...", "tokens": 200, "source": "enwiki", "mock_mode": false}
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
| `hnsw_m` | `int` | `32` | HNSW graph degree when `faiss_type="hnsw_flat"` |
| `hnsw_ef_construction` | `int` | `200` | HNSW construction search width when `faiss_type="hnsw_flat"` |
| `dim` | `int` | `768` | Mock embedding width; real mode uses the encoder output width |
| `dev_sample_size` | `Optional[int]` | `None` | Limit passages |

**Side effects:** Also saves an ordered `.ids.json` mapping and a canonical
ordered `.jsonl` corpus sidecar alongside the index.

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

Defined in `factuality_rag.index.builder`; it is not re-exported by the
`factuality_rag.index` package initializer.

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
        *,
        corpus_path: Optional[str] = None,
    ) -> None
```

Hybrid dense + sparse retriever with optional per-query score normalisation.

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
    "dense_score": float,   # Raw inner product or negated squared-L2 score
    "bm25_score": float,    # Raw BM25 score
    "dense_norm": float,    # Min-max normalised dense score [0,1]
    "bm25_norm": float,     # Min-max normalised BM25 score [0,1]
    "combined_score": float, # Weighted normalised or raw components
    "metadata": dict,       # {"rank": int, ...}
}
```

**Normalisation:** With `normalize=True`, dense and sparse scores are separately
min-max scaled per query and combined as
`α * dense_norm + (1-α) * bm25_norm`. With `normalize=False`, the raw dense and
BM25 components are combined instead; the `dense_norm` and `bm25_norm` fields
remain zero-valued compatibility fields. `metadata.dense_metric` identifies
whether the dense score is an inner product or negated squared-L2 distance.

#### `build_mock(dim=768, n_docs=20, seed=42, alpha=0.6)` (classmethod)

```python
@classmethod
def build_mock(
    cls,
    dim: int = 768,
    n_docs: int = 20,
    seed: int = 42,
    alpha: float = 0.6,
) -> HybridRetriever
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
        nonfinite_policy: str = "raise",
        model: Any = None,       # (new) pre-loaded model instance
        tokenizer: Any = None,   # (new) pre-loaded tokenizer
    ) -> None
```

Next-token or multi-token logit probe for adaptive retrieval gating. It loads
models through the shared `model_registry` to avoid double-loading with the
generator. `nonfinite_policy="raise"` fails closed on invalid gate signals;
the explicit `"retrieve"` policy conservatively forces retrieval instead.

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

**Multi-token mode (v0.3):** When `probe_tokens > 1`, uses
`_get_multi_token_logits()` to run an autoregressive loop over `k` positions
(greedy argmax per step). Entropy and logit gap are computed at each position
and averaged across all `k` positions before applying thresholds. Whether this
improves stability, false-skip rate, latency, or end-to-end quality is not
established by this implementation.

**Returns:** `True` if retrieval should happen, `False` to skip.

#### `_get_multi_token_logits(prompt, k)` *(new in v0.3)*

```python
def _get_multi_token_logits(
    self,
    prompt: str,
    k: int = 3,
) -> List[np.ndarray]
```

Autoregressive loop: for each of `k` positions, forward the current input through the model, extract the last-position logits, append the greedy argmax token, and continue. Returns a list of `k` one-dimensional NumPy logit arrays.

**Mock mode:** Returns `k` deterministic, step-specific NumPy logit arrays
derived from the prompt and position; it does not load a model or repeat one
shared vector.

#### `calibrate_temperature(dev_prompts, targets=None)`

```python
def calibrate_temperature(
    self,
    dev_prompts: List[str],
    targets: Optional[List[str]] = None,
) -> NoReturn
```

**Disabled fail-closed boundary.** This method raises `NotImplementedError`
without loading a model or changing `self.temp`. The removed implementation
ignored `targets` and inferred correctness from the same logit gap it claimed
to calibrate. A replacement requires explicit ground-truth correctness targets
and held-out calibration data.

---

### `compute_ece()` *(new in v0.3)*

```python
def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float
```

Module-level function defined in `factuality_rag.gating.probe`; it is not
re-exported by the `factuality_rag.gating` package initializer. Computes the
**equal-width top-label Expected Calibration Error** variant: each confidence
is compared with an externally established binary correctness outcome, and
per-bin gaps are weighted by bin frequency. This metric does not fit a
calibrator and does not infer correctness from model scores.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `confidences` | `np.ndarray` | required | Max softmax probability per query |
| `accuracies` | `np.ndarray` | required | Externally established binary correctness outcome (0 or 1) per prediction |
| `n_bins` | `int` | `15` | Number of equal-width bins in [0, 1] |

Inputs must be non-empty, one-dimensional, equal-length arrays. Confidences must be finite real values in [0, 1], correctness outcomes must be finite binary values, and `n_bins` must be a positive non-boolean integer. Invalid inputs raise instead of producing a potentially misleading score.

**Returns:** Equal-width top-label ECE in [0, 1] (lower is better).

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
        nli_batch_size: int = 8,
        cross_encoder_model: Optional[str] = None,     # (new v0.3)
    ) -> None
```

Passage-level evidence/relevance scorer with optional sentence-unit NLI and cross-encoder reranking. The NLI term uses the passage as premise and the query as hypothesis; it is not an answer-factuality judgment.

**Selected scoring parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `nli_mode` | `str` | `"passage"` | `"passage"` for full-passage NLI; `"sentence"` for the maximum over regex-derived sentence-like units |
| `nli_batch_size` | `int` | `8` | Positive Transformers pipeline batch size for full-passage NLI; sentence-unit mode continues to score one unit at a time |
| `cross_encoder_model` | `Optional[str]` | `None` | HF cross-encoder model ID for reranking; `None` disables reranking |

#### `score_passages(query, passages)`

```python
def score_passages(
    self,
    query: str,
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]
```

Returns passage dictionaries with the following score fields. Without a
cross-encoder, the input dictionaries receive these fields in-place. With a
cross-encoder, shallow copies are scored and returned in reranked order:

| Key | Type | Description |
|-----|------|-------------|
| `nli_score` | `float` | Query–passage evidence signal from the NLI model — **premise=passage, hypothesis=query**; not a factuality probability |
| `overlap_score` | `float` | Token-overlap F1 or character-set Jaccard, selected by `overlap_metric` |
| `final_score` | `float` | Weighted fusion: `w_nli*nli + w_overlap*overlap + w_ret*ret_norm` |
| `cross_encoder_score` | `float` | *(optional)* Cross-encoder relevance score (only when `cross_encoder_model` is set) |

**Sentence-unit NLI (v0.3):** When `nli_mode="sentence"`, a simple punctuation-plus-whitespace regex produces heuristic units, strips terminal punctuation, and removes fragments shorter than four characters. It has no abbreviation handling or linguistic sentence parser. Each retained unit is scored independently, and the passage receives the maximum unit-level entailment score.

**Cross-encoder reranking (v0.3):** When `cross_encoder_model` is set, a reranking step runs before NLI scoring. It adds `cross_encoder_score` to shallow passage copies and reorders the complete retrieved list. The implementation does not apply a second top-k truncation; every reranked passage proceeds to NLI scoring.

**Full-passage NLI batching:** In real `nli_mode="passage"`, all passage/query
pairs are submitted through one Transformers pipeline call with `batch_size`
set to `nli_batch_size`. Mock mode and sentence-unit mode retain their
single-pair semantics. Real NLI truncates only the passage premise and fails if
the complete query/hypothesis cannot fit within the effective model limit.

#### `_split_sentences(text)` *(new in v0.3, static method)*

```python
@staticmethod
def _split_sentences(text: str) -> List[str]
```

Regex-based heuristic splitting on sentence-ending punctuation followed by whitespace. It strips terminal punctuation and drops fragments shorter than four characters; it does not handle abbreviations specially or produce atomic facts.

#### `_sentence_level_nli(query, passage_text)` *(new in v0.3)*

```python
def _sentence_level_nli(self, query: str, passage_text: str) -> float
```

Score a passage by its best-scoring regex-derived unit. It calls passage-level NLI only when the splitter returns no retained units; a single retained unit is scored directly.

#### `_cross_encoder_rerank(query, passages)` *(new in v0.3)*

```python
def _cross_encoder_rerank(
    self,
    query: str,
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]
```

Rerank passages using the cross-encoder model. Returns shallow passage copies with `cross_encoder_score`, sorted by that score. It preserves the number of input passages and does not mutate the caller's list or dictionaries.

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

LLM generator with lazy loading via `model_registry`. Compatible generator and
gating requests reuse the same cached model object. The current implementation
uses a Mistral `[INST]` prompt even though `model_name` is configurable; prompt
compatibility with alternative model families is not validated.

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

**Real mode:** Tokenises with the Mistral `[INST]` template, calls
`model.generate()`, and decodes only token IDs generated after the prompt.

---

## 7. `factuality_rag.pipeline`

### `run_pipeline()`

```python
def run_pipeline(
    query: str,
    k: int = 10,
    gate: bool = True,
    score_threshold: float = 0.4,
    config_path: Optional[str] = None,
    seed: int = 42,
    mock_mode: bool = False,
    *,
    config: Optional[Dict[str, Any]] = None,
    probe: Optional[Any] = None,       # pre-built component
    retriever: Optional[Any] = None,
    scorer: Optional[Any] = None,
    learned_scorer: Optional[Any] = None,
    generator: Optional[Any] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]
```

One-shot functional convenience API. It reseeds the process-global Python and
NumPy random generators; compatible model-registry cache entries can persist
across calls. An omitted `config_path` loads the packaged `exp_sample.yaml`; an
explicit `config` mapping takes precedence. Pre-built components can be supplied
through keyword arguments to avoid re-instantiation.

**Returns:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `answer` | `str` | Generated answer |
| 1 | `trusted_passages` | `List[Dict]` | Compatibility name for passages with `final_score ≥ threshold`; when `scorer.enabled=false`, all retrieved passages pass through without that threshold. Neither case means verified truth |
| 2 | `provenance` | `Dict[str, Any]` | Compatibility name for `{str(sentence_unit_idx): [best_passage_id]}`; unsupported units map to `[]`, and scorer-bypass mode returns `{}` |
| 3 | `confidence_tag` | `str` | `"high"`, `"medium"`, or `"low"` |

**Confidence logic (fixed in v0.2):** Gating-skipped queries return `"medium"` rather than `"high"` because no passage-conditioned evidence score is available. The tag is qualitative, not a calibrated probability.

**Heuristic evidence links (`provenance` compatibility field, v0.3):**
`_build_provenance()` splits the generated answer into regex-derived sentence
units and compares every unit with the selected passages using the scorer's NLI
callable. It currently uses the support primitive's default threshold of 0.7.
Each retained unit appears in the mapping: a supported unit maps to its single
best passage ID and an unsupported unit maps to `[]`. This is neither atomic
claim provenance nor human adjudication, FactScore output, or calibrated
confidence.

**Config wiring:** `nli_mode`, `nli_batch_size`, and `cross_encoder_model` from
YAML config are passed to the `PassageScorer` constructor.

---

### `Pipeline` *(new in v0.2)*

Import this class from `factuality_rag.pipeline.orchestrator`. The
`factuality_rag.pipeline` package namespace currently re-exports
`run_pipeline` only.

```python
class Pipeline:
    def __init__(
        self,
        config_path: Optional[str] = None,
        mock_mode: bool = False,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None
```

Reusable pipeline that instantiates component wrappers once and reuses them
across calls. Heavy models and indexes remain lazy-loaded and are cached when
first needed.

An omitted `config_path` loads packaged `exp_sample.yaml`; an explicit
in-memory `config` takes precedence.

**Attributes:** `probe`, `retriever`, `scorer`, `generator`, `score_threshold`,
`k`, `gate_enabled`, and `scorer_enabled`.

#### `run(query, *, k=None, gate=None, score_threshold=None, seed=None, info=None)`

```python
def run(
    self,
    query: str,
    *,
    k: Optional[int] = None,
    gate: Optional[bool] = None,
    score_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]
```

Returns the same 4-tuple as `run_pipeline()`. `gate=None` preserves the exact
`gating.enabled` value from the config; an explicit boolean overrides it.
When supplied, `info` is populated with run metadata such as
`retrieval_triggered`, `gating_enabled`, and `scorer_enabled`.

---

## 8. `factuality_rag.eval`

### `compute_em(prediction, reference)`

```python
def compute_em(prediction: str, reference: str) -> float
```

SQuAD/NQ-open-style normalized exact match: lowercase, remove ASCII punctuation
and English articles, then collapse whitespace. Returns `1.0` or `0.0`.

### `compute_f1(prediction, reference)`

```python
def compute_f1(prediction: str, reference: str) -> float
```

Token-overlap F1 in [0, 1] after the same normalization as `compute_em()`.

### `compute_lexical_support(claims, passages)`

```python
def compute_lexical_support(
    claims: List[str],
    passages: List[Dict[str, Any]],
) -> float
```

Explicitly labelled lexical-overlap diagnostic. It is never returned under a
FactScore key and is not a substitute for entailment verification.

### `decompose_claims(answer)` *(new in v0.2)*

```python
def decompose_claims(answer: str) -> List[str]
```

Split an answer into heuristic sentence-like units at sentence-ending punctuation followed by whitespace, strip terminal punctuation, and drop fragments shorter than four characters. Despite the compatibility name, this performs no atomic-fact decomposition or abbreviation handling.

### `compute_nli_claim_support(answer, passages, *, nli_fn, nli_batch_fn=None, entailment_threshold=0.7)`

```python
def compute_nli_claim_support(
    answer: str,
    passages: List[Dict[str, Any]],
    *,
    nli_fn: Callable[[str, str], float],
    nli_batch_fn: Optional[
        Callable[[List[Tuple[str, str]]], Sequence[float]]
    ] = None,
    entailment_threshold: float = 0.7,
) -> Dict[str, Any]
```

Standalone sentence-unit NLI evidence-support primitive: heuristic split →
validate explicit per-passage entailment probabilities → aggregate. It has no
lexical fallback. Compatibility keys retain the word `claim`, but each item is
a regex-derived sentence unit rather than an atomic fact. When at least one pair
exists and `nli_batch_fn` is supplied, it receives every
`(passage_text, sentence_unit)` pair in stable sentence-unit-major order and
must return exactly one ordered probability per pair. When it is omitted,
`nli_fn` is called once per pair. `nli_fn` remains required and callable in
either mode; neither path has a lexical fallback. The batch evaluator does not
expose this as FactScore, and the current regex units are not atomic facts.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `answer` | `str` | required | Generated answer |
| `passages` | `List[Dict]` | required | Passage dicts with `"text"` key |
| `nli_fn` | `Callable` | required keyword | `(premise, hypothesis) → finite P(ent) in [0, 1]`; invalid or missing scorers fail closed |
| `nli_batch_fn` | `Optional[Callable]` | `None` | Optional ordered batch callback over all `(premise, hypothesis)` pairs; its result length and every probability are validated |
| `entailment_threshold` | `float` | `0.7` | Min P(ent) for "supported" |

**Return schema:**

```python
{
    "nli_claim_support": float,  # supported / sentence-split claims
    "n_claims": int,
    "n_supported": int,
    "details": [
        {"claim": str, "supported": bool, "best_score": float, "best_passage_id": Any},
        ...
    ],
}
```

### `evaluate_predictions(predictions, references=None, *, support_metric="none", nli_fn=None)`

```python
def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    references: Optional[Sequence[Reference]] = None,
    *,
    support_metric: str = "none",
    nli_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, Any]
```

Batch evaluator. It returns EM/F1 when references are supplied, explicit answer
coverage/counts (excluding blank answers and the canonical abstention response),
and an optional `lexical_support_*` diagnostic only when
`support_metric="lexical"`. The default is `"none"`. Supplying `nli_fn` fails
closed because batch NLI/FactScore aggregation is not implemented and the
current regex units are not atomic facts; no FactScore key is emitted.

---

## 9. `factuality_rag.model_registry` *(new in v0.2)*

Singleton registry through which compatible gating and generation requests reuse the same cached model; incompatible load settings raise rather than reusing a mismatched object.

### `get_model(model_id, device="cuda", quantize_4bit=True, trust_remote_code=False)`

```python
def get_model(
    model_id: str,
    device: str = "cuda",
    quantize_4bit: bool = True,
    trust_remote_code: bool = False,
) -> Any
```

Returns a cached `AutoModelForCausalLM`. On first call, 4-bit mode requires both `bitsandbytes` and `accelerate` and fails closed when either is unavailable; there is no automatic full-precision fallback. Callers may explicitly set `quantize_4bit=False`.

### `get_tokenizer(model_id, trust_remote_code=False)`

```python
def get_tokenizer(
    model_id: str,
    trust_remote_code: bool = False,
) -> Any
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
| `build_index` | Build FAISS + Pyserini indexes from an explicit corpus | `--corpus`, `--faiss-out`, `--pyserini-out`, `--dev-sample-size`, `--dry-run`, `--mock-mode` |
| `chunk_wiki` | Generate explicitly marked synthetic chunks; currently requires `--mock-mode` | `--output`, `--chunk-size`, `--chunk-overlap`, `--dev-sample-size`, `--dry-run`, `--mock-mode` |
| `run` | Run the full pipeline on a query | `--query`, `--k`, `--config`, `--mock-mode`, `--no-gate` |
| `evaluate` | Evaluate predictions JSONL | `--predictions`, `--references`, `--support-metric {none,lexical}` |

Flags are command-specific: `evaluate` has neither `--mock-mode` nor
`--dev-sample-size`, and `run` has no `--dev-sample-size`. `chunk_wiki` has no
real dump parser: omitting `--mock-mode` fails closed, while combining
`--input` with `--mock-mode` is rejected. Use `scripts/build_corpus.py` for the
current HuggingFace acquisition path. `build_index --mock-mode` replaces only
the dense embeddings with deterministic random vectors; the Lucene index is
still built from the supplied corpus.

The `run` command uses the reusable `Pipeline` class internally. Component
wrappers are instantiated once; heavy models and indexes remain lazy-loaded.

---

## 11. `factuality_rag.experiment_runner`

### `run()`

```python
def run(
    config: Dict[str, Any],
    queries: Optional[Sequence[str]] = None,
    references: Optional[Sequence[Reference]] = None,
    config_path: str = "",
    mock_mode: bool = False,
    runs_dir: str = "runs",
    run_id_prefix: Optional[str] = None,
    support_metric: Optional[str] = None,
    resume_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]
```

`queries` must be an explicit non-empty ordered sequence of already-trimmed,
non-blank strings without control characters. `None` raises; there is no
demo-query fallback. Optional `references` must be parallel to `queries` and
contain a string or non-empty alias sequence for each item. An empty
`config_path` identifies the supplied config as in-memory rather than pretending
it came from a sample file.

`support_metric` accepts only `"none"` or `"lexical"`. When omitted, the
configuration may request the lexical diagnostic with
`eval.metrics: [lexical_support]`; legacy FactScore metric names are rejected
rather than relabelled.

`resume_dir` resumes an existing run and is mutually exclusive with
`run_id_prefix`. A new run creates an integrity-hashed `resume_manifest.json`
before an empty `predictions.jsonl`, then validates, appends, flushes, and
`fsync`s each completed record. Resume is accepted only when the manifest's
config, input, execution, Git/runtime-source, and library-version bindings match
the current invocation. Existing checkpoint rows are strict-schema validated
against their expected query/reference positions; only a final non-newline
fragment is treated as a torn write and discarded. Processing continues from
the first unfinished row, and the original run-start metadata timestamp is
preserved. A fully populated valid checkpoint is evaluated without rebuilding
the pipeline. Final evaluation writes or refreshes the metrics and metadata
artifacts, plus reference artifacts when references are present, in that run
directory.

**Returns:**

```python
{
    "run_id": str,           # "20260228_145000_a1b2c3d4"
    "predictions": List,     # Per-query result dicts
    "metrics": Dict,         # Aggregated metrics
    "metadata": Dict,        # Recorded identities/versions; not proof of reproducibility
    "run_dir": str,          # Path to saved run
}
```

### `build_metadata(config, config_path="", extra=None)`

```python
def build_metadata(
    config: Dict[str, Any],
    config_path: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]
```

Returns metadata including timestamp, source-root-verified full Git revision and
dirty state (or an unavailable sentinel), a path-free config identity, the
effective-config hash and (when a file or package resource is identified) exact
source-byte hash, seed, models, datasets, and library versions.
These fields support later reproduction checks but do not by themselves prove
that datasets, indexes, models, or results are reproducible.

**Library versions tracked:** `faiss`, `datasets`, `transformers`, `sentence_transformers`.

The experiment runner uses the reusable `Pipeline` class internally. Component
wrappers are instantiated once and reused across all queries in a run; heavy
models and indexes remain lazy-loaded.

---

## 12. `scripts` *(new in v0.3)*

Standalone analysis and experiment scripts. Not part of the `factuality_rag` library; run directly with `python scripts/<name>.py`.

### `build_corpus.py`

Build a Wikipedia chunk corpus with FAISS + Lucene indexes.

```bash
python scripts/build_corpus.py --n-docs 100000 \
    --faiss-out indexes/wiki100k.faiss \
    --pyserini-out indexes/wiki100k_lucene
```

### `analyze_gating.py`

Exploratory gating proxy analysis. It compares gate decisions with closed-book
exact-match errors; it does not estimate a retrieval oracle or retrieval
utility, and stamps its output as not publication-safe.

```bash
python scripts/analyze_gating.py --full-run runs/<full-id>/ --closedbook-run runs/<cb-id>/
```

### `analyze_scorer.py`

Exploratory scorer discrimination analysis over independently supplied passage
judgments. It reports ROC-AUC, average precision, and an in-sample threshold;
the threshold is not confirmatory.

```bash
python scripts/analyze_scorer.py --predictions runs/<run-id>/predictions.jsonl \
    --judgments data/scorer_judgments.json
```

### `analyze_errors.py`

The CLI intentionally raises instead of emitting a causal error taxonomy. The
retained compatibility helper reports only `correct`, `unadjudicated`, or
`unknown`; it does not infer gating, retrieval, scoring, or generation causes.

```bash
python scripts/analyze_errors.py  # intentionally fails closed
```

### `train_scorer.py`

The training CLI intentionally raises and cannot create a model artifact. The
generic `LearnedScorer` API remains available to callers that already have
feature and label arrays.

```bash
python scripts/train_scorer.py  # intentionally fails closed
```

### `tune_scorer_weights.py`

The tuning CLI intentionally raises and cannot select weights. A pure
`generate_weight_grid()` helper remains available, and `analyze_scorer.py` is
the supported independently labelled analysis path.

```bash
python scripts/analyze_scorer.py --predictions runs/<run-id>/predictions.jsonl --judgments data/scorer_judgments.json
```

### `aggregate_results.py`

Cross-seed metric aggregation. Discovers run subdirectories beneath
`--runs-dir`, with optional exact config and directory-name filters. Strict mode
requires sealed, compatible run bundles; all current outputs remain stamped
`publication_safe=false`. `--exploratory` permits legacy/unsealed inputs but is
also explicitly not claim-safe.

```bash
python scripts/aggregate_results.py --runs-dir runs \
    --pattern "full_nq_500_s*" \
    --output analysis/aggregated_results.json
```

### `bootstrap_test.py`

One-sided paired bootstrap test comparing two systems on the same reference-bound
examples. `--system-a` and `--system-b` take the two `predictions.jsonl` files,
not run directories. Strict mode validates the sealed run bundles; all current
outputs remain stamped `publication_safe=false`.

```bash
python scripts/bootstrap_test.py \
    --system-a runs/baseline/predictions.jsonl \
    --system-b runs/full/predictions.jsonl \
    --metric exact_match --n-bootstrap 10000 \
    --output analysis/bootstrap_test.json
```
