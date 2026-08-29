# Architecture — Factuality-First RAG

> Version 0.4.0 · implemented research-stage architecture

## 1. Scope

The shipped system is a research prototype for adaptive retrieval and passage
evidence selection. It decides whether to retrieve from generator uncertainty,
fuses dense and sparse candidates, scores passages against the query, and
generates with selected context when available.

Query–passage NLI is an evidence/relevance signal, not certification that a
generated answer is factually correct. The returned confidence tag is
qualitative and is not a calibrated probability.

### Public future-work boundary

Future work may split retrieval into a genuine BM25-first stage that can stop
before dense work and add a separate verifier-assisted answer-or-abstain path.
The current retriever always executes dense and sparse search in one call, and
that controller and verifier are not implemented.

## 2. Runtime data flow

~~~mermaid
flowchart TD
    Q["User query"] --> G{"Entropy / logit-gap gate"}
    G -->|"confident"| N["No-context generation"]
    G -->|"uncertain"| R["Hybrid retrieval<br/>FAISS + Lucene BM25"]
    R --> S["Passage scoring<br/>NLI + overlap + retrieval"]
    S --> F{"Score meets threshold?"}
    F -->|"one or more"| C["Context-conditioned generation"]
    F -->|"none"| N
    C --> O["Answer + selected passages<br/>+ heuristic evidence links + tag"]
    N --> O
~~~

The gate probes the same no-context instruction prompt that generation would
receive after a retrieval skip. Retrieval is requested when either:

- entropy exceeds <code>entropy_thresh</code>; or
- the top-two logit gap is below <code>logit_gap_thresh</code>.

With the shipped defaults those thresholds are 1.2 and 2.0. A disabled gate
always proceeds to retrieval. A request with <code>k=0</code> produces no
retrieval candidates and therefore uses no-context generation.

## 3. Module map

~~~text
factuality_rag/
├── abstention.py              # Canonical abstention text and matcher
├── data/
│   ├── loader.py              # Dataset adapters
│   ├── splits.py              # Dataset split artifacts
│   └── wikipedia.py           # Wikipedia chunking
├── index/builder.py           # FAISS and Lucene index construction
├── retriever/
│   ├── hybrid.py              # Dense+sparse retrieval and fusion
│   └── pyserini_worker.py     # Isolated Anserini search boundary
├── gating/probe.py            # Entropy/logit-gap retrieval gate
├── scorer/
│   ├── passage.py             # NLI, overlap, and retrieval fusion
│   └── learned_scorer.py      # Optional authenticated scorer loading
├── generator/wrapper.py       # Prompt formatting and causal-LM generation
├── pipeline/orchestrator.py   # Functional and reusable pipeline APIs
├── eval/                      # Answer, support, and retrieval metrics
├── experiment_runner.py       # Durable checkpoints, resume, and run outputs
├── model_registry.py          # Shared model/tokenizer cache
└── reproducibility.py         # Artifact identities and manifest validation
~~~

Large model weights, corpora, generated indexes, and run directories are
external artifacts rather than package resources.

## 4. Components

### 4.1 Gating probe

<code>GatingProbe</code> reads the configured causal model's next-token logits,
applies the configured softmax temperature, and calculates:

- entropy over the vocabulary distribution; and
- the difference between the two largest logits.

The default <code>probe_tokens=1</code> uses one position. Larger values run an
autoregressive greedy probe and average both signals across positions. A fixed
softmax temperature is only an input to the calculation; it does not establish
calibration.

The probe and generator lazy-load through the same model registry, so compatible
requests share model and tokenizer objects. Cache requests with incompatible
device, precision, or remote-code settings fail rather than reuse mismatched
objects.

### 4.2 Hybrid retrieval

#### Dense path

The query is encoded with the configured sentence-transformer and searched
against the bound FAISS index. The loader validates that vectors, ordered
document IDs, and passage texts align exactly.

- Inner-product indexes use L2-normalized query vectors.
- Squared-L2 distances are negated before score fusion so larger values remain
  better.
- Unknown FAISS metrics fail rather than being interpreted heuristically.

#### Sparse path

Sparse search is Lucene BM25 through Pyserini's Java bridge to Anserini
<code>io.anserini.search.SimpleSearcher</code>.

On Windows, each request runs behind the short-lived
<code>pyserini_worker</code> process boundary. The parent binds the request to
the resolved index, enforces size and timeout limits, and validates the
single-response JSON before accepting hits. On other platforms the retriever
lazy-loads and caches the same Anserini class in process.

Real sparse retrieval requires a readable Lucene index and Java 21. The worker
uses <code>FACTUALITY_RAG_JAVA_HOME</code> when set, otherwise
<code>JAVA_HOME</code>. Missing or malformed sparse state fails closed rather
than silently relabelling a dense-only result as hybrid retrieval.

#### Fusion

Dense and sparse top-k results are unioned by document ID. With normalization
enabled, each source is min-max scaled per query, and constant non-empty source
scores map to 1.0. Missing-source contributions map to 0.0.

<code>combined = alpha*dense_component + (1-alpha)*bm25_component</code>

The default <code>alpha</code> is 0.6. Both searches occur before fusion, so
setting <code>alpha=0</code> changes ranking weights but does not avoid dense
encoding or FAISS search.

### 4.3 Passage evidence scorer

The scorer treats the passage as NLI premise and the query as hypothesis. Its
default fusion is:

<code>final_score = 0.5*nli + 0.2*overlap + 0.3*retriever</code>

The retriever contribution is normalized again across the candidate set. If all
candidate combined scores are equal, each receives 0.5 for this scorer term.
Passages below the effective threshold, 0.4 by default, are removed before
generation.

Two NLI modes are available:

- <code>passage</code> builds all passage/query pairs and sends them through one
  Transformers pipeline call. The positive <code>nli_batch_size</code> controls
  that call's batching and defaults to 8.
- <code>sentence</code> splits passage text with the lightweight regex unit
  splitter and uses the largest single-unit entailment score. It uses the
  single-pair scoring path.

The real NLI path protects the hypothesis from truncation with
<code>truncation="only_first"</code>, validates model-aware sequence limits, and
requires one unambiguous entailment probability per input. Mock scores are
deterministic for fixed inputs.

When configured, a cross-encoder first reorders the complete retrieved list.
It does not apply a second top-k truncation; every reordered passage proceeds to
the NLI scorer.

### 4.4 Generator and model registry

The current orchestrator constructs <code>Generator</code> with its wrapper
defaults:

| Setting | Effective value |
|---|---|
| Model | <code>mistralai/Mistral-7B-Instruct-v0.3</code> |
| Device | <code>cuda</code> |
| <code>max_new_tokens</code> | 256 |
| <code>temperature</code> | 0.1 |
| <code>do_sample</code> | <code>false</code> |

These decoding values currently live in code rather than the YAML pipeline
configuration. The generation call passes the temperature with a minimum value
of 0.0001, uses the tokenizer's end-of-sequence token for padding, and decodes
only tokens generated after the prompt.

With context, the instruction says to answer only from that context and to emit
the exact canonical abstention when support is absent:

~~~text
I cannot answer based on the provided context.
~~~

Without selected context, the prompt asks for a concise answer. The current API
does not produce structured citations.

Model loading requests 4-bit weights by default through:

<code>BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=float16)</code>

Missing bitsandbytes or Accelerate support fails before loading; there is no
automatic full-precision fallback in the pipeline path. Direct registry callers
may explicitly request <code>quantize_4bit=False</code>.

### 4.5 Orchestrator and return value

Two entry points expose the same pipeline:

- <code>run_pipeline()</code> is the functional interface and accepts optional
  prebuilt components.
- <code>Pipeline</code> creates reusable wrappers once while keeping heavy
  models and indexes lazy-loaded.

Both return:

~~~text
(answer, selected_passages, provenance, confidence_tag)
~~~

The compatibility field <code>provenance</code> contains model-derived
sentence-unit-to-passage evidence links. Answer text is split into lightweight
regex units; a supported unit maps to one best-scoring passage ID, and an
unsupported unit maps to an empty list. The ordered NLI batch hook is used when
it is compatible with any scorer customization.

These links are not atomic-fact decomposition or human-adjudicated provenance.
The confidence tag is assigned as follows:

| Condition | Tag |
|---|---|
| Gate skipped retrieval | <code>medium</code> |
| No selected passages | <code>low</code> |
| Mean selected score at least 0.70 | <code>high</code> |
| Mean selected score at least 0.45 | <code>medium</code> |
| Lower mean selected score | <code>low</code> |

## 5. Evaluation surfaces

The evaluation package provides normalized Exact Match and token F1, retrieval
metrics, and an optional explicitly labelled lexical-support diagnostic. The
standalone evaluate command defaults to no support metric.

<code>compute_nli_claim_support()</code> requires an explicit NLI callable and
operates over the same regex-derived sentence-like units. It does not implement
atomic-fact decomposition, and the batch evaluator does not report FactScore.

## 6. Durable experiment runs

<code>experiment_runner</code> writes a new run beneath
<code>runs/&lt;run-id&gt;/</code>. Before the query loop it:

1. builds and fsyncs <code>resume_manifest.json</code>, binding the run identity,
   inputs, effective settings, and source metadata;
2. creates an empty <code>predictions.jsonl</code> checkpoint.

Every completed prediction record is validated, serialized as strict finite
JSON, appended with a newline, flushed, and fsynced. Resume mode validates the
manifest against the current invocation, validates the complete checkpoint
prefix in dataset order, and processes only the unfinished suffix. It may
discard one final non-newline fragment as an uncommitted torn write; malformed
newline-terminated records fail closed.

After the query loop, the runner writes:

~~~text
runs/<run-id>/
├── resume_manifest.json
├── predictions.jsonl
├── metrics.json
├── metadata.json
├── references_by_example_id.json   # when references are available
└── references.json                 # unambiguous-query compatibility map
~~~

The ordinary runner records structured research outputs; it does not by itself
establish a benchmark result.

## 7. Effective configuration

The package-distributed sample YAML supplies runtime defaults when no explicit
path is provided. Explicit configuration paths are treated literally and fail
if missing or malformed.

| Parameter | Effective default |
|---|---:|
| <code>retriever.alpha</code> | 0.6 |
| <code>retriever.top_k</code> | 10 |
| <code>gating.entropy_thresh</code> | 1.2 |
| <code>gating.logit_gap_thresh</code> | 2.0 |
| <code>gating.probe_tokens</code> | 1 |
| <code>gating.softmax_temperature</code> | 1.0 |
| <code>scorer.score_threshold</code> | 0.4 |
| <code>scorer.weights</code> | 0.5 / 0.2 / 0.3 |
| <code>scorer.nli_mode</code> | <code>passage</code> |
| <code>scorer.nli_batch_size</code> | 8 |
| <code>scorer.cross_encoder_model</code> | <code>null</code> |

Generator decoding defaults remain code settings, as described in section 4.4.

## 8. Mock-mode boundary

Mock mode avoids model-weight loading and GPU execution for direct pipeline
queries:

| Component | Mock behavior |
|---|---|
| Index builder | Deterministic synthetic vectors |
| Retriever | Deterministic dense and BM25 scores |
| Gate | Deterministic synthetic logits |
| Passage scorer | Deterministic NLI scores |
| Generator | Deterministic query-labelled answer |
| Model registry | Not called |

The experiment runner can still load its selected dataset when mock mode is
enabled unless that dataset is already cached. Mock execution is a development
and test surface, not evidence about real-model quality or performance.

## 9. Artifact and scaling boundaries

- Model checkpoints, corpus snapshots, generated indexes, and run directories
  remain outside the source package.
- Lucene index construction uses temporary output followed by final placement.
  FAISS vectors and their ordered ID/text sidecars form one bound artifact set
  and are validated together when loaded.
- Source and run metadata record identities and content hashes where the
  relevant API requires them.
- No production throughput, latency, memory, deployment, or quality claim is
  established by this architecture description.

Possible future engineering work includes sparse-first execution, structured
citations, verifier-assisted abstention, compressed or sharded indexes, and
batched query processing. Each remains unimplemented until represented by code
and independently reproducible artifacts.
