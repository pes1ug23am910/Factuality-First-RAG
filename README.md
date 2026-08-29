# Factuality-First RAG

**Research-stage adaptive retrieval and passage evidence scoring.** This
repository implements a Retrieval-Augmented Generation pipeline that uses
generator uncertainty to decide whether retrieval is needed, combines dense
and sparse retrieval, scores passages before generation, and records structured
run artifacts.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-0.4.0-orange.svg)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)

## Motivation

Retrieving for every question can add compute and expose generation to
irrelevant context. This project studies a different operating point:

1. **Adaptive retrieval gating** inspects entropy and the top-two logit gap from
   the configured generator. A confident probe can skip retrieval; an uncertain
   probe triggers it.
2. **Passage evidence scoring** combines query-passage NLI, lexical overlap, and
   retrieval strength before passages reach the generator.

The NLI term uses the passage as premise and the query as hypothesis. It is
therefore an evidence/relevance signal, not a judgment that a generated answer
is factually correct. The returned confidence tag is qualitative and is not a
calibrated probability.

## How the implemented pipeline works

1. **Gate.** Retrieve when
   <code>entropy &gt; entropy_thresh</code> or
   <code>logit_gap &lt; logit_gap_thresh</code>. The default probe examines one
   generated position; <code>probe_tokens &gt; 1</code> averages signals across
   autoregressive positions.
2. **Retrieve.** Search a bound FAISS index and a Lucene BM25 index, normalize
   scores per query, and fuse them with
   <code>combined = alpha*dense + (1-alpha)*BM25</code>.
3. **Score and filter.** Compute
   <code>0.5*NLI + 0.2*overlap + 0.3*retrieval</code> by default and retain
   passages at or above the configured threshold.
4. **Generate.** Use an <code>[INST]</code> context prompt when passages
   survive. A gate skip or empty filtered set uses the no-context prompt.
5. **Record.** Return the answer, selected passages, heuristic sentence-unit
   evidence links under the compatibility field name <code>provenance</code>,
   and a qualitative confidence tag.

~~~mermaid
flowchart TD
    Q["User query"] --> G{"Uncertainty gate"}
    G -->|"confident"| N["No-context generation"]
    G -->|"uncertain"| R["FAISS + BM25 retrieval"]
    R --> S["NLI + overlap + retrieval scoring"]
    S --> F{"Passes threshold?"}
    F -->|"yes"| C["Context-conditioned generation"]
    F -->|"none survive"| N
    C --> O["Answer + selected passages + evidence links + tag"]
    N --> O
~~~

The hybrid retriever currently performs both dense and sparse work in one call.
Setting <code>alpha=0</code> changes score fusion but does not create a
sparse-only execution path.

### Public research direction

A possible successor would make retrieval sequential: search BM25 first, stop
when that evidence is sufficient, and acquire dense retrieval only when needed.
A separate evidence verifier could support answer-or-abstain behavior. Those
paths are future work; they are not implemented or validated by this repository.

## Project status

- The adaptive gate, hybrid retriever, passage scorer, generator wrapper,
  reusable pipeline API, experiment runner, and offline test surfaces are
  implemented.
- Real model, GPU, Java, and index integrations are selected separately from
  offline tests.
- Corpora, indexes, model weights, and run outputs are external artifacts and
  are not distributed with the source tree.
- This README reports implementation scope only. It makes no benchmark,
  latency, factuality, safety, deployment, or impact claim.

## Tech stack

| Layer | Technologies |
|---|---|
| Language | Python 3.10+ |
| Generator | Hugging Face causal LM; configured default is Mistral-7B-Instruct-v0.3 |
| NLI | RoBERTa-large checkpoint trained on SNLI, MNLI, FEVER, and ANLI |
| Dense retrieval | sentence-transformers + FAISS |
| Sparse retrieval | Pyserini bridge to Anserini <code>SimpleSearcher</code> / Lucene BM25 |
| Configuration | YAML |
| Quality tooling | pytest, Ruff, mypy |
| Packaging | setuptools + <code>pyproject.toml</code> |

## Requirements

- Python 3.10 or newer.
- JDK 21 for real sparse retrieval and Lucene index construction.
- A CUDA/bitsandbytes-compatible environment for the default real generator
  path. Install the <code>quantization</code> extra for bitsandbytes and
  Accelerate.
- Model checkpoints and datasets are downloaded from their configured upstream
  sources unless already cached.

## Quickstart

Start from a local checkout:

~~~bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -e ".[dev]"

# Deterministic local demo; no model weights are loaded
python scripts/demo.py

# Offline suite
pytest tests/ -v -m "not integration"
~~~

For the default 4-bit real generator path:

~~~bash
pip install -e ".[dev,quantization]"
~~~

The base install uses <code>faiss-cpu</code>. The quantization extra adds the
generator runtime dependencies; it does not install a second FAISS distribution.
The commands above are usage instructions, not a passing-test or performance
claim.

### Build synthetic development artifacts

~~~bash
python -m factuality_rag.cli chunk_wiki \
    --output data/wiki_chunks.jsonl \
    --chunk-size 200 \
    --chunk-overlap 50 \
    --dev-sample-size 50 \
    --mock-mode

python -m factuality_rag.cli build_index \
    --corpus data/wiki_chunks.jsonl \
    --faiss-out indexes/faiss.index \
    --pyserini-out indexes/pyserini_dir \
    --dev-sample-size 50 \
    --mock-mode
~~~

Real BM25 use requires a Java 21 installation. Set <code>JAVA_HOME</code>, or
set <code>FACTUALITY_RAG_JAVA_HOME</code> to select a Java 21 runtime for the
isolated Pyserini worker.

### Answer a query

~~~bash
python -m factuality_rag.cli run \
    --query "What is the capital of France?" \
    --k 10 \
    --mock-mode
~~~

### Evaluate predictions

~~~bash
python -m factuality_rag.cli evaluate \
    --predictions runs/&lt;run-id&gt;/predictions.jsonl \
    --support-metric lexical
~~~

The evaluator defaults to <code>--support-metric none</code>. The optional
lexical support score is explicitly labelled and is not FactScore.

### Python API

~~~python
from factuality_rag.pipeline.orchestrator import Pipeline

pipe = Pipeline(mock_mode=True)
answer, passages, provenance, confidence = pipe.run(
    "What is the capital of France?"
)
~~~

The <code>Pipeline</code> object reuses component wrappers across queries.
Heavy models and indexes remain lazy-loaded and are cached when first needed.

## Experiment runs and resume support

<code>python -m factuality_rag.experiment_runner</code> writes each run under
<code>runs/&lt;run-id&gt;/</code>. Before processing the first query it durably
creates:

- <code>resume_manifest.json</code>, which binds the immutable run identity and
  inputs;
- <code>predictions.jsonl</code>, an append-only per-query checkpoint.

Each completed record is validated, flushed, and fsynced. Repeating the original
invocation with <code>--resume runs/&lt;run-id&gt;</code> validates the manifest
and completed prefix before processing only the unfinished suffix. A final
non-newline JSON fragment can be discarded as an uncommitted torn write;
malformed completed records fail closed.

On completion the runner also writes <code>metrics.json</code>,
<code>metadata.json</code>, and, when references are available,
<code>references_by_example_id.json</code> plus the unambiguous-query
compatibility map <code>references.json</code>.

Mock mode replaces model and retrieval work; the experiment runner may still
load its configured dataset unless that dataset is already cached.

## Configuration

Current pipeline settings live in YAML under <code>configs/</code>. Omitting
<code>config_path</code> from the Python API uses the package-distributed sample
config; an explicit path is interpreted literally and fails if missing or
malformed.

| Parameter | Effective default | Meaning |
|---|---:|---|
| <code>retriever.alpha</code> | <code>0.6</code> | Dense contribution to fused retrieval score |
| <code>retriever.top_k</code> | <code>10</code> | Requested result count |
| <code>gating.entropy_thresh</code> | <code>1.2</code> | Retrieve above this entropy |
| <code>gating.logit_gap_thresh</code> | <code>2.0</code> | Retrieve below this top-two gap |
| <code>gating.probe_tokens</code> | <code>1</code> | Autoregressive positions averaged by the probe |
| <code>scorer.score_threshold</code> | <code>0.4</code> | Minimum fused passage score |
| <code>scorer.weights</code> | <code>0.5 / 0.2 / 0.3</code> | NLI / overlap / retrieval weights |
| <code>scorer.nli_mode</code> | <code>"passage"</code> | Full-passage or heuristic sentence-unit NLI |
| <code>scorer.nli_batch_size</code> | <code>8</code> | Transformers batch size for full-passage NLI |
| <code>scorer.cross_encoder_model</code> | <code>null</code> | Optional reranker checkpoint |

The current orchestrator constructs the generator with the wrapper defaults:
<code>max_new_tokens=256</code>, <code>temperature=0.1</code>, and
<code>do_sample=False</code>. These decoding values are currently code defaults
rather than YAML settings.

## Selected repository structure

~~~text
factuality_rag/
├── data/                    # Dataset loading, splits, and Wikipedia chunking
├── index/builder.py         # FAISS and Lucene index construction
├── retriever/
│   ├── hybrid.py            # Dense+sparse retrieval and score fusion
│   └── pyserini_worker.py   # Isolated Anserini SimpleSearcher boundary
├── gating/probe.py          # Entropy/logit-gap gate
├── scorer/
│   ├── passage.py           # Passage NLI, overlap, and retrieval fusion
│   └── learned_scorer.py    # Authenticated optional scorer loading
├── generator/wrapper.py     # Prompt formatting and causal-LM generation
├── pipeline/orchestrator.py # Functional and reusable pipeline APIs
├── eval/                    # Answer and retrieval metrics
├── experiment_runner.py     # Durable checkpoints, resume, and run outputs
├── model_registry.py        # Shared model/tokenizer cache
└── reproducibility.py       # Artifact identities and manifest validation

configs/                     # Runtime and experiment YAML configurations
scripts/                     # Corpus, analysis, statistics, and demo commands
docs/                        # Architecture, API, and JarvisLabs documentation
tests/                       # Offline tests and selected integrations
~~~

## Excluded assets

The repository does not commit model weights, generated indexes, corpus
snapshots, or run outputs. See <code>.gitignore</code> for the complete
local-artifact boundary. Upstream datasets, checkpoints, and downloaded
artifacts retain their own licenses and terms.

## Documentation

| Document | Description |
|---|---|
| [<code>docs/ARCHITECTURE.md</code>](docs/ARCHITECTURE.md) | Implemented pipeline, artifact flow, and public future-work boundary |
| [<code>docs/API_REFERENCE.md</code>](docs/API_REFERENCE.md) | Selected APIs and command-line surfaces |
| [<code>docs/JARVISLABS.md</code>](docs/JARVISLABS.md) | Linux GPU environment and runbook |

## Roadmap

- Add a genuine sparse-first retrieval path that can stop before dense work.
- Add structured citations and verifier-assisted abstention.
- Evaluate compressed or sharded indexes and batched query execution.
- Publish benchmark claims only with independently reproducible artifacts.

## Author

**Yash Verma** — PES University (PES1UG23AM910)

## License

The repository's original code is released under the [MIT License](LICENSE).
Datasets, model weights, and downloaded artifacts are governed by their
respective upstream licenses and terms.
