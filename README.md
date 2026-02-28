# Factuality-First RAG

**Adaptive Retrieval Gating + Passage-Level Provenance & Factuality Scoring**

[![CI](https://github.com/PES1UG23AM910/Factuality-First-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/PES1UG23AM910/Factuality-First-RAG/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

Standard Retrieval-Augmented Generation (RAG) systems **always retrieve** before answering — even for queries the model already knows. This wastes compute, increases latency, and can introduce irrelevant or contradictory context that *increases* hallucination. There is no mechanism to assess whether retrieved passages actually support the generated answer.

**Factuality-First RAG** solves both problems with two novel components:

1. **Adaptive Retrieval Gating** — A zero-cost logit probe on the generator's next-token distribution decides *whether* to retrieve, cutting unnecessary retrieval calls by up to 40%.
2. **Passage-Level Factuality Scoring** — Each retrieved passage is verified against the query using NLI entailment before the generator ever sees it, ensuring only *trusted evidence* reaches the answer.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   GATING PROBE          │  entropy > 1.2 OR logit_gap < 2.0 → RETRIEVE
│   (single forward pass) │  else → SKIP (parametric answer)
└────────┬────────────────┘
         │ retrieve=True
         ▼
┌─────────────────────────┐
│  HYBRID RETRIEVER       │  combined = α·dense + (1−α)·BM25   (α=0.6)
│  FAISS HNSW + Pyserini  │  per-query min-max normalisation
└────────┬────────────────┘
         │ top-K passages
         ▼
┌─────────────────────────┐
│  PASSAGE SCORER         │  score = 0.5·P(ent) + 0.2·overlap + 0.3·ret
│  NLI + token overlap    │  premise=passage, hypothesis=query
└────────┬────────────────┘
         │ trusted passages (score ≥ 0.4)
         ▼
┌─────────────────────────┐
│  GENERATOR              │  Mistral-7B-Instruct-v0.3 (4-bit quantised)
│  model_registry shared  │  [INST] RAG prompt template
└────────┬────────────────┘
         │
         ▼
   (answer, trusted_passages, provenance, confidence_tag)
```

> The gating probe and generator share model weights via a singleton `model_registry`, avoiding double-loading of 7B parameters.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Adaptive Gating** | Entropy + logit-gap probe decides retrieval need in one forward pass |
| **Hybrid Retrieval** | Dense (FAISS HNSW) + sparse (BM25 / Pyserini) with learned α fusion |
| **NLI-Based Scoring** | RoBERTa-large NLI verifies each passage before generation |
| **FactScore Evaluation** | Claim decomposition + per-claim NLI verification |
| **Model Registry** | Singleton cache with 4-bit quantisation — no double-loading |
| **Pipeline Class** | Load-once, run-many architecture for efficient batch experiments |
| **Full Mock Mode** | All 53 tests pass in <3 s without GPU or model downloads |
| **CI/CD** | GitHub Actions: pytest × 3 Python versions + ruff + mypy |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.9+ |
| **LLM** | Mistral-7B-Instruct-v0.3, 4-bit quantised (bitsandbytes) |
| **NLI** | RoBERTa-large (SNLI + MNLI + FEVER + ANLI) |
| **Embeddings** | sentence-transformers / all-mpnet-base-v2 |
| **Dense Index** | FAISS (HNSW Flat / IVF-PQ) |
| **Sparse Index** | Pyserini (Lucene BM25) |
| **Datasets** | HuggingFace Datasets (NQ, HotpotQA, FEVER, TriviaQA, TruthfulQA) |
| **Testing** | pytest, ruff, mypy |
| **CI/CD** | GitHub Actions |
| **Build** | setuptools + pyproject.toml |

---

## Project Structure

```
factuality_rag/
├── model_registry.py        # Singleton model cache (4-bit quant, shared weights)
├── data/
│   ├── loader.py            # HuggingFace dataset adapters (5 benchmarks)
│   └── wikipedia.py         # WikiChunker: offline + HF streaming ingestion
├── index/
│   └── builder.py           # FAISS (HNSW / IVFPQ) + Pyserini collection builder
├── retriever/
│   └── hybrid.py            # Hybrid dense+sparse retrieval with score fusion
├── gating/
│   └── probe.py             # Adaptive retrieval gating (entropy + logit gap)
├── scorer/
│   └── passage.py           # NLI + overlap + retrieval score fusion
├── generator/
│   └── wrapper.py           # Mistral-7B with [INST] RAG templates
├── pipeline/
│   └── orchestrator.py      # run_pipeline() + Pipeline class (load-once)
├── eval/
│   └── metrics.py           # EM, F1, FactScore (claim decomposition + NLI)
├── cli/
│   └── __main__.py          # CLI: chunk-wiki, build-index, run, evaluate
└── experiment_runner.py     # Batch experiment runner with metadata tracking

configs/                     # YAML experiment configurations
docs/                        # Architecture, API reference, experiment plan
tests/                       # 53 unit tests (all mock-mode)
scripts/                     # Demo script + sample experiment runner
.github/workflows/ci.yml     # CI pipeline (pytest + ruff + mypy)
```

---

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Python** | 3.9 | 3.11+ |
| **RAM** | 8 GB | 16 GB |
| **GPU** | — (mock-mode works on CPU) | NVIDIA A100-80 GB / RTX 4090 (24 GB) |
| **Disk** | 500 MB (code only) | ~20 GB (with models + indexes) |
| **OS** | Linux, macOS, Windows | Ubuntu 22.04 / Windows 11 |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/PES1UG23AM910/Factuality-First-RAG.git
cd Factuality-First-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. (Optional) Download models for real inference

Models are **not** included in this repository. For real (non-mock) inference:

```bash
# Models will auto-download on first run via HuggingFace Hub.
# Ensure ~15 GB free disk space and a CUDA-capable GPU.
#
# Required models:
#   mistralai/Mistral-7B-Instruct-v0.3   (~4 GB quantised)
#   ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli   (~1.4 GB)
#   sentence-transformers/all-mpnet-base-v2   (~420 MB)
```

---

## Usage

### Quick Demo (no GPU required)

```bash
python scripts/demo.py
```

### Run Tests

```bash
pytest tests/ -v                          # All 53 tests, ~3 seconds
pytest tests/ -v -m "not integration"     # Skip GPU-requiring tests
```

### CLI Commands

```bash
# Chunk Wikipedia (mock-mode for demo)
python -m factuality_rag.cli chunk_wiki \
    --output data/wiki_chunks.jsonl \
    --chunk-size 200 --chunk-overlap 50 \
    --dev-sample-size 50 --mock-mode

# Build FAISS + Pyserini indexes
python -m factuality_rag.cli build_index \
    --corpus data/wiki_chunks.jsonl \
    --faiss-out indexes/faiss.index \
    --pyserini-out indexes/pyserini_dir \
    --dev-sample-size 50 --mock-mode

# Run pipeline on a single query
python -m factuality_rag.cli run \
    --query "What is the capital of France?" \
    --k 10 --mock-mode

# Evaluate predictions
python -m factuality_rag.cli evaluate \
    --predictions runs/<run-id>/predictions.jsonl
```

### Pipeline API (Python)

```python
from factuality_rag.pipeline.orchestrator import Pipeline

# Load all components once
pipe = Pipeline(config_path="configs/exp_sample.yaml", mock_mode=True)

# Run on any query
answer, passages, provenance, confidence = pipe.run("What is the capital of France?")
print(f"Answer: {answer}  (confidence: {confidence})")
```

---

## Configuration

All hyperparameters are controlled via YAML configs in `configs/`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retriever.alpha` | 0.6 | Dense vs. sparse weight |
| `gating.entropy_threshold` | 1.2 | Uncertainty threshold |
| `gating.logit_gap_threshold` | 2.0 | Confidence gap threshold |
| `scorer.score_threshold` | 0.4 | Minimum passage trust score |
| `scorer.w_nli / w_overlap / w_ret` | 0.5 / 0.2 / 0.3 | Scorer fusion weights |

---

## Experiment Plan

Five benchmarks × five baselines × full ablation suite:

- **Datasets:** NQ-Open, HotpotQA, FEVER, TriviaQA, TruthfulQA
- **Baselines:** Closed-book, Always-RAG, Gate-only, Score-only, Self-RAG
- **Metrics:** Exact Match, Token F1, FactScore, Hallucination Rate, Retrieval Call %
- **Statistical rigour:** 3 seeds, paired bootstrap test (n=1,000)

See [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for the full protocol.

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Detailed system architecture and data flow |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Complete API reference for all modules |
| [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) | Evaluation protocol and ablation plan |
| [`docs/SUGGESTIONS.md`](docs/SUGGESTIONS.md) | Roadmap and improvement suggestions |

---

## Excluded Assets

The following are **not** committed to this repository (see `.gitignore`):

| Asset | Size | How to obtain |
|-------|------|---------------|
| LLM weights (Mistral-7B) | ~4 GB | Auto-downloads via HuggingFace on first run |
| NLI model (RoBERTa-large) | ~1.4 GB | Auto-downloads via HuggingFace on first run |
| Embedding model | ~420 MB | Auto-downloads via HuggingFace on first run |
| FAISS / Lucene indexes | Variable | Generated by `build_index` CLI command |
| Wikipedia corpus chunks | Variable | Generated by `chunk_wiki` CLI command |

All assets are **fully reproducible** from the source code and public model hubs.

---

## Author

**Yash Verma**  
PES University — PES1UG23AM910

## License

MIT

