"""
factuality_rag.pipeline.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end RAG pipeline: gating → retrieval → scoring → generation.

Provides both a stateless ``run_pipeline()`` convenience function
and a ``Pipeline`` class that loads components once and reuses them
across queries (fixing the re-instantiation performance bug).

Example (mock-mode)::

    >>> answer, passages, provenance, tag = run_pipeline(
    ...     "What is Python?", mock_mode=True)
    >>> tag in ("high", "medium", "low")
    True

Example (pipeline class)::

    >>> pipe = Pipeline(mock_mode=True)
    >>> ans, ps, prov, tag = pipe.run("What is DNA?")
    >>> tag in ("high", "medium", "low")
    True
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Default config path (relative to repo root)
_DEFAULT_CONFIG = "configs/exp_sample.yaml"


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(config_path)
    if not p.exists():
        logger.warning("Config not found at %s – using defaults.", config_path)
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_git_commit() -> str:
    """Return current git commit hash or a fallback string.

    Returns:
        Short git commit hash or ``"git-not-available"``.
    """
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "git-not-available"


def run_pipeline(
    query: str,
    k: int = 10,
    gate: bool = True,
    score_threshold: float = 0.4,
    config_path: str = _DEFAULT_CONFIG,
    seed: int = 42,
    mock_mode: bool = False,
    *,
    probe: Optional[Any] = None,
    retriever: Optional[Any] = None,
    scorer: Optional[Any] = None,
    generator: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
    """Run the full Factuality-first RAG pipeline.

    Steps:
        1. (Optional) Gating probe decides if retrieval is needed.
        2. Hybrid retrieval (dense + BM25).
        3. Passage scoring (NLI + overlap + retriever fusion).
        4. Filter passages by *score_threshold*.
        5. Generate answer from trusted passages.
        6. Assign confidence tag.

    Pre-built components can be passed via keyword arguments to
    avoid re-instantiation on every call (see :class:`Pipeline`).

    Args:
        query: User question.
        k: Number of passages to retrieve.
        gate: Whether to apply the gating probe.
        score_threshold: Minimum ``final_score`` to keep a passage.
        config_path: Path to the experiment YAML config.
        seed: Random seed for reproducibility.
        mock_mode: If ``True``, all components run in mock-mode
                   (no model downloads, deterministic outputs).
        probe: Optional pre-built :class:`~factuality_rag.gating.probe.GatingProbe`.
        retriever: Optional pre-built :class:`~factuality_rag.retriever.hybrid.HybridRetriever`.
        scorer: Optional pre-built :class:`~factuality_rag.scorer.passage.PassageScorer`.
        generator: Optional pre-built :class:`~factuality_rag.generator.wrapper.Generator`.

    Returns:
        Tuple of ``(answer, trusted_passages, provenance, confidence_tag)``:

        - **answer** – generated answer string.
        - **trusted_passages** – list of passage dicts with
          ``final_score`` ≥ *score_threshold*.
        - **provenance** – mapping ``{claim_idx: [passage_ids]}``.
        - **confidence_tag** – ``'high'`` | ``'medium'`` | ``'low'``.

    Example::

        >>> ans, ps, prov, tag = run_pipeline("test?", mock_mode=True)
        >>> isinstance(ans, str)
        True
    """
    import numpy as np

    # ── Seed everything ───────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    cfg = _load_config(config_path)

    # ── Component imports (lazy, inside function) ─────────────
    from factuality_rag.gating.probe import GatingProbe
    from factuality_rag.generator.wrapper import Generator
    from factuality_rag.retriever.hybrid import HybridRetriever
    from factuality_rag.scorer.passage import PassageScorer

    # ── 1. Gating ─────────────────────────────────────────────
    retrieval_needed = True
    gating_cfg = cfg.get("gating", {})
    if gate:
        _probe = probe or GatingProbe(
            generator_model_hf=cfg.get("models", {}).get(
                "generator", "mistralai/Mistral-7B-Instruct-v0.3"
            ),
            mock_mode=mock_mode,
            temp=gating_cfg.get("calibration_temp", 1.0),
        )
        retrieval_needed = _probe.should_retrieve(
            query,
            probe_tokens=gating_cfg.get("probe_tokens", 1),
            entropy_thresh=gating_cfg.get("entropy_thresh", 1.2),
            logit_gap_thresh=gating_cfg.get("logit_gap_thresh", 2.0),
        )
        logger.info("Gating decision: %s", "RETRIEVE" if retrieval_needed else "SKIP")

    # ── 2. Retrieval ──────────────────────────────────────────
    passages: List[Dict[str, Any]] = []
    if retrieval_needed:
        ret_cfg = cfg.get("retriever", {})
        if retriever is not None:
            _retriever = retriever
        elif mock_mode:
            _retriever = HybridRetriever.build_mock(
                n_docs=max(k * 2, 20),
                seed=seed,
                alpha=ret_cfg.get("alpha", 0.6),
            )
        else:
            idx_cfg = cfg.get("index", {})
            _retriever = HybridRetriever(
                faiss_index_path=idx_cfg.get("faiss_out", "indexes/faiss.index"),
                pyserini_index_path=idx_cfg.get("pyserini_out", "indexes/pyserini_dir"),
                embed_model=cfg.get("models", {}).get(
                    "dense_embedder", "sentence-transformers/all-mpnet-base-v2"
                ),
                alpha=ret_cfg.get("alpha", 0.6),
                normalize=ret_cfg.get("normalize", True),
            )
        passages = _retriever.retrieve(query, k=k, rerank=ret_cfg.get("rerank", True))

    # ── 3. Scoring ────────────────────────────────────────────
    if passages:
        scorer_cfg = cfg.get("scorer", {})
        weights = scorer_cfg.get("weights", {})
        _scorer = scorer or PassageScorer(
            nli_model_hf=cfg.get("models", {}).get(
                "nli_verifier",
                "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            ),
            overlap_metric=scorer_cfg.get("overlap_metric", "token"),
            mock_mode=mock_mode,
            w_nli=weights.get("w_nli", 0.5),
            w_overlap=weights.get("w_overlap", 0.2),
            w_ret=weights.get("w_ret", 0.3),
        )
        passages = _scorer.score_passages(query, passages)

    # ── 4. Filter ─────────────────────────────────────────────
    trusted = [p for p in passages if p.get("final_score", 0) >= score_threshold]

    # ── 5. Generate ───────────────────────────────────────────
    _gen = generator or Generator(
        model_name=cfg.get("models", {}).get(
            "generator", "mistralai/Mistral-7B-Instruct-v0.3"
        ),
        mock_mode=mock_mode,
    )
    context = "\n\n".join(p["text"] for p in trusted) if trusted else ""
    answer = _gen.generate(query, context=context)

    # ── 6. Provenance & confidence ────────────────────────────
    provenance: Dict[str, Any] = {
        str(i): [p["id"]] for i, p in enumerate(trusted)
    }

    confidence_tag = _compute_confidence(trusted, retrieval_needed, gate)

    logger.info(
        "Pipeline done: %d trusted passages, confidence=%s", len(trusted), confidence_tag
    )

    return answer, trusted, provenance, confidence_tag


def _compute_confidence(
    trusted: List[Dict[str, Any]],
    retrieval_needed: bool,
    gating_enabled: bool,
) -> str:
    """Determine a confidence tag from the pipeline output.

    Args:
        trusted: List of trusted passages.
        retrieval_needed: Whether retrieval was triggered.
        gating_enabled: Whether the gating probe was used.

    Returns:
        ``'high'``, ``'medium'``, or ``'low'``.

    Example::

        >>> _compute_confidence([], True, True)
        'low'
    """
    if not retrieval_needed and gating_enabled:
        # Gating skipped retrieval → model was confident, but we cannot
        # verify the answer without passages, so cap at "medium".
        return "medium"
    if not trusted:
        return "low"
    avg_score = sum(p.get("final_score", 0) for p in trusted) / len(trusted)
    if avg_score >= 0.7:
        return "high"
    elif avg_score >= 0.45:
        return "medium"
    return "low"


# ── Pipeline class (reusable, loads components once) ─────────


class Pipeline:
    """Reusable pipeline that loads all components once at init.

    Solves the performance problem of ``run_pipeline()`` re-creating
    every model on every call.

    Args:
        config_path: Path to the experiment YAML config.
        mock_mode: If ``True``, all components run in mock-mode.
        seed: Default random seed.

    Example::

        >>> pipe = Pipeline(mock_mode=True)
        >>> ans, ps, prov, tag = pipe.run("What is DNA?")
        >>> tag in ("high", "medium", "low")
        True
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        mock_mode: bool = False,
        seed: int = 42,
    ) -> None:
        from factuality_rag.gating.probe import GatingProbe
        from factuality_rag.generator.wrapper import Generator
        from factuality_rag.retriever.hybrid import HybridRetriever
        from factuality_rag.scorer.passage import PassageScorer

        self.cfg = _load_config(config_path)
        self.mock_mode = mock_mode
        self.seed = seed

        models_cfg = self.cfg.get("models", {})
        gating_cfg = self.cfg.get("gating", {})
        ret_cfg = self.cfg.get("retriever", {})
        scorer_cfg = self.cfg.get("scorer", {})
        weights = scorer_cfg.get("weights", {})
        idx_cfg = self.cfg.get("index", {})

        # ── Build components once ─────────────────────────────
        generator_id = models_cfg.get(
            "generator", "mistralai/Mistral-7B-Instruct-v0.3"
        )

        self.probe = GatingProbe(
            generator_model_hf=generator_id,
            mock_mode=mock_mode,
            temp=gating_cfg.get("calibration_temp", 1.0),
        )

        if mock_mode:
            self.retriever = HybridRetriever.build_mock(
                n_docs=max(ret_cfg.get("top_k", 10) * 2, 20),
                seed=seed,
                alpha=ret_cfg.get("alpha", 0.6),
            )
        else:
            self.retriever = HybridRetriever(
                faiss_index_path=idx_cfg.get("faiss_out", "indexes/faiss.index"),
                pyserini_index_path=idx_cfg.get(
                    "pyserini_out", "indexes/pyserini_dir"
                ),
                embed_model=models_cfg.get(
                    "dense_embedder",
                    "sentence-transformers/all-mpnet-base-v2",
                ),
                alpha=ret_cfg.get("alpha", 0.6),
                normalize=ret_cfg.get("normalize", True),
            )

        self.scorer = PassageScorer(
            nli_model_hf=models_cfg.get(
                "nli_verifier",
                "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            ),
            overlap_metric=scorer_cfg.get("overlap_metric", "token"),
            mock_mode=mock_mode,
            w_nli=weights.get("w_nli", 0.5),
            w_overlap=weights.get("w_overlap", 0.2),
            w_ret=weights.get("w_ret", 0.3),
        )

        self.generator = Generator(
            model_name=generator_id,
            mock_mode=mock_mode,
        )

        self.score_threshold = scorer_cfg.get("score_threshold", 0.4)
        self.k = ret_cfg.get("top_k", 10)
        self._gating_cfg = gating_cfg
        logger.info("Pipeline initialised (mock_mode=%s).", mock_mode)

    def run(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        gate: bool = True,
        score_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
        """Run the pipeline on a single query, reusing loaded components.

        Args:
            query: User question.
            k: Override for retrieval depth (default from config).
            gate: Whether to apply the gating probe.
            score_threshold: Override for minimum ``final_score``.
            seed: Override for random seed.

        Returns:
            ``(answer, trusted_passages, provenance, confidence_tag)``

        Example::

            >>> pipe = Pipeline(mock_mode=True)
            >>> ans, ps, prov, tag = pipe.run("test?")
            >>> isinstance(ans, str)
            True
        """
        return run_pipeline(
            query,
            k=k or self.k,
            gate=gate,
            score_threshold=score_threshold or self.score_threshold,
            config_path=_DEFAULT_CONFIG,
            seed=seed or self.seed,
            mock_mode=self.mock_mode,
            probe=self.probe,
            retriever=self.retriever,
            scorer=self.scorer,
            generator=self.generator,
        )
