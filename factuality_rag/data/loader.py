"""
factuality_rag.data.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified dataset loading wrapper around HuggingFace ``datasets``.

Supported datasets:
    natural_questions, hotpot_qa, fever, trivia_qa, EleutherAI/truthful_qa_mc

Example::

    >>> from factuality_rag.data import load_dataset
    >>> ds = load_dataset("natural_questions", split="validation", dev_sample_size=100)
"""

from __future__ import annotations

import logging
from typing import Optional

import datasets as hf_datasets

logger = logging.getLogger(__name__)

# ── Known dataset configs ─────────────────────────────────────
_DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "natural_questions": {"path": "natural_questions", "name": "default"},
    "hotpot_qa": {"path": "hotpot_qa", "name": "fullwiki"},
    "fever": {"path": "fever", "name": "v1.0"},
    "trivia_qa": {"path": "trivia_qa", "name": "rc"},
    "truthful_qa": {"path": "EleutherAI/truthful_qa_mc", "name": None},
    "EleutherAI/truthful_qa_mc": {"path": "EleutherAI/truthful_qa_mc", "name": None},
}


def load_dataset(
    name: str,
    split: str = "train",
    dev_sample_size: Optional[int] = None,
    *,
    streaming: bool = False,
    trust_remote_code: bool = True,
) -> hf_datasets.Dataset:
    """Load a HuggingFace dataset with optional dev-sampling.

    Args:
        name: Dataset identifier – one of the keys in ``_DATASET_CONFIGS``
              or any HuggingFace dataset path.
        split: Dataset split (e.g. ``"train"``, ``"validation"``).
        dev_sample_size: If set, randomly sample this many rows (deterministic,
                         seed=42) for fast dev iteration.
        streaming: Whether to use streaming mode.
        trust_remote_code: Passed to ``datasets.load_dataset``.

    Returns:
        A ``datasets.Dataset`` (or ``IterableDataset`` when streaming).

    Example::

        >>> ds = load_dataset("hotpot_qa", split="validation", dev_sample_size=50)
        >>> len(ds) <= 50
        True
    """
    cfg = _DATASET_CONFIGS.get(name, {"path": name, "name": None})
    logger.info("Loading dataset '%s' (split=%s) ...", name, split)

    kwargs: dict = {
        "path": cfg["path"],
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }
    if cfg.get("name"):
        kwargs["name"] = cfg["name"]

    ds = hf_datasets.load_dataset(**kwargs)

    if dev_sample_size is not None and not streaming:
        ds = ds.shuffle(seed=42).select(range(min(dev_sample_size, len(ds))))
        logger.info("Dev-sampled to %d rows.", len(ds))

    return ds
