#!/usr/bin/env python
"""Fail-closed entry point for learned-scorer training.

The previous script generated overlap and retrieval features from the target
label, randomly split those synthetic rows, and described the resulting score
as held-out or cross-dataset evaluation.  Its ``--eval-dataset`` option did not
load that dataset.  Such artifacts cannot establish ranking quality or
generalisation and must not be used by the B5 learned-scorer route.

Training remains disabled until this entry point consumes a versioned feature
artifact built from a frozen development split, independently labeled passage
relevance, immutable model/index revisions, and disjoint evaluation data.  The
generic :class:`factuality_rag.scorer.learned_scorer.LearnedScorer` API remains
available to code that already has defensible feature/label arrays; loading its
pickle artifact is separately hash-bound and explicitly opt-in.
"""

from __future__ import annotations


def main() -> None:
    """Refuse to create a misleading learned-scorer artifact."""
    raise RuntimeError(
        "learned-scorer training is disabled until a frozen split, independently "
        "labeled feature artifact, immutable component revisions, and disjoint "
        "evaluation set are implemented"
    )


if __name__ == "__main__":
    main()
