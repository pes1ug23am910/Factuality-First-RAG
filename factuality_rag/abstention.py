"""Shared, exact policy for the generator's canonical abstention response."""

from __future__ import annotations


CANONICAL_ABSTENTION = "I cannot answer based on the provided context."


def is_canonical_abstention(answer: str) -> bool:
    """Return whether *answer* is the canonical abstention.

    Matching ignores case and differences in surrounding or internal
    whitespace. It deliberately does not use substring or punctuation-
    stripping heuristics, so an answer that merely contains the abstention
    sentence is still treated as an answered response.
    """

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    normalized = " ".join(answer.split()).casefold()
    canonical = " ".join(CANONICAL_ABSTENTION.split()).casefold()
    return normalized == canonical


__all__ = ["CANONICAL_ABSTENTION", "is_canonical_abstention"]
