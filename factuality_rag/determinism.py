"""Small dependency-free helpers for cross-process deterministic simulation."""

from __future__ import annotations

import hashlib
import json
from typing import Union

SeedPart = Union[str, int]


def stable_seed(namespace: str, *parts: SeedPart) -> int:
    """Return a stable uint32 seed for a namespaced sequence of values.

    Python's built-in :func:`hash` is intentionally salted per process, so it
    must not be used for reproducible mock experiments.  Canonical JSON keeps
    boundaries and value types unambiguous before hashing.
    """

    if not isinstance(namespace, str) or not namespace or namespace != namespace.strip():
        raise ValueError("namespace must be a non-empty trimmed string")
    if any(isinstance(part, bool) or not isinstance(part, (str, int)) for part in parts):
        raise TypeError("seed parts must be strings or integers")

    payload = json.dumps(
        {"namespace": namespace, "parts": list(parts)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)
