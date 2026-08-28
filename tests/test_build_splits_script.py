"""Strict-input tests for the split-manifest CLI wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_splits import _load_jsonl


def test_jsonl_loader_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(
        '{"example_id":"a","example_id":"b","family_ids":["example:a"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _load_jsonl(source)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_jsonl_loader_rejects_nonfinite_numbers(tmp_path: Path, value: str) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(
        '{"example_id":"a","family_ids":["example:a"],"weight":' + value + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite JSON number"):
        _load_jsonl(source)


def test_jsonl_loader_accepts_blank_lines_without_losing_records(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(
        '\n{"example_id":"a","family_ids":["example:a"]}\n\n',
        encoding="utf-8",
    )

    assert _load_jsonl(source) == [{"example_id": "a", "family_ids": ["example:a"]}]
