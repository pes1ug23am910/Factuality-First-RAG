"""
tests.test_data
~~~~~~~~~~~~~~~~
Unit tests for data loading and wiki chunker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from factuality_rag.data import loader
from factuality_rag.data.wikipedia import WikiChunker


class TestWikiChunker:
    def test_chunk_text_produces_chunks(self) -> None:
        c = WikiChunker(chunk_size=5, chunk_overlap=2)
        chunks = list(c.chunk_text("Title", "a b c d e f g h i j"))
        assert len(chunks) > 0

    def test_chunk_schema(self) -> None:
        c = WikiChunker(chunk_size=10, chunk_overlap=3)
        chunks = list(c.chunk_text("T", "word " * 30))
        for ch in chunks:
            assert set(ch.keys()) == {
                "id",
                "title",
                "text",
                "tokens",
                "source",
                "mock_mode",
            }
            assert ch["source"] == "enwiki"
            assert ch["mock_mode"] is False
            assert ch["tokens"] <= 10

    def test_dedupe(self) -> None:
        c = WikiChunker(chunk_size=100, chunk_overlap=0)
        _ = list(c.chunk_text("A", "hello world"))
        chunks2 = list(c.chunk_text("A", "hello world"))
        assert len(chunks2) == 0  # duplicate

    def test_mock_articles(self) -> None:
        c = WikiChunker(mock_mode=True)
        arts = c.generate_mock_articles(5)
        assert len(arts) == 5
        chunks = list(c.chunk_text(arts[0]["title"], arts[0]["text"]))
        assert chunks[0]["source"] == "synthetic-wikipedia-mock"
        assert chunks[0]["mock_mode"] is True

    def test_process_articles_dry_run(self) -> None:
        c = WikiChunker(chunk_size=50, chunk_overlap=10, dry_run=True)
        arts = [{"title": "A", "text": "word " * 120}]
        result = c.process_articles(arts)
        assert len(result) > 0


def test_loader_uses_requested_development_sample_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.shuffle_seed = None

        def __len__(self) -> int:
            return 5

        def shuffle(self, seed: int) -> "FakeDataset":
            self.shuffle_seed = seed
            return self

        def select(self, indexes: range) -> "FakeDataset":
            assert list(indexes) == [0, 1]
            return self

    dataset = FakeDataset()
    monkeypatch.setattr(loader.hf_datasets, "load_dataset", lambda **kwargs: dataset)

    loaded = loader.load_dataset("custom", dev_sample_size=2, seed=19)

    assert loaded is dataset
    assert dataset.shuffle_seed == 19


@pytest.mark.parametrize(
    "name",
    [
        "fever",
        "truthful_qa",
        "EleutherAI/truthful_qa_mc",
        "popqa",
        "akariasai/PopQA",
        "hagrid",
        "miracl/hagrid",
    ],
)
def test_task_specific_datasets_fail_before_huggingface_access(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(**kwargs: object) -> object:
        raise AssertionError("disabled task reached HuggingFace")

    monkeypatch.setattr(loader.hf_datasets, "load_dataset", forbidden)

    with pytest.raises(NotImplementedError, match="task-specific"):
        loader.load_dataset(name)


def _chunk_args(
    tmp_path: Path, *, mock_mode: bool, input_path: object = None
) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        output=str(tmp_path),
        chunk_size=20,
        chunk_overlap=5,
        dev_sample_size=2,
        dry_run=False,
        mock_mode=mock_mode,
    )


def test_chunk_cli_requires_explicit_mock_mode(tmp_path: Path) -> None:
    from factuality_rag.cli.__main__ import _cmd_chunk_wiki

    with pytest.raises(RuntimeError, match="no real dump parser"):
        _cmd_chunk_wiki(_chunk_args(tmp_path, mock_mode=False))


def test_chunk_cli_rejects_ignored_input_in_mock_mode(tmp_path: Path) -> None:
    from factuality_rag.cli.__main__ import _cmd_chunk_wiki

    with pytest.raises(ValueError, match="cannot be combined"):
        _cmd_chunk_wiki(_chunk_args(tmp_path, mock_mode=True, input_path="dump.xml"))
