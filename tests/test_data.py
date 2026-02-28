"""
tests.test_data
~~~~~~~~~~~~~~~~
Unit tests for data loading and wiki chunker.
"""

from __future__ import annotations

import pytest

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
            assert set(ch.keys()) == {"id", "title", "text", "tokens", "source"}
            assert ch["source"] == "enwiki"
            assert ch["tokens"] <= 10

    def test_dedupe(self) -> None:
        c = WikiChunker(chunk_size=100, chunk_overlap=0)
        chunks1 = list(c.chunk_text("A", "hello world"))
        chunks2 = list(c.chunk_text("A", "hello world"))
        assert len(chunks2) == 0  # duplicate

    def test_mock_articles(self) -> None:
        c = WikiChunker()
        arts = c.generate_mock_articles(5)
        assert len(arts) == 5

    def test_process_articles_dry_run(self) -> None:
        c = WikiChunker(chunk_size=50, chunk_overlap=10, dry_run=True)
        arts = [{"title": "A", "text": "word " * 120}]
        result = c.process_articles(arts)
        assert len(result) > 0
