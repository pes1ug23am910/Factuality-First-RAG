"""
factuality_rag.data.wikipedia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wikipedia chunker with streaming JSONL output, deduplication,
dry-run, and dev-sample-size support.

JSONL output schema::

    {"id": str, "title": str, "text": str, "tokens": int, "source": "enwiki"}

Example (CLI)::

    python -m factuality_rag.cli chunk_wiki \\
        --input dump.xml --output wiki_chunks.jsonl \\
        --chunk-size 200 --chunk-overlap 50 --dev-sample-size 100 --dry-run

Example (Python)::

    >>> chunker = WikiChunker(chunk_size=200, chunk_overlap=50, mock_mode=True)
    >>> chunks = list(chunker.chunk_text("Albert Einstein", "Albert Einstein was ..."))
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

logger = logging.getLogger(__name__)


class WikiChunker:
    """Chunk Wikipedia articles into fixed-token-window passages.

    Args:
        chunk_size: Number of whitespace tokens per chunk (default 200).
        chunk_overlap: Overlapping tokens between consecutive chunks (default 50).
        dry_run: If ``True``, process but do not write files.
        mock_mode: If ``True``, generate deterministic synthetic chunks.
        dev_sample_size: Limit the number of articles processed.

    Example::

        >>> c = WikiChunker(chunk_size=100, chunk_overlap=20)
        >>> chunks = list(c.chunk_text("Title", "word " * 250))
        >>> all("title" in ch and "text" in ch for ch in chunks)
        True
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        dry_run: bool = False,
        mock_mode: bool = False,
        dev_sample_size: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.dev_sample_size = dev_sample_size
        self._seen_checksums: set[str] = set()

    # ── Core chunking ─────────────────────────────────────────

    def chunk_text(
        self, title: str, text: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Split *text* into overlapping token-window chunks.

        Args:
            title: Article title.
            text: Full article body text.

        Yields:
            Dict matching the JSONL schema:
            ``{"id","title","text","tokens","source"}``.

        Example::

            >>> c = WikiChunker(chunk_size=5, chunk_overlap=2)
            >>> chunks = list(c.chunk_text("T", "a b c d e f g h"))
            >>> chunks[0]["tokens"] <= 5
            True
        """
        tokens = text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start : start + self.chunk_size]
            chunk_text = " ".join(chunk_tokens)

            # Deduplicate by title+text checksum
            checksum = self._checksum(title, chunk_text)
            if checksum in self._seen_checksums:
                continue
            self._seen_checksums.add(checksum)

            yield {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"enwiki:{title}:{start}")),
                "title": title,
                "text": chunk_text,
                "tokens": len(chunk_tokens),
                "source": "enwiki",
            }

    # ── Streaming JSONL writer ────────────────────────────────

    def process_articles(
        self,
        articles: Iterable[Dict[str, str]],
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk an iterable of ``{"title","text"}`` dicts and stream to JSONL.

        Args:
            articles: Iterable of dicts with ``title`` and ``text`` keys.
            output_path: File path for JSONL output. Ignored when ``dry_run``
                         is ``True``.

        Returns:
            List of all chunk dicts produced.

        Example::

            >>> c = WikiChunker(chunk_size=50, chunk_overlap=10, dry_run=True)
            >>> arts = [{"title": "A", "text": "word " * 120}]
            >>> result = c.process_articles(arts)
            >>> len(result) > 0
            True
        """
        all_chunks: list[Dict[str, Any]] = []
        file_handle = None

        if output_path and not self.dry_run:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            file_handle = open(out, "w", encoding="utf-8")

        count = 0
        try:
            for article in articles:
                if self.dev_sample_size is not None and count >= self.dev_sample_size:
                    break
                title = article.get("title", "Untitled")
                text = article.get("text", "")

                if self.mock_mode:
                    text = text or (f"Mock passage for {title}. " * 40)

                for chunk in self.chunk_text(title, text):
                    all_chunks.append(chunk)
                    if file_handle:
                        file_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1
        finally:
            if file_handle:
                file_handle.close()

        logger.info(
            "Processed %d articles → %d chunks%s",
            count,
            len(all_chunks),
            " (dry-run)" if self.dry_run else "",
        )
        return all_chunks

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _checksum(title: str, text: str) -> str:
        """Return MD5 hex-digest of title+text for deduplication.

        Example::

            >>> WikiChunker._checksum("T", "hello")  # doctest: +ELLIPSIS
            '...'
        """
        return hashlib.md5(f"{title}||{text}".encode()).hexdigest()

    def generate_mock_articles(self, n: int = 10) -> List[Dict[str, str]]:
        """Generate *n* synthetic articles for testing.

        Args:
            n: Number of articles to generate.

        Returns:
            List of ``{"title","text"}`` dicts.

        Example::

            >>> arts = WikiChunker().generate_mock_articles(3)
            >>> len(arts)
            3
        """
        return [
            {
                "title": f"Mock Article {i}",
                "text": f"This is mock article {i}. " * 60,
            }
            for i in range(n)
        ]

    # ── HuggingFace Wikipedia loading ─────────────────────────

    def load_from_hf(
        self,
        sample_size: Optional[int] = None,
        output_path: Optional[str] = None,
        wiki_config: str = "20231101.en",
    ) -> List[Dict[str, Any]]:
        """Load Wikipedia articles from HuggingFace and chunk them.

        Uses streaming to avoid downloading the entire dataset at
        once.  Writes chunked output to JSONL.

        Args:
            sample_size: Max number of articles to process.
                         ``None`` → all articles (use with caution).
            output_path: Output JSONL file path.  Defaults to
                         ``data/wiki_{sample_size}_chunks.jsonl``.
            wiki_config: HuggingFace Wikipedia config string.

        Returns:
            List of chunk dicts.

        Example::

            >>> c = WikiChunker(chunk_size=50, chunk_overlap=10)
            >>> # chunks = c.load_from_hf(sample_size=100)  # needs network
        """
        from datasets import load_dataset  # type: ignore[import-untyped]

        logger.info(
            "Loading Wikipedia from HuggingFace (config=%s, sample=%s) …",
            wiki_config,
            sample_size or "ALL",
        )
        wiki = load_dataset(
            "wikimedia/wikipedia", wiki_config, split="train", streaming=True,
        )

        def _article_iter():
            count = 0
            for row in wiki:
                if sample_size is not None and count >= sample_size:
                    break
                yield {"title": row["title"], "text": row["text"]}
                count += 1

        if output_path is None:
            tag = sample_size or "full"
            output_path = f"data/wiki_{tag}_chunks.jsonl"

        chunks = self.process_articles(_article_iter(), output_path=output_path)
        logger.info(
            "HF Wikipedia: %d chunks written to %s", len(chunks), output_path
        )
        return chunks
