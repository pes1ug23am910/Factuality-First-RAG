"""
factuality_rag.generator.wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generator wrapper with mock-mode for answer generation.

In real mode the generator uses a 4-bit quantised causal LM
(e.g. Mistral-7B-Instruct) via the model registry so the same
weights are shared with the gating probe.

In mock-mode, returns a deterministic pseudo-answer without
loading any model.

Example::

    >>> gen = Generator("mistral-7b-instruct", mock_mode=True)
    >>> ans = gen.generate("What is Python?", context="A programming language.")
    >>> isinstance(ans, str) and len(ans) > 0
    True
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ── Default RAG prompt template (Mistral [INST] format) ──────
_RAG_PROMPT_TEMPLATE = (
    "<s>[INST] Answer the question using ONLY the provided context. "
    "If the context does not support an answer, say "
    '"I cannot answer based on the provided context."\n\n'
    "Context:\n{context}\n\n"
    "Question: {query} [/INST]"
)

_RAG_PROMPT_NO_CONTEXT = (
    "<s>[INST] Answer the following question concisely.\n\n"
    "Question: {query} [/INST]"
)


class Generator:
    """LLM generator wrapper with lazy loading and mock-mode.

    The generator loads its model through the shared
    :mod:`factuality_rag.model_registry` so that the gating probe
    and generator share the same weights (no double-loading).

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 ≈ greedy).
        do_sample: Whether to use sampling.
        mock_mode: If ``True``, return deterministic pseudo-answers.
        model: Optional pre-loaded model instance (skips registry).
        tokenizer: Optional pre-loaded tokenizer instance.

    Example::

        >>> g = Generator("mock", mock_mode=True)
        >>> g.generate("q", context="c")
        'Mock answer for query: q'
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        mock_mode: bool = False,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.mock_mode = mock_mode
        self._model = model
        self._tokenizer = tokenizer

    # ── Lazy loading via model registry ───────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load model + tokenizer through the shared registry.

        Skipped in mock-mode or if pre-loaded instances were passed.
        """
        if self.mock_mode:
            return
        if self._model is None:
            from factuality_rag.model_registry import get_model

            self._model = get_model(self.model_name, device=self.device)
        if self._tokenizer is None:
            from factuality_rag.model_registry import get_tokenizer

            self._tokenizer = get_tokenizer(self.model_name)

    # ── Public API ────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        passages: Optional[List[str]] = None,
    ) -> str:
        """Generate an answer given a query and optional context.

        Args:
            query: User question.
            context: Pre-formatted context string.
            passages: List of passage texts (joined if *context* is
                      ``None``).

        Returns:
            Generated answer string.

        Example::

            >>> g = Generator("mock", mock_mode=True)
            >>> g.generate("capital of France", context="Paris is the capital")
            'Mock answer for query: capital of France'
        """
        if self.mock_mode:
            return f"Mock answer for query: {query}"

        self._ensure_loaded()

        # Build context
        ctx = context or ""
        if not ctx and passages:
            ctx = "\n\n".join(passages)

        prompt = self._format_prompt(query, ctx)
        return self._generate_from_prompt(prompt)

    # ── Generation internals ──────────────────────────────────

    def _generate_from_prompt(self, prompt: str) -> str:
        """Tokenise, forward-pass, decode.

        Args:
            prompt: Fully formatted prompt string.

        Returns:
            Decoded answer text (prompt prefix stripped).
        """
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 1e-4),
                do_sample=self.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the NEW tokens (strip the prompt)
        generated_ids = outputs[0][input_length:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer.strip()

    # ── Prompt formatting ─────────────────────────────────────

    @staticmethod
    def _format_prompt(query: str, context: str) -> str:
        """Format a RAG prompt with context and query.

        Uses the Mistral ``[INST]`` chat template when context
        is present, and a simpler prompt otherwise.

        Args:
            query: User question.
            context: Retrieved passage context.

        Returns:
            Formatted prompt string.

        Example::

            >>> '[INST]' in Generator._format_prompt("q", "c")
            True
            >>> '[INST]' in Generator._format_prompt("q", "")
            True
        """
        if context:
            return _RAG_PROMPT_TEMPLATE.format(query=query, context=context)
        return _RAG_PROMPT_NO_CONTEXT.format(query=query)
