"""
tests.test_model_registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the model registry singleton.
"""

from __future__ import annotations

import pytest

from factuality_rag.model_registry import clear_registry, is_loaded


class TestModelRegistry:
    """Test suite for model_registry module."""

    def test_clear_registry(self) -> None:
        clear_registry()  # should not raise

    def test_is_loaded_returns_false_for_unknown(self) -> None:
        clear_registry()
        assert is_loaded("nonexistent-model") is False

    def test_clear_then_check(self) -> None:
        clear_registry()
        assert is_loaded("any-model") is False
