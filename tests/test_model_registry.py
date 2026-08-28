"""
tests.test_model_registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the model registry singleton.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import factuality_rag.model_registry as registry
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


def test_4bit_runtime_fails_closed_with_actionable_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match=r"factuality_rag\[quantization\]"):
        registry._require_4bit_runtime()


@pytest.mark.parametrize(
    ("device", "cuda_available", "current_device", "device_count", "expected_device_map"),
    [
        ("cuda", True, 1, 2, {"": 1}),
        ("cuda:1", True, 0, 2, {"": 1}),
        ("cpu", False, 0, 0, {"": "cpu"}),
    ],
)
def test_quantized_model_honors_explicit_requested_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    cuda_available: bool,
    current_device: int,
    device_count: int,
    expected_device_map: dict[str, Any],
) -> None:
    import torch
    import transformers

    registry.clear_registry()
    load_call: dict[str, Any] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            load_call["model_id"] = model_id
            load_call["kwargs"] = kwargs
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(registry, "_require_4bit_runtime", lambda: None)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)

    model = registry.get_model("quantized-model", device=device, quantize_4bit=True)

    assert load_call["model_id"] == "quantized-model"
    assert load_call["kwargs"]["device_map"] == expected_device_map
    assert isinstance(load_call["kwargs"]["quantization_config"], FakeBitsAndBytesConfig)
    assert model.eval_called is True
    registry.clear_registry()


def test_requested_cuda_must_be_available_before_quantization_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry.clear_registry()

    def fail_if_quantization_setup_runs() -> None:
        raise AssertionError("quantization setup ran before CUDA validation")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(registry, "_require_4bit_runtime", fail_if_quantization_setup_runs)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        registry.get_model("quantized-model", device="cuda", quantize_4bit=True)
    assert registry.is_loaded("quantized-model") is False


def test_requested_cuda_index_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    registry.clear_registry()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match=r"only 1 CUDA device"):
        registry.get_model("quantized-model", device="cuda:1", quantize_4bit=True)
    assert registry.is_loaded("quantized-model") is False


@pytest.mark.parametrize(
    "requested",
    [
        ("cuda", True, False),
        ("cuda", False, True),
        ("cpu", False, False),
    ],
)
def test_registry_rejects_cached_model_with_incompatible_load_settings(
    requested: tuple[str, bool, bool],
) -> None:
    clear_registry()
    registry._models["model"] = object()
    registry._model_load_configs["model"] = ("cuda", False, False)

    with pytest.raises(RuntimeError, match="incompatible with requested settings"):
        registry.get_model(
            "model",
            device=requested[0],
            quantize_4bit=requested[1],
            trust_remote_code=requested[2],
        )
    clear_registry()


def test_registry_returns_cache_for_identical_load_settings() -> None:
    clear_registry()
    cached = object()
    registry._models["model"] = cached
    registry._model_load_configs["model"] = ("cuda", True, False)

    assert registry.get_model("model") is cached
    clear_registry()


def test_tokenizer_registry_rejects_incompatible_trust_setting() -> None:
    clear_registry()
    registry._tokenizers["tokenizer"] = object()
    registry._tokenizer_load_configs["tokenizer"] = False

    with pytest.raises(RuntimeError, match="incompatible with requested"):
        registry.get_tokenizer("tokenizer", trust_remote_code=True)
    clear_registry()


def test_tokenizer_registry_returns_cache_for_identical_trust_setting() -> None:
    clear_registry()
    cached = object()
    registry._tokenizers["tokenizer"] = cached
    registry._tokenizer_load_configs["tokenizer"] = False

    assert registry.get_tokenizer("tokenizer") is cached
    clear_registry()


def test_concurrent_first_model_load_returns_one_shared_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    registry.clear_registry()
    first_load_entered = threading.Event()
    release_first_load = threading.Event()
    second_call_started = threading.Event()
    duplicate_load_entered = threading.Event()
    loaded_models: list[FakeModel] = []
    loaded_models_lock = threading.Lock()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            del model_id, kwargs
            model = FakeModel()
            with loaded_models_lock:
                loaded_models.append(model)
                call_number = len(loaded_models)
            if call_number == 1:
                first_load_entered.set()
                if not release_first_load.wait(timeout=3.0):
                    raise AssertionError("timed out waiting to finish first model load")
            else:
                duplicate_load_entered.set()
            return model

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)

    def load_model(*, mark_started: bool = False) -> FakeModel:
        if mark_started:
            second_call_started.set()
        return registry.get_model(
            "concurrent-model",
            device="cpu",
            quantize_4bit=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(load_model)
        try:
            assert first_load_entered.wait(timeout=2.0)
            second_future = executor.submit(load_model, mark_started=True)
            assert second_call_started.wait(timeout=2.0)
            duplicate_load_observed = duplicate_load_entered.wait(timeout=0.5)
        finally:
            release_first_load.set()
        results = [first_future.result(timeout=2.0), second_future.result(timeout=2.0)]

    assert duplicate_load_observed is False
    assert len(loaded_models) == 1
    assert results[0] is results[1] is loaded_models[0]
    assert loaded_models[0].eval_called is True
    assert registry._model_load_configs["concurrent-model"] == ("cpu", False, False)
    registry.clear_registry()


def test_concurrent_first_tokenizer_load_returns_one_shared_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    registry.clear_registry()
    first_load_entered = threading.Event()
    release_first_load = threading.Event()
    second_call_started = threading.Event()
    duplicate_load_entered = threading.Event()
    loaded_tokenizers: list[object] = []
    loaded_tokenizers_lock = threading.Lock()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del model_id, kwargs
            tokenizer = object()
            with loaded_tokenizers_lock:
                loaded_tokenizers.append(tokenizer)
                call_number = len(loaded_tokenizers)
            if call_number == 1:
                first_load_entered.set()
                if not release_first_load.wait(timeout=3.0):
                    raise AssertionError("timed out waiting to finish first tokenizer load")
            else:
                duplicate_load_entered.set()
            return tokenizer

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)

    def load_tokenizer(*, mark_started: bool = False) -> object:
        if mark_started:
            second_call_started.set()
        return registry.get_tokenizer("concurrent-tokenizer")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(load_tokenizer)
        try:
            assert first_load_entered.wait(timeout=2.0)
            second_future = executor.submit(load_tokenizer, mark_started=True)
            assert second_call_started.wait(timeout=2.0)
            duplicate_load_observed = duplicate_load_entered.wait(timeout=0.5)
        finally:
            release_first_load.set()
        results = [first_future.result(timeout=2.0), second_future.result(timeout=2.0)]

    assert duplicate_load_observed is False
    assert len(loaded_tokenizers) == 1
    assert results[0] is results[1] is loaded_tokenizers[0]
    assert registry._tokenizer_load_configs["concurrent-tokenizer"] is False
    registry.clear_registry()
