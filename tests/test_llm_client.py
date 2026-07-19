"""Tests for the internal extraction LLM client.

These run offline and without any LangChain adapter installed: ``LLMClient``
takes an injected fake chat model, so no adapter import and no network call
happen. They cover cost derivation, the availability gate, the structured-output
call, and usage accounting.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from phileas.config import LLMConfig
from phileas.llm import LLMClient
from phileas.llm.client import _cost_usd, build_chat_model


class _Out(BaseModel):
    """A throwaway schema for the generic structured call."""

    value: str


# -- Cost derivation ---------------------------------------------------------


class TestCost:
    def test_known_model_priced(self):
        # Haiku 4.5: $1/MTok input, $5/MTok output.
        assert _cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == pytest.approx(6.0)

    def test_unknown_model_zero(self):
        assert _cost_usd("some-other-model", 1_000_000, 1_000_000) == 0.0


# -- Fakes for the chat model + usage tracker --------------------------------


class _FakeStructured:
    """Stand-in for the runnable returned by ``with_structured_output``."""

    def __init__(self, result):
        self._result = result
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._result


class _FakeChatModel:
    """Records the schema it is bound to and returns a canned structured result."""

    def __init__(self, result):
        self._result = result
        self.structured_calls: list[tuple] = []
        self.bound: _FakeStructured | None = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_calls.append((schema, kwargs))
        self.bound = _FakeStructured(self._result)
        return self.bound


class _FakeTracker:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, **kwargs):
        self.records.append(kwargs)


def _raw(input_tokens=10, output_tokens=20):
    """A raw message carrying LangChain-style usage metadata."""
    return SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    )


def _result(parsed, *, error=None, input_tokens=10, output_tokens=20):
    """The ``include_raw=True`` envelope: raw message, parsed object, parse error."""
    return {"raw": _raw(input_tokens, output_tokens), "parsed": parsed, "parsing_error": error}


# -- Availability gate -------------------------------------------------------


# A keyed provider config, so availability keys on the key (the default provider,
# claude_code, is keyless and always available).
_KEYED = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001", api_key_env="PHILEAS_ANTHROPIC_API_KEY")


class TestAvailability:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        assert LLMClient(_KEYED).available is False

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", "sk-test")
        assert LLMClient(_KEYED).available is True

    def test_keyless_claude_code_available_without_key(self, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        assert LLMClient(LLMConfig()).available is True

    def test_keyless_provider_available_without_key(self, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        assert LLMClient(LLMConfig(provider="ollama", model="llama3.1")).available is True


# -- build_chat_model --------------------------------------------------------


class TestBuildChatModel:
    def test_unknown_provider_raises(self):
        # Guards before any adapter import, so this is safe with no extra installed.
        with pytest.raises(ValueError, match="unsupported extraction provider"):
            build_chat_model(LLMConfig(provider="grok"))


# -- invoke_structured -------------------------------------------------------


class TestInvokeStructured:
    def test_returns_parsed_and_records_usage(self):
        tracker = _FakeTracker()
        model = _FakeChatModel(_result(_Out(value="ok"), input_tokens=100, output_tokens=200))
        client = LLMClient(LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001"), tracker, model=model)

        out = client.invoke_structured("extraction", _Out, [("human", "hi")])

        assert out == _Out(value="ok")
        # The schema was bound with the raw message kept for usage.
        schema, kwargs = model.structured_calls[0]
        assert schema is _Out
        assert kwargs == {"include_raw": True}
        assert model.bound.calls == [[("human", "hi")]]

        rec = tracker.records[0]
        assert rec["operation"] == "extraction"
        assert rec["model"] == "claude-haiku-4-5-20251001"
        assert rec["provider"] == "anthropic"
        assert rec["prompt_tokens"] == 100
        assert rec["completion_tokens"] == 200
        assert rec["total_tokens"] == 300
        assert rec["cost_usd"] == pytest.approx(100 / 1e6 * 1.0 + 200 / 1e6 * 5.0)
        assert rec["success"] is True
        assert rec["error"] is None

    def test_parse_error_raises_and_records_failure(self):
        tracker = _FakeTracker()
        model = _FakeChatModel(_result(None, error=ValueError("bad")))
        client = LLMClient(LLMConfig(), tracker, model=model)

        with pytest.raises(ValueError, match="structured output parse failed"):
            client.invoke_structured("extraction", _Out, "x")

        assert tracker.records[0]["success"] is False
        # Usage is still accounted for even on a failed parse.
        assert tracker.records[0]["prompt_tokens"] == 10

    def test_empty_parse_raises(self):
        model = _FakeChatModel(_result(None))
        client = LLMClient(LLMConfig(), model=model)
        with pytest.raises(ValueError, match="no parsed value"):
            client.invoke_structured("extraction", _Out, "x")

    def test_no_tracker_is_fine(self):
        model = _FakeChatModel(_result(_Out(value="ok")))
        client = LLMClient(LLMConfig(), model=model)
        assert client.invoke_structured("extraction", _Out, "x") == _Out(value="ok")
