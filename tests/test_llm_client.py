"""Tests for the internal extraction LLM client.

These run offline: ``LLMClient`` takes an injected fake Anthropic client, so no
SDK import and no network call happen. They cover the response helpers, the
availability gate, request assembly, and usage accounting.
"""

from types import SimpleNamespace

import pytest

from phileas.config import LLMConfig
from phileas.llm import LLMClient, parse_json_response, text_from, tool_input_from
from phileas.llm.client import _cost_usd

# -- Response parsing helpers ------------------------------------------------


class TestParseJsonResponse:
    def test_bare_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fence_without_lang(self):
        assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}

    def test_trailing_prose_after_value(self):
        assert parse_json_response('{"a": 1}\n\nThat is the answer.') == {"a": 1}

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_json_response("not json at all")


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(name, payload):
    return SimpleNamespace(type="tool_use", name=name, input=payload)


class TestContentExtractors:
    def test_text_from_concatenates_text_blocks(self):
        msg = SimpleNamespace(content=[_text_block("Hello "), _tool_block("x", {}), _text_block("world")])
        assert text_from(msg) == "Hello world"

    def test_text_from_empty_when_no_text(self):
        assert text_from(SimpleNamespace(content=[_tool_block("x", {})])) == ""
        assert text_from(SimpleNamespace(content=None)) == ""

    def test_tool_input_first_match(self):
        msg = SimpleNamespace(content=[_tool_block("a", {"k": 1}), _tool_block("b", {"k": 2})])
        assert tool_input_from(msg) == {"k": 1}

    def test_tool_input_by_name(self):
        msg = SimpleNamespace(content=[_tool_block("a", {"k": 1}), _tool_block("b", {"k": 2})])
        assert tool_input_from(msg, "b") == {"k": 2}

    def test_tool_input_none_when_absent(self):
        assert tool_input_from(SimpleNamespace(content=[_text_block("hi")]), "b") is None


# -- Cost derivation ---------------------------------------------------------


class TestCost:
    def test_known_model_priced(self):
        # Haiku 4.5: $1/MTok input, $5/MTok output.
        assert _cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == pytest.approx(6.0)

    def test_unknown_model_zero(self):
        assert _cost_usd("some-other-model", 1_000_000, 1_000_000) == 0.0


# -- Fakes for the SDK + usage tracker ---------------------------------------


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropic:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


class _RaisingAnthropic:
    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    def __init__(self):
        self.messages = self._Messages()


class _FakeTracker:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, **kwargs):
        self.records.append(kwargs)


def _message(text="ok", input_tokens=10, output_tokens=20):
    return SimpleNamespace(
        content=[_text_block(text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# -- Availability gate -------------------------------------------------------


class TestAvailability:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        assert LLMClient(LLMConfig()).available is False

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", "sk-test")
        assert LLMClient(LLMConfig()).available is True


# -- complete() --------------------------------------------------------------


class TestComplete:
    def test_returns_response_and_records_usage(self):
        tracker = _FakeTracker()
        fake = _FakeAnthropic(_message(input_tokens=100, output_tokens=200))
        client = LLMClient(LLMConfig(model="claude-haiku-4-5-20251001"), tracker, client=fake)

        result = client.complete("extraction", messages=[{"role": "user", "content": "hi"}])

        assert text_from(result) == "ok"
        assert len(tracker.records) == 1
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

    def test_passes_optional_params_through(self):
        fake = _FakeAnthropic(_message())
        client = LLMClient(LLMConfig(), client=fake)
        tools = [{"name": "record", "input_schema": {"type": "object"}}]
        choice = {"type": "tool", "name": "record"}

        client.complete(
            "extraction",
            messages=[{"role": "user", "content": "x"}],
            system="be terse",
            tools=tools,
            tool_choice=choice,
            max_tokens=512,
        )

        sent = fake.messages.calls[0]
        assert sent["system"] == "be terse"
        assert sent["tools"] == tools
        assert sent["tool_choice"] == choice
        assert sent["max_tokens"] == 512

    def test_omits_optional_params_when_unset(self):
        fake = _FakeAnthropic(_message())
        client = LLMClient(LLMConfig(), client=fake)
        client.complete("extraction", messages=[{"role": "user", "content": "x"}])
        sent = fake.messages.calls[0]
        assert "system" not in sent
        assert "tools" not in sent
        assert "tool_choice" not in sent

    def test_failure_records_and_reraises(self):
        tracker = _FakeTracker()
        client = LLMClient(LLMConfig(), tracker, client=_RaisingAnthropic())
        with pytest.raises(RuntimeError, match="boom"):
            client.complete("extraction", messages=[{"role": "user", "content": "x"}])
        assert len(tracker.records) == 1
        assert tracker.records[0]["success"] is False
        assert "boom" in tracker.records[0]["error"]

    def test_no_tracker_is_fine(self):
        client = LLMClient(LLMConfig(), client=_FakeAnthropic(_message()))
        # Should not raise despite no usage tracker.
        assert text_from(client.complete("extraction", messages=[{"role": "user", "content": "x"}])) == "ok"
