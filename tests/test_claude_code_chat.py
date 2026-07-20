"""Tests for the claude_code subscription extraction provider.

These run offline: ``subprocess.run`` is monkeypatched so no `claude` CLI is
spawned. They cover command construction (model pin, MCP off, key stripped),
result-envelope handling, and the prompt-based structured-output contract that
``LLMClient.invoke_structured`` depends on.
"""

import json
import subprocess

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from phileas.llm.claude_code_chat import (  # noqa: E402
    ClaudeCodeError,
    PhileasClaudeCodeChat,
    _coerce_input,
    _parse_json_object,
    _split_messages,
)


class _Out(BaseModel):
    value: str


def _envelope(result: str, **usage) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result,
            "usage": usage or {"input_tokens": 3, "output_tokens": 5},
        }
    )


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# -- JSON tolerance ----------------------------------------------------------


class TestParseJson:
    def test_clean(self):
        assert _parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_preamble_and_trailer(self):
        assert _parse_json_object('Sure, here:\n{"a": 1}\nHope that helps') == {"a": 1}

    def test_garbage_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_object("no json here")


# -- Message coercion --------------------------------------------------------


class TestMessages:
    def test_split_system_and_body(self):
        system, user = _split_messages([SystemMessage(content="sys"), HumanMessage(content="hi")])
        assert system == "sys"
        assert user == "hi"

    def test_coerce_plain_string(self):
        assert _coerce_input("hello") == (None, "hello")


class TestUsage:
    def test_sums_all_input_buckets(self):
        usage = PhileasClaudeCodeChat._usage_metadata(
            {
                "input_tokens": 2,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": 4,
            }
        )
        assert usage == {"input_tokens": 32, "output_tokens": 4, "total_tokens": 36}


# -- CLI invocation ----------------------------------------------------------


class TestCallCli:
    def test_command_pins_model_disables_mcp_and_strips_key(self, monkeypatch):
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input")
            captured["env"] = kwargs.get("env")
            return _FakeProc(stdout=_envelope("hi", input_tokens=1, output_tokens=2))

        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-stripped")
        monkeypatch.setattr(subprocess, "run", fake_run)

        text, usage = PhileasClaudeCodeChat(model="sonnet")._call_cli("prompt-body", "sys")

        assert text == "hi"
        assert usage == {"input_tokens": 1, "output_tokens": 2}
        cmd = captured["cmd"]
        assert cmd[:4] == ["claude", "-p", "--model", "sonnet"]
        assert "--strict-mcp-config" in cmd
        assert "--append-system-prompt" in cmd
        # The prompt rides stdin, so a session larger than the 128 KB argv cap still runs.
        assert captured["input"] == "prompt-body"
        assert "prompt-body" not in cmd
        # The subscription is forced: a generic key must not reach the child.
        assert "ANTHROPIC_API_KEY" not in captured["env"]

    def test_oversized_prompt_is_not_passed_as_an_argument(self, monkeypatch):
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input")
            return _FakeProc(stdout=_envelope("ok"))

        monkeypatch.setattr(subprocess, "run", fake_run)

        huge = "x" * 200_000  # over MAX_ARG_STRLEN (128 KB), the old E2BIG failure
        PhileasClaudeCodeChat()._call_cli(huge, None)

        assert captured["input"] == huge
        assert all(len(arg) < 4096 for arg in captured["cmd"])

    def test_nonzero_exit_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _FakeProc(stderr="boom", returncode=1))
        with pytest.raises(ClaudeCodeError, match="exited 1"):
            PhileasClaudeCodeChat()._call_cli("p", None)

    def test_error_envelope_raises(self, monkeypatch):
        env = json.dumps({"subtype": "error_during_execution", "is_error": True, "result": "nope"})
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _FakeProc(stdout=env))
        with pytest.raises(ClaudeCodeError, match="error"):
            PhileasClaudeCodeChat()._call_cli("p", None)

    def test_non_json_stdout_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _FakeProc(stdout="not json"))
        with pytest.raises(ClaudeCodeError, match="non-JSON"):
            PhileasClaudeCodeChat()._call_cli("p", None)


# -- Structured output (the invoke_structured contract) ----------------------


class TestStructuredOutput:
    def test_include_raw_success(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _FakeProc(stdout=_envelope('{"value": "ok"}')))
        out = PhileasClaudeCodeChat().with_structured_output(_Out, include_raw=True).invoke("json please")
        assert out["parsed"] == _Out(value="ok")
        assert out["parsing_error"] is None
        assert out["raw"].usage_metadata["input_tokens"] == 3

    def test_include_raw_parse_failure_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _FakeProc(stdout=_envelope("not json at all")))
        out = PhileasClaudeCodeChat().with_structured_output(_Out, include_raw=True).invoke("x")
        assert out["parsed"] is None
        assert out["parsing_error"] is not None

    def test_schema_instruction_appended_to_prompt(self, monkeypatch):
        seen: dict = {}

        def fake_run(cmd, **kwargs):
            seen["user"] = kwargs.get("input")
            return _FakeProc(stdout=_envelope('{"value": "ok"}'))

        monkeypatch.setattr(subprocess, "run", fake_run)
        PhileasClaudeCodeChat().with_structured_output(_Out).invoke("base prompt")
        assert "base prompt" in seen["user"]
        assert "JSON Schema" in seen["user"]
