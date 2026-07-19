"""A LangChain chat model that runs `claude -p` against the Claude Code subscription.

This is Phileas's subscription-backed extraction provider: it shells out to the
Claude Code CLI in print mode, so an extraction call draws on the user's Max plan
instead of an API key. Auth is the CLI's own (the logged-in credentials or
``CLAUDE_CODE_OAUTH_TOKEN``); ``ANTHROPIC_API_KEY`` is stripped from the child's
environment so a generic key on the box never silently reroutes the call to
paid API billing.

The CLI is the full agent runtime, so it has no native structured-output binding
the way a raw API model does. ``with_structured_output`` therefore takes the
prompt-based route: it appends the schema's JSON Schema to the prompt, asks for a
bare JSON object, and validates the reply against the Pydantic model. That is the
same shape ``LLMClient.invoke_structured`` consumes from the tool-calling
providers, so the extraction worker is unaware of which provider it holds.

MCP is disabled per call (``--strict-mcp-config`` with an empty config) and the
call runs from a neutral working directory, so extraction context stays the base
runtime rather than whatever project the daemon happens to sit in.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

# Print-mode call that pins the model, returns the JSON result envelope, and loads
# no MCP servers. Kept identical to the validated spike so behavior matches what
# was measured.
_BASE_FLAGS = ("--output-format", "json", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}')


class ClaudeCodeError(RuntimeError):
    """A `claude -p` call failed to produce a usable result envelope."""


def _parse_json_object(text: str) -> Any:
    """Parse a JSON object from model text, tolerating a stray fence or preamble.

    The prompt asks for a bare object, and Sonnet obliges, but a defensive parse
    keeps one stray ```json fence or leading sentence from failing an otherwise
    good extraction.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        stripped = stripped[4:] if stripped.startswith("json") else stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


class PhileasClaudeCodeChat(BaseChatModel):
    """Run extraction through `claude -p` on the Claude Code subscription."""

    model: str = "sonnet"
    timeout_s: float = 300.0
    binary: str = "claude"

    @property
    def _llm_type(self) -> str:
        return "phileas-claude-code-cli"

    def _call_cli(self, user_prompt: str, system: str | None) -> tuple[str, dict]:
        """Run one print-mode call; return the model's text and its usage block."""
        cmd = [self.binary, "-p", user_prompt, "--model", self.model, *_BASE_FLAGS]
        if system:
            cmd += ["--append-system-prompt", system]

        # Force the subscription: a generic ANTHROPIC_API_KEY on the box would
        # otherwise take precedence and bill the paid API instead.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=tempfile.gettempdir(),
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError(f"claude -p timed out after {self.timeout_s}s") from exc

        if proc.returncode != 0:
            raise ClaudeCodeError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeError(f"claude -p returned non-JSON: {proc.stdout.strip()[:500]}") from exc
        if envelope.get("is_error") or envelope.get("subtype") != "success":
            raise ClaudeCodeError(
                f"claude -p error (subtype={envelope.get('subtype')}): {str(envelope.get('result'))[:500]}"
            )
        return envelope.get("result", ""), envelope.get("usage") or {}

    @staticmethod
    def _usage_metadata(usage: dict) -> dict:
        """Fold the envelope's token counts into LangChain's usage_metadata shape.

        The CLI reports cache-created and cache-read input separately; the ledger
        wants total input, so all three input buckets sum into ``input_tokens``.
        """
        input_tokens = (
            (usage.get("input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0)
        )
        output_tokens = usage.get("output_tokens") or 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, user = _split_messages(messages)
        text, usage = self._call_cli(user, system)
        message = AIMessage(content=text, usage_metadata=self._usage_metadata(usage))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any) -> Runnable:
        """Prompt-based structured output: ask for JSON matching ``schema``, validate.

        Returns the same ``{"raw", "parsed", "parsing_error"}`` shape as the
        tool-calling providers when ``include_raw`` is set, so
        ``LLMClient.invoke_structured`` is provider-agnostic. ``raw`` carries the
        usage the ledger reads.
        """
        json_schema = schema.model_json_schema()
        instruction = (
            "\n\nReturn ONLY a single JSON object matching this JSON Schema, with no prose "
            "and no markdown fences:\n" + json.dumps(json_schema)
        )

        def _invoke(model_input: Any) -> Any:
            system, user = _coerce_input(model_input)
            text, usage = self._call_cli(user + instruction, system)
            raw = AIMessage(content=text, usage_metadata=self._usage_metadata(usage))
            try:
                parsed = schema.model_validate(_parse_json_object(text))
            except Exception as exc:  # parse or validation failure
                if include_raw:
                    return {"raw": raw, "parsed": None, "parsing_error": exc}
                raise
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(_invoke)


def _split_messages(messages: list[BaseMessage]) -> tuple[str | None, str]:
    """Split a message list into (system, joined-user-text) for a print-mode call."""
    system_parts: list[str] = []
    body_parts: list[str] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.type == "system":
            system_parts.append(content)
        else:
            body_parts.append(content)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, "\n\n".join(body_parts)


def _coerce_input(model_input: Any) -> tuple[str | None, str]:
    """Coerce a runnable input (str, message list, or prompt value) to (system, user)."""
    if isinstance(model_input, str):
        return None, model_input
    if isinstance(model_input, BaseMessage):
        return _split_messages([model_input])
    if isinstance(model_input, list):
        return _split_messages(model_input)
    to_messages = getattr(model_input, "to_messages", None)
    if callable(to_messages):
        return _split_messages(to_messages())
    return None, str(model_input)
