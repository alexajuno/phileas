from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ClientAdapter(ABC):
    """Abstract interface for client-specific hook execution details."""

    @abstractmethod
    def read_prompt(self, payload: dict) -> str:
        """Extract user prompt from hook payload."""
        pass

    @abstractmethod
    def parse_transcript(self, transcript_path: str) -> tuple[bool, str, str]:
        """Parse transcript to return (already_memorized, user_text, assistant_text)."""
        pass

    @abstractmethod
    def format_recall_output(self, content: str) -> Any:
        """Format the recall output for the client."""
        pass

    @abstractmethod
    def format_memorize_output(self, decision: str, reason: str) -> dict:
        """Format the stop hook output for the client."""
        pass


class ClaudeAdapter(ClientAdapter):
    """Adapter for Claude Code CLI integration."""

    def read_prompt(self, payload: dict) -> str:
        if isinstance(payload, dict):
            return str(payload.get("prompt", "")).strip()
        return ""

    def parse_transcript(self, transcript_path: str) -> tuple[bool, str, str]:
        try:
            with open(transcript_path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return False, "", ""

        boundary = None
        user_text = ""
        for i in range(len(lines) - 1, -1, -1):
            try:
                obj = json.loads(lines[i])
            except Exception:
                continue
            if obj.get("type") != "user":
                continue
            msg = obj.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                boundary = i
                user_text = msg["content"]
                break
        if boundary is None:
            return False, "", ""

        memorized = False
        texts: list[str] = []
        for line in lines[boundary + 1 :]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "assistant":
                continue
            msg = obj.get("message", {})
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                ct = c.get("type")
                if ct == "tool_use":
                    name = c.get("name", "") or ""
                    if "memorize" in name.lower():
                        memorized = True
                elif ct == "text":
                    texts.append(c.get("text", "") or "")
        return memorized, user_text, "\n".join(texts)

    def format_recall_output(self, content: str) -> Any:
        return content  # Prints plain text directly to stdout

    def format_memorize_output(self, decision: str, reason: str) -> dict:
        return {
            "decision": decision,
            "reason": reason,
        }


class AntigravityAdapter(ClientAdapter):
    """Adapter for Google Antigravity IDE integration."""

    def read_prompt(self, payload: dict) -> str:
        transcript_path = payload.get("transcriptPath") or payload.get("transcript_path")
        if not transcript_path or not Path(transcript_path).exists():
            return ""
        try:
            with open(transcript_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            for line in reversed(lines):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") == "USER_INPUT":
                    content = obj.get("content", "")
                    match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    return content.strip()
        except Exception:
            pass
        return ""

    def parse_transcript(self, transcript_path: str) -> tuple[bool, str, str]:
        try:
            with open(transcript_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return False, "", ""

        boundary = None
        user_text = ""
        for i in range(len(lines) - 1, -1, -1):
            try:
                obj = json.loads(lines[i])
            except Exception:
                continue
            if obj.get("type") == "USER_INPUT":
                boundary = i
                content = obj.get("content", "")
                match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                if match:
                    user_text = match.group(1).strip()
                else:
                    user_text = content.strip()
                break

        if boundary is None:
            return False, "", ""

        memorized = False
        assistant_texts = []
        for line in lines[boundary + 1 :]:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            tool_calls = obj.get("tool_calls") or []
            for tc in tool_calls:
                name = tc.get("name", "")
                if "memorize" in name.lower():
                    memorized = True

            if obj.get("type") == "PLANNER_RESPONSE":
                content = obj.get("content")
                if content:
                    assistant_texts.append(content)

        return memorized, user_text, "\n".join(assistant_texts)

    def format_recall_output(self, content: str) -> Any:
        # PreInvocation returns injectSteps to inject ephemeral system messages
        return {
            "injectSteps": [
                {
                    "ephemeralMessage": content,
                }
            ]
        }

    def format_memorize_output(self, decision: str, reason: str) -> dict:
        return {
            "decision": decision,
            "reason": reason,
        }


class CodexAdapter(ClientAdapter):
    """Adapter for Codex CLI integration."""

    def read_prompt(self, payload: dict) -> str:
        if isinstance(payload, dict):
            return str(payload.get("prompt", "")).strip()
        return ""

    def parse_transcript(self, transcript_path: str) -> tuple[bool, str, str]:
        try:
            with open(transcript_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return False, "", ""

        boundary = None
        user_text = ""
        for i in range(len(lines) - 1, -1, -1):
            try:
                obj = json.loads(lines[i])
            except Exception:
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            if obj.get("type") == "event_msg" and payload.get("type") == "user_message":
                message = payload.get("message")
                if isinstance(message, str):
                    boundary = i
                    user_text = message
                    break
            if (
                obj.get("type") == "response_item"
                and payload.get("type") == "message"
                and payload.get("role") == "user"
            ):
                content = payload.get("content")
                text = _extract_codex_text(content)
                if text:
                    boundary = i
                    user_text = text
                    break

        if boundary is None:
            return False, "", ""

        memorized = False
        assistant_texts: list[str] = []
        for line in lines[boundary + 1 :]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue

            if payload.get("type") == "function_call":
                name = payload.get("name", "") or ""
                if "memorize" in name.lower():
                    memorized = True
            elif obj.get("type") == "event_msg" and payload.get("type") == "agent_message":
                message = payload.get("message")
                if isinstance(message, str) and message:
                    assistant_texts.append(message)
            elif (
                obj.get("type") == "response_item"
                and payload.get("type") == "message"
                and payload.get("role") == "assistant"
            ):
                text = _extract_codex_text(payload.get("content"))
                if text:
                    assistant_texts.append(text)

        return memorized, user_text.strip(), "\n".join(assistant_texts)

    def format_recall_output(self, content: str) -> Any:
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": content,
            }
        }

    def format_memorize_output(self, decision: str, reason: str) -> dict:
        return {
            "decision": decision,
            "reason": reason,
        }


def _extract_codex_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"input_text", "output_text"}:
            text_parts.append(block.get("text", "") or "")
    return "".join(text_parts)


def get_adapter(client_name: str) -> ClientAdapter:
    """Return the client adapter for the specified client name."""
    if client_name == "codex":
        return CodexAdapter()
    if client_name == "antigravity":
        return AntigravityAdapter()
    return ClaudeAdapter()
