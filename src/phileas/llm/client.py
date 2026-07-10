"""LangChain-backed client for Phileas's internal extraction calls.

The daemon constructs one ``LLMClient`` after the engine loads and hands it to
the extraction worker. The client owns two concerns: building a chat model for
the configured provider, and running one structured-output call while recording
token usage to the existing ``UsageTracker``.

Provider portability is the reason for LangChain here. ``build_chat_model`` maps
``LLMConfig.provider`` onto a LangChain chat adapter (Anthropic, OpenAI, or a
local Ollama), so switching the model Phileas extracts with is a config change,
not a code change. Each adapter is imported lazily inside the branch that needs
it, so importing this module stays cheap on the no-extraction path and a provider
whose adapter is not installed only fails when it is actually selected.

``invoke_structured`` is synchronous (the worker runs in a daemon thread, like
the reinforcement loop). It asks the model for output shaped to a Pydantic schema
via ``with_structured_output`` — for a tool-calling provider that is forced tool
use under the hood, the same mechanism the hand-rolled client drove by hand, now
with validation and provider-agnostic. ``include_raw=True`` keeps the underlying
message alongside the parsed object so token usage still reaches the ledger.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel

    from phileas.config import LLMConfig

# Output-token cap for an extraction call. Fixed rather than configurable: the
# extraction prompt returns a small, bounded set of memories, so this is a
# safety ceiling, not a knob to hand-tune.
DEFAULT_MAX_TOKENS = 2048

# Per-model price in USD per million (input, output) tokens. LangChain reports
# token counts uniformly across providers, but not cost, so we derive it for the
# usage ledger; an unrecognized model records its tokens with zero cost rather
# than a wrong guess. Source: the claude-api model/pricing reference.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}


# Providers the extraction client can talk to, one LangChain adapter each. The
# tuple is the offered set for a provider picker and the guard in
# ``build_chat_model``. ``ollama`` runs a model locally with no API key.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "ollama")

# The env var each keyed provider reads its credential from by default. Namespaced
# with ``PHILEAS_`` so Phileas's key never collides with the host agent's generic
# ``ANTHROPIC_API_KEY``/``OPENAI_API_KEY``. ``set-provider`` writes the matching
# name into ``api_key_env`` when the provider changes; a keyless provider (Ollama)
# has no entry.
_DEFAULT_API_KEY_ENV: dict[str, str] = {
    "anthropic": "PHILEAS_ANTHROPIC_API_KEY",
    "openai": "PHILEAS_OPENAI_API_KEY",
}


def default_api_key_env(provider: str) -> str | None:
    """The conventional key env var for ``provider``; ``None`` for a keyless one."""
    return _DEFAULT_API_KEY_ENV.get(provider)


def known_models() -> list[str]:
    """Model names with known pricing — the CLI's suggestion set for ``set-model``.

    Any model string is accepted; these are the ones whose cost the usage ledger
    can derive, so they make the natural offered choices.
    """
    return sorted(_PRICE_PER_MTOK)


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Derive call cost from the per-model price table; 0.0 for unknown models."""
    price = _PRICE_PER_MTOK.get(model)
    if not price:
        return 0.0
    price_in, price_out = price
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


def build_chat_model(
    config: LLMConfig, *, home: Path | None = None, max_tokens: int = DEFAULT_MAX_TOKENS
) -> BaseChatModel:
    """Construct the LangChain chat model for the configured provider.

    The key never lives in config: it is resolved at call time (environment first,
    then the profile's stored secrets file under ``home``; see
    :func:`phileas.secrets.resolve_key`) and passed to the adapter explicitly, so
    Phileas's namespaced key (``PHILEAS_ANTHROPIC_API_KEY``) is used rather than the
    generic ``ANTHROPIC_API_KEY`` the host agent may also hold. A keyless local
    provider (Ollama) takes no key.

    Adapters are imported inside their branch so only the selected provider's
    package needs to be installed.
    """
    from phileas import secrets

    provider = config.provider
    api_key = secrets.resolve_key(home, config.api_key_env)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model, max_tokens=max_tokens, api_key=api_key)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model, max_tokens=max_tokens, api_key=api_key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Local server, no key. num_predict is Ollama's output-token cap.
        return ChatOllama(model=config.model, num_predict=max_tokens)

    raise ValueError(f"unsupported extraction provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}")


TSchema = TypeVar("TSchema", bound="BaseModel")


class LLMClient:
    """Synchronous, provider-agnostic wrapper for daemon-side structured calls."""

    def __init__(
        self,
        config: LLMConfig,
        usage_tracker: Any | None = None,
        *,
        home: Path | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        """``home`` locates the stored secrets file for key resolution; ``model`` is
        injectable so tests run without an adapter or a network."""
        self._config = config
        self._usage = usage_tracker
        self._home = home
        self._model = model

    @property
    def available(self) -> bool:
        """True when the configured provider can run (key reachable, or keyless)."""
        from phileas.config import key_reachable

        return key_reachable(self._config, self._home)

    def _chat_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = build_chat_model(self._config, home=self._home)
        return self._model

    def invoke_structured(self, operation: str, schema: type[TSchema], messages: Any) -> TSchema:
        """Run one call whose output is validated against ``schema`` and return it.

        ``with_structured_output(schema, include_raw=True)`` binds the schema as
        the model's response shape and returns ``{"raw", "parsed", "parsing_error"}``:
        the parsed Pydantic instance for the caller, the raw message for the usage
        it carries. A parse failure or an empty parse raises, so the worker records
        the failure against the source events instead of writing a guessed memory.

        Usage (tokens, cost, latency, success) is recorded in a ``finally`` so a
        failed call is still accounted for. Exceptions propagate.
        """
        structured = self._chat_model().with_structured_output(schema, include_raw=True)

        start = perf_counter()
        success = True
        error: str | None = None
        input_tokens = 0
        output_tokens = 0
        try:
            result = structured.invoke(messages)

            raw = result.get("raw")
            usage = getattr(raw, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0

            if result.get("parsing_error"):
                raise ValueError(f"structured output parse failed: {result['parsing_error']}")
            parsed = result.get("parsed")
            if parsed is None:
                raise ValueError("structured output returned no parsed value")
            return parsed
        except Exception as exc:
            success = False
            error = str(exc)[:500]
            raise
        finally:
            self._record(
                operation,
                self._config.model,
                input_tokens,
                output_tokens,
                (perf_counter() - start) * 1000,
                success,
                error,
            )

    def _record(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error: str | None,
    ) -> None:
        if self._usage is None:
            return
        try:
            self._usage.record(
                operation=operation,
                model=model,
                provider=self._config.provider,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=_cost_usd(model, input_tokens, output_tokens),
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        except Exception:
            # Usage accounting must never break an extraction call.
            pass
