"""LLM providers for the analyst report.

Each provider turns a ``(system, user_text, params)`` triple into assistant
prose.  Three implementations are offered:

* :class:`LocalProvider` — POST to the benchmarked OpenAI-compatible
  endpoint (the same wire format the suite already uses elsewhere).
* :class:`OpenAICompatProvider` — the same wire format against an arbitrary
  ``url`` / ``model`` (another vLLM/SGLang/OpenAI server).
* :class:`ClaudeProvider` — the OFFICIAL ``anthropic`` SDK with adaptive
  thinking.  Never an OpenAI-compatible shim.

Any failure (network, missing SDK, API error) is normalized to
:class:`ProviderError` so the analyst can fall back to the deterministic
template without the run ever blocking.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import requests


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a report.

    The analyst catches this and falls back to the deterministic report;
    it never propagates out of report generation.
    """


@runtime_checkable
class Provider(Protocol):
    """A text-generation provider for the analyst report."""

    def generate(self, system: str, user_text: str, params: dict) -> str:
        """Return assistant prose for the given system + user prompt.

        Raises:
            ProviderError: on any failure.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible providers
# ---------------------------------------------------------------------------

def _post_chat_completions(
    url: str,
    system: str,
    user_text: str,
    params: dict,
) -> str:
    """POST a chat-completion request and return the assistant text.

    Uses ``temperature=0`` for determinism and passes a ``seed`` when one is
    supplied in ``params``.  Mirrors the suite's existing request style
    (``requests`` against ``/v1/chat/completions``).

    Raises:
        ProviderError: on a network error, a non-200 status, or a response
            that does not carry an assistant message.
    """
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint = f"{endpoint}/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": params.get("model") or "default",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    }
    if params.get("max_tokens"):
        payload["max_tokens"] = params["max_tokens"]
    if params.get("seed") is not None:
        payload["seed"] = params["seed"]

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=params.get("timeout", 120),
        )
    except requests.RequestException as exc:  # network unreachable, timeout, ...
        raise ProviderError(f"request to {endpoint} failed: {exc}") from exc

    if response.status_code != 200:
        raise ProviderError(f"{endpoint} returned status {response.status_code}")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"malformed response from {endpoint}: {exc}") from exc


class LocalProvider:
    """Generate the report on the benchmarked OpenAI-compatible endpoint.

    The target URL comes from ``params['url']`` when present, else the
    benchmarked server's ``api_url``.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Store the default base URL of the benchmarked server."""
        self.base_url = base_url

    def generate(self, system: str, user_text: str, params: dict) -> str:
        """Return assistant prose from the local OpenAI-compatible server."""
        url = params.get("url") or self.base_url
        return _post_chat_completions(url, system, user_text, params)


class OpenAICompatProvider:
    """Generate the report on an arbitrary OpenAI-compatible server."""

    def __init__(self, url: str, model: str | None = None) -> None:
        """Store the target ``url`` and optional ``model``."""
        self.url = url
        self.model = model

    def generate(self, system: str, user_text: str, params: dict) -> str:
        """Return assistant prose from the configured OpenAI-compatible server."""
        merged = dict(params)
        merged.setdefault("url", self.url)
        if self.model and not merged.get("model"):
            merged["model"] = self.model
        url = merged.get("url") or self.url
        return _post_chat_completions(url, system, user_text, merged)


# ---------------------------------------------------------------------------
# Claude (official anthropic SDK)
# ---------------------------------------------------------------------------

#: Default Claude model.  Must NOT carry a date suffix.
CLAUDE_DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeProvider:
    """Generate the report with the official ``anthropic`` SDK.

    Uses adaptive thinking and streaming; reads ``ANTHROPIC_API_KEY`` from
    the environment.  The ``anthropic`` package is imported lazily so this
    module imports cleanly without it installed.
    """

    def __init__(self, model: str | None = None) -> None:
        """Store the Claude model id (defaults to :data:`CLAUDE_DEFAULT_MODEL`)."""
        self.model = model or CLAUDE_DEFAULT_MODEL

    def generate(self, system: str, user_text: str, params: dict) -> str:
        """Return assistant prose from Claude via the official SDK.

        Uses ``thinking={"type": "adaptive"}`` and the streaming context
        manager with ``get_final_message()``.  Does NOT pass
        ``temperature``/``top_p``/``top_k`` or ``budget_tokens`` — those are
        rejected by this model.  Any typed SDK error (or a missing SDK) is
        normalized to :class:`ProviderError`.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "the 'anthropic' package is not installed; "
                "install the 'report-claude' extra"
            ) from exc

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        model = params.get("model") or self.model
        try:
            with client.messages.stream(
                model=model,
                max_tokens=params.get("max_tokens", 8000),
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": user_text}],
            ) as stream:
                msg = stream.get_final_message()
        except (
            anthropic.RateLimitError,
            anthropic.AuthenticationError,
            anthropic.APIConnectionError,
            anthropic.APIError,
        ) as exc:
            raise ProviderError(f"Claude request failed: {exc}") from exc

        return "".join(b.text for b in msg.content if b.type == "text")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(name: str, params: dict) -> Provider:
    """Build a provider by name.

    Args:
        name: One of ``"local"``, ``"openai"`` or ``"claude"``.
        params: Provider parameters.  ``"url"`` selects the endpoint for the
            local/openai providers; ``"model"`` selects the model for openai
            and claude.

    Returns:
        A :class:`Provider`.

    Raises:
        ProviderError: when ``name`` is not a known provider.
    """
    params = params or {}
    if name == "local":
        return LocalProvider(base_url=params.get("url") or "http://localhost:8000")
    if name == "openai":
        url = params.get("url")
        if not url:
            raise ProviderError("the 'openai' provider requires a 'url' parameter")
        return OpenAICompatProvider(url=url, model=params.get("model"))
    if name == "claude":
        return ClaudeProvider(model=params.get("model"))
    raise ProviderError(f"unknown provider: {name!r} (expected local|openai|claude)")
