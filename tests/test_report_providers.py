"""Tests for the analyst-report providers (fully offline).

Every provider is exercised with mocks — no network, no API key.  The
Claude path is asserted to use the exact official-SDK call shape: model
``claude-opus-4-8``, ``thinking={"type": "adaptive"}``, the streaming
context manager + ``get_final_message()``, and NO temperature / budget_tokens.
Typed SDK exceptions map to :class:`ProviderError`.
"""

from __future__ import annotations

import sys
import types

import pytest

from vllm_benchmark.analysis.report.providers import (
    CLAUDE_DEFAULT_MODEL,
    ClaudeProvider,
    LocalProvider,
    OpenAICompatProvider,
    ProviderError,
    get_provider,
)

# ---------------------------------------------------------------------------
# Local / OpenAI-compatible (HTTP mocked)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {
            "choices": [{"message": {"content": "REPORT BODY"}}]
        }

    def json(self):
        return self._payload


def test_local_provider_posts_chat_completions(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr("vllm_benchmark.analysis.report.providers.requests.post", fake_post)

    provider = LocalProvider(base_url="http://localhost:8000")
    out = provider.generate("SYS", "USER", {"seed": 7})

    assert out == "REPORT BODY"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["json"]["temperature"] == 0
    assert captured["json"]["seed"] == 7
    assert captured["json"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "USER"}


def test_openai_provider_uses_configured_url_and_model(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr("vllm_benchmark.analysis.report.providers.requests.post", fake_post)

    provider = OpenAICompatProvider(url="http://remote:9000", model="gpt-oss")
    provider.generate("SYS", "USER", {})

    assert captured["url"] == "http://remote:9000/v1/chat/completions"
    assert captured["json"]["model"] == "gpt-oss"
    assert captured["json"]["temperature"] == 0


def test_local_provider_unreachable_raises_provider_error(monkeypatch):
    import requests as real_requests

    def fake_post(*a, **kw):
        raise real_requests.ConnectionError("refused")

    monkeypatch.setattr("vllm_benchmark.analysis.report.providers.requests.post", fake_post)

    with pytest.raises(ProviderError):
        LocalProvider().generate("SYS", "USER", {})


def test_local_provider_non_200_raises(monkeypatch):
    monkeypatch.setattr(
        "vllm_benchmark.analysis.report.providers.requests.post",
        lambda *a, **kw: _FakeResponse(status_code=500),
    )
    with pytest.raises(ProviderError):
        LocalProvider().generate("SYS", "USER", {})


def test_local_provider_malformed_response_raises(monkeypatch):
    monkeypatch.setattr(
        "vllm_benchmark.analysis.report.providers.requests.post",
        lambda *a, **kw: _FakeResponse(payload={"unexpected": True}),
    )
    with pytest.raises(ProviderError):
        LocalProvider().generate("SYS", "USER", {})


# ---------------------------------------------------------------------------
# Claude (official SDK mocked)
# ---------------------------------------------------------------------------

class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ThinkingBlock:
    type = "thinking"

    def __init__(self, thinking):
        self.thinking = thinking


class _FakeMessage:
    def __init__(self, blocks):
        self.content = blocks


class _FakeStream:
    def __init__(self, message, kwargs_sink):
        self._message = message
        self._kwargs_sink = kwargs_sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


def _install_fake_anthropic(monkeypatch, *, blocks=None, raise_exc_name=None):
    """Install a fake ``anthropic`` module; return a sink capturing call kwargs.

    When ``raise_exc_name`` is given, ``messages.stream`` raises that named
    exception class *from the same fake module*, so the provider's typed
    except-clauses (which import the same module) catch it.
    """
    sink = {}

    fake = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class RateLimitError(APIError):
        pass

    class AuthenticationError(APIError):
        pass

    class APIConnectionError(APIError):
        pass

    fake.APIError = APIError
    fake.RateLimitError = RateLimitError
    fake.AuthenticationError = AuthenticationError
    fake.APIConnectionError = APIConnectionError

    class _FakeMessages:
        def stream(self, **kwargs):
            sink["stream_kwargs"] = kwargs
            if raise_exc_name is not None:
                raise getattr(fake, raise_exc_name)("boom")
            msg = _FakeMessage(blocks if blocks is not None else [_TextBlock("CLAUDE REPORT")])
            return _FakeStream(msg, sink)

    class _FakeClient:
        def __init__(self, *a, **kw):
            sink["client_init"] = (a, kw)
            self.messages = _FakeMessages()

    fake.Anthropic = _FakeClient

    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return sink, fake


def test_claude_uses_exact_sdk_shape(monkeypatch):
    sink, _ = _install_fake_anthropic(
        monkeypatch, blocks=[_ThinkingBlock("hmm"), _TextBlock("CLAUDE REPORT")]
    )

    out = ClaudeProvider().generate("SYS", "USER", {})

    assert out == "CLAUDE REPORT"  # only text blocks concatenated
    kwargs = sink["stream_kwargs"]
    assert kwargs["model"] == CLAUDE_DEFAULT_MODEL == "claude-opus-4-8"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["system"] == "SYS"
    assert kwargs["messages"] == [{"role": "user", "content": "USER"}]
    # Forbidden params must be absent.
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "budget_tokens" not in kwargs


def test_claude_default_model_has_no_date_suffix(monkeypatch):
    sink, _ = _install_fake_anthropic(monkeypatch)
    ClaudeProvider().generate("SYS", "USER", {})
    assert sink["stream_kwargs"]["model"] == "claude-opus-4-8"


def test_claude_respects_model_override(monkeypatch):
    sink, _ = _install_fake_anthropic(monkeypatch)
    ClaudeProvider().generate("SYS", "USER", {"model": "claude-sonnet-4-6"})
    assert sink["stream_kwargs"]["model"] == "claude-sonnet-4-6"


@pytest.mark.parametrize("exc_name", [
    "RateLimitError", "AuthenticationError", "APIConnectionError", "APIError",
])
def test_claude_typed_exceptions_become_provider_error(monkeypatch, exc_name):
    _install_fake_anthropic(monkeypatch, raise_exc_name=exc_name)
    with pytest.raises(ProviderError):
        ClaudeProvider().generate("SYS", "USER", {})


def test_claude_missing_sdk_raises_provider_error(monkeypatch):
    # Simulate the import failing.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ProviderError):
        ClaudeProvider().generate("SYS", "USER", {})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_provider_local():
    assert isinstance(get_provider("local", {"url": "http://x:8000"}), LocalProvider)


def test_get_provider_openai_requires_url():
    assert isinstance(get_provider("openai", {"url": "http://x"}), OpenAICompatProvider)
    with pytest.raises(ProviderError):
        get_provider("openai", {})


def test_get_provider_claude():
    assert isinstance(get_provider("claude", {}), ClaudeProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(ProviderError):
        get_provider("bogus", {})
