"""Tests for the perplexity quality mode and its pure helper."""

import math

from vllm_benchmark.analysis import quality
from vllm_benchmark.analysis.quality import perplexity_from_logprobs, run_quality

# ---------------------------------------------------------------------------
# perplexity_from_logprobs (pure)
# ---------------------------------------------------------------------------


class TestPerplexityFromLogprobs:
    def test_hand_computed(self):
        # logprobs = [-1, -2, -3]; mean = -2; PPL = exp(2).
        ppl = perplexity_from_logprobs([-1.0, -2.0, -3.0])
        assert math.isclose(ppl, math.exp(2.0), rel_tol=1e-9)

    def test_uniform_logprobs(self):
        # All logprobs -ln(2) -> PPL = 2.
        lp = -math.log(2.0)
        ppl = perplexity_from_logprobs([lp, lp, lp, lp])
        assert math.isclose(ppl, 2.0, rel_tol=1e-9)

    def test_none_entries_ignored(self):
        # Leading None (first token) is dropped; rest -> exp(1).
        ppl = perplexity_from_logprobs([None, -1.0, -1.0])
        assert math.isclose(ppl, math.exp(1.0), rel_tol=1e-9)

    def test_empty_is_inf(self):
        assert perplexity_from_logprobs([]) == float("inf")
        assert perplexity_from_logprobs([None]) == float("inf")


# ---------------------------------------------------------------------------
# Backend gating (mocked)
# ---------------------------------------------------------------------------


class _Cfg:
    api_url = "http://localhost:8000"
    model_name = "test-model"
    request_timeout = 30


class _VLLMInfo:
    backend = "vllm"
    model_name = "test-model"


class _SGLangInfo:
    backend = "sglang"
    model_name = "test-model"


def test_unsupported_backend_skipped():
    """Non-vLLM backend cannot give prompt logprobs -> skipped with reason."""
    result = run_quality("perplexity", _Cfg(), _SGLangInfo())
    assert result["mode"] == "perplexity"
    assert result["status"] == "skipped"
    assert "reason" in result and result["reason"]


def test_endpoint_without_logprobs_skipped(monkeypatch):
    """vLLM backend but endpoint returns no usable logprobs -> skipped."""

    def fake_post_json(url, body, timeout):
        # No prompt_logprobs / logprobs in the response.
        return {"choices": [{"text": "", "logprobs": None}]}

    monkeypatch.setattr(quality, "_post_json", fake_post_json)
    result = run_quality("perplexity", _Cfg(), _VLLMInfo())
    assert result["status"] == "skipped"
    assert "logprobs" in result["reason"].lower()


def test_supported_path_computes_ppl(monkeypatch):
    """vLLM-style prompt_logprobs are parsed and a finite PPL is returned."""

    def fake_post_json(url, body, timeout):
        # vLLM prompt_logprobs: first entry None, then token->{"logprob":...}.
        return {
            "choices": [
                {
                    "prompt_logprobs": [
                        None,
                        {"123": {"logprob": -1.0}},
                        {"456": {"logprob": -1.0}},
                    ]
                }
            ]
        }

    monkeypatch.setattr(quality, "_post_json", fake_post_json)
    result = run_quality("perplexity", _Cfg(), _VLLMInfo())
    assert result["status"] == "ok"
    assert math.isfinite(result["perplexity"])
    # Every token logprob is -1 -> PPL = exp(1).
    assert math.isclose(result["perplexity"], math.exp(1.0), rel_tol=1e-9)
    assert result["n_tokens"] > 0
