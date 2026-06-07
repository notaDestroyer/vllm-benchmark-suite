"""Tests for the KL quality mode and its pure helpers."""

import math

from vllm_benchmark.analysis import quality
from vllm_benchmark.analysis.quality import (
    kl_divergence,
    run_quality,
    top1_agreement,
)

# ---------------------------------------------------------------------------
# kl_divergence (pure)
# ---------------------------------------------------------------------------


class TestKLDivergence:
    def test_identical_is_zero(self):
        p = {"a": math.log(0.5), "b": math.log(0.5)}
        assert math.isclose(kl_divergence(p, p), 0.0, abs_tol=1e-9)

    def test_hand_computed(self):
        # P = {a:0.5, b:0.5}, Q = {a:0.25, b:0.75}.
        # KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
        #    = 0.5*ln2 + 0.5*ln(2/3).
        p = {"a": math.log(0.5), "b": math.log(0.5)}
        q = {"a": math.log(0.25), "b": math.log(0.75)}
        expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        assert math.isclose(kl_divergence(p, q), expected, rel_tol=1e-9)

    def test_nonnegative(self):
        p = {"a": math.log(0.7), "b": math.log(0.3)}
        q = {"a": math.log(0.4), "b": math.log(0.6)}
        assert kl_divergence(p, q) >= 0.0

    def test_empty_p_is_zero(self):
        assert kl_divergence({}, {"a": math.log(1.0)}) == 0.0

    def test_missing_q_token_is_finite(self):
        p = {"a": math.log(0.5), "b": math.log(0.5)}
        q = {"a": math.log(1.0)}  # 'b' missing -> floored, still finite
        kl = kl_divergence(p, q)
        assert math.isfinite(kl)
        assert kl > 0.0


# ---------------------------------------------------------------------------
# top1_agreement (pure)
# ---------------------------------------------------------------------------


class TestTop1Agreement:
    def test_perfect(self):
        assert top1_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_none(self):
        assert top1_agreement(["a", "b"], ["x", "y"]) == 0.0

    def test_partial(self):
        assert top1_agreement(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5

    def test_empty(self):
        assert top1_agreement([], []) == 0.0
        assert top1_agreement(["a"], []) == 0.0

    def test_overlapping_prefix(self):
        # Different lengths -> compare over the shorter prefix.
        assert top1_agreement(["a", "b"], ["a", "b", "c"]) == 1.0


# ---------------------------------------------------------------------------
# Mode gating (mocked)
# ---------------------------------------------------------------------------


class _Cfg:
    api_url = "http://localhost:8000"
    model_name = "test-model"
    request_timeout = 30


class _Info:
    backend = "vllm"
    model_name = "test-model"


def test_missing_ref_url_skipped():
    result = run_quality("kl", _Cfg(), _Info(), ref_url=None)
    assert result["mode"] == "kl"
    assert result["status"] == "skipped"
    assert "ref_url" in result["reason"]


def test_tokenizer_mismatch_detected(monkeypatch):
    """Differing model names between endpoints is flagged as a mismatch."""

    class _RefBackend:
        def server_info(self, cfg):
            class _RefInfo:
                backend = "vllm"
                model_name = "different-model"

            return _RefInfo()

    monkeypatch.setattr(
        "vllm_benchmark.core.backends.detect.detect_backend",
        lambda url, forced=None: _RefBackend(),
    )

    # Both endpoints return usable logprob distributions.
    def fake_post_json(url, body, timeout):
        return {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["x", "y"],
                        "top_logprobs": [
                            {"x": -0.1, "z": -2.0},
                            {"y": -0.2, "w": -2.0},
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(quality, "_post_json", fake_post_json)

    result = run_quality("kl", _Cfg(), _Info(), ref_url="http://localhost:8001")
    assert result["status"] == "ok"
    assert result["tokenizer_mismatch"] is True
    assert math.isfinite(result["mean_kl"])
    assert 0.0 <= result["top1_agreement"] <= 1.0


def test_kl_unsupported_logprobs_skipped(monkeypatch):
    """If an endpoint returns no usable logprobs, KL is skipped."""

    class _RefBackend:
        def server_info(self, cfg):
            return _Info()

    monkeypatch.setattr(
        "vllm_benchmark.core.backends.detect.detect_backend",
        lambda url, forced=None: _RefBackend(),
    )
    monkeypatch.setattr(quality, "_post_json", lambda url, body, timeout: {"choices": [{}]})

    result = run_quality("kl", _Cfg(), _Info(), ref_url="http://localhost:8001")
    assert result["status"] == "skipped"
