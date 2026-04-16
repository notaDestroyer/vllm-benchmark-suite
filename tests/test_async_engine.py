"""Tests for the async benchmark engine helpers.

Covers token counting, percentile calculation, and stats aggregation
without requiring a running server or GPU.
"""

from unittest.mock import MagicMock, patch

from vllm_benchmark.core.async_engine import (
    _compute_stats,
    calculate_percentiles,
    count_tokens,
)

# ---------------------------------------------------------------------------
# calculate_percentiles
# ---------------------------------------------------------------------------


def test_percentiles_known_values():
    """Percentiles of a simple sorted range should land on known points."""
    values = list(range(1, 101))  # 1..100
    p = calculate_percentiles(values)
    assert abs(p["p50"] - 50.5) < 1.0
    assert abs(p["p90"] - 90.1) < 1.0
    assert p["p95"] > p["p90"]
    assert p["p99"] > p["p95"]


def test_percentiles_single_value():
    p = calculate_percentiles([42.0])
    assert p["p50"] == 42.0
    assert p["p99"] == 42.0


def test_percentiles_empty():
    p = calculate_percentiles([])
    assert p == {"p50": 0, "p90": 0, "p95": 0, "p99": 0}


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_fallback():
    """When no tokenizer can be loaded, falls back to len(text)//4."""
    # Clear any cached tokenizers to force the fallback path
    from vllm_benchmark.core import async_engine
    saved = dict(async_engine._tokenizer_cache)
    async_engine._tokenizer_cache.clear()

    try:
        with patch.dict("sys.modules", {"transformers": None}):
            # Force import failure
            with patch("builtins.__import__", side_effect=ImportError):
                result = count_tokens("hello world!", model_name="__test_missing__")
    finally:
        async_engine._tokenizer_cache.clear()
        async_engine._tokenizer_cache.update(saved)

    assert isinstance(result, int)
    assert result == len("hello world!") // 4


def test_count_tokens_with_mock_tokenizer():
    """When the tokenizer works, returns its token count."""
    from vllm_benchmark.core import async_engine
    saved = dict(async_engine._tokenizer_cache)

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    async_engine._tokenizer_cache["__mock__"] = fake_tokenizer

    try:
        result = count_tokens("anything", model_name="__mock__")
    finally:
        async_engine._tokenizer_cache.clear()
        async_engine._tokenizer_cache.update(saved)

    assert result == 5


def test_count_tokens_returns_positive_int():
    """Regardless of path, result should be int > 0 for non-empty text."""
    from vllm_benchmark.core import async_engine
    saved = dict(async_engine._tokenizer_cache)

    # Force the fallback path with a None tokenizer
    async_engine._tokenizer_cache["__pos_test__"] = None
    try:
        result = count_tokens("The quick brown fox jumps.", model_name="__pos_test__")
    finally:
        async_engine._tokenizer_cache.clear()
        async_engine._tokenizer_cache.update(saved)

    assert isinstance(result, int)
    assert result > 0


# ---------------------------------------------------------------------------
# _compute_stats
# ---------------------------------------------------------------------------


def _make_request_result(**overrides):
    """Build a mock individual-request result dict."""
    base = {
        "request_id": 0,
        "duration": 1.5,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "inter_token_latency": 0.03,
        "ttft": 0.12,
        "prefill_time_estimate": 0.12,
        "decode_time_estimate": 1.38,
        "success": True,
        "streaming": True,
    }
    base.update(overrides)
    return base


def test_compute_stats_all_expected_keys():
    """Verify that the basic set of keys is always present."""
    results = [_make_request_result(request_id=i) for i in range(4)]
    stats = _compute_stats(
        results=results,
        total_time=2.0,
        context_length=32000,
        num_concurrent_users=4,
        prompt_type="classic",
        actual_prompt_tokens=100,
        gpu_stats=None,
        metrics_stats=None,
        cost_per_hour=None,
    )
    assert stats is not None
    expected_keys = {
        "context_length",
        "concurrent_users",
        "prompt_type",
        "total_time",
        "successful",
        "failed",
        "total_requests",
        "avg_latency",
        "std_latency",
        "min_latency",
        "max_latency",
        "latency_p50",
        "latency_p90",
        "latency_p95",
        "latency_p99",
        "ttft_estimate",
        "ttft_p50",
        "ttft_p90",
        "ttft_p95",
        "ttft_p99",
        "tokens_per_second",
        "requests_per_second",
        "throughput_per_user",
        "avg_prompt_tokens",
        "avg_completion_tokens",
    }
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"


def test_compute_stats_returns_none_on_all_failures():
    results = [_make_request_result(success=False)]
    stats = _compute_stats(
        results=results,
        total_time=1.0,
        context_length=32000,
        num_concurrent_users=1,
        prompt_type="classic",
        actual_prompt_tokens=100,
        gpu_stats=None,
        metrics_stats=None,
        cost_per_hour=None,
    )
    assert stats is None


def test_compute_stats_cost_fields_present_when_cost_provided():
    results = [_make_request_result(request_id=i) for i in range(4)]
    stats = _compute_stats(
        results=results,
        total_time=2.0,
        context_length=32000,
        num_concurrent_users=4,
        prompt_type="classic",
        actual_prompt_tokens=100,
        gpu_stats=None,
        metrics_stats=None,
        cost_per_hour=2.50,
    )
    assert stats is not None
    assert "cost_per_hour" in stats
    assert "cost_per_1m_tokens" in stats
    assert "cost_total" in stats
    assert stats["cost_per_hour"] == 2.50
    assert stats["cost_total"] > 0


def test_compute_stats_no_cost_fields_without_cost():
    results = [_make_request_result()]
    stats = _compute_stats(
        results=results,
        total_time=1.0,
        context_length=32000,
        num_concurrent_users=1,
        prompt_type="classic",
        actual_prompt_tokens=100,
        gpu_stats=None,
        metrics_stats=None,
        cost_per_hour=None,
    )
    assert stats is not None
    assert "cost_per_hour" not in stats
    assert "cost_per_1m_tokens" not in stats
    assert "cost_total" not in stats


def test_compute_stats_with_gpu_stats():
    results = [_make_request_result()]
    gpu_stats = {
        "avg_gpu_util": 85.0,
        "avg_power": 280.0,
        "avg_temperature": 65.0,
    }
    stats = _compute_stats(
        results=results,
        total_time=1.0,
        context_length=32000,
        num_concurrent_users=1,
        prompt_type="classic",
        actual_prompt_tokens=100,
        gpu_stats=gpu_stats,
        metrics_stats=None,
        cost_per_hour=None,
    )
    assert stats is not None
    assert "avg_gpu_util" in stats
    assert "watts_per_token" in stats
    assert "tokens_per_watt" in stats
    assert "energy_joules" in stats
    assert "energy_per_token" in stats


def test_compute_stats_itl_percentiles():
    """When requests have inter_token_latency, ITL percentile keys appear."""
    results = [_make_request_result(request_id=i, inter_token_latency=0.02 + i * 0.01)
               for i in range(10)]
    stats = _compute_stats(
        results=results,
        total_time=3.0,
        context_length=32000,
        num_concurrent_users=10,
        prompt_type="classic",
        actual_prompt_tokens=100,
        gpu_stats=None,
        metrics_stats=None,
        cost_per_hour=None,
    )
    assert stats is not None
    assert "itl_p50" in stats
    assert "itl_p99" in stats
