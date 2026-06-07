"""Tests for prefill/decode (pp/tg) split derivation.

Feeds synthetic SSE chunk sequences to the pure streaming parser and
asserts ttft, decode_tps, inter-token latency, and pp_tg_source labels.
Also verifies the non-streaming path reports an ``unavailable`` source
and omits the split, and that ``_compute_stats`` aggregates the fields.

Author: amit
"""

from vllm_benchmark.core.async_engine import _compute_stats, parse_streaming_chunks


def _content(text: str) -> str:
    return f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}'


def _usage(prompt: int, completion: int) -> str:
    return (f'data: {{"choices":[{{"delta":{{}}}}],'
            f'"usage":{{"prompt_tokens":{prompt},"completion_tokens":{completion}}}}}')


# ---------------------------------------------------------------------------
# Streaming parser — multiple chunks
# ---------------------------------------------------------------------------


def test_multiple_chunks_with_usage():
    """N content chunks + final usage: ttft, decode_tps, itl all derived."""
    events = [
        (1.0, _content("a")),   # first content @ 1.0  (t_send=0.8 -> ttft 0.2)
        (1.1, _content("b")),
        (1.2, _content("c")),
        (1.3, _content("d")),   # last content @ 1.3
        (1.35, _usage(100, 4)),
        (1.4, "data: [DONE]"),
    ]
    r = parse_streaming_chunks(events, t_send=0.8)

    assert r["pp_tg_source"] == "client_stream"
    assert abs(r["ttft"] - 0.2) < 1e-9
    assert r["prompt_tokens"] == 100
    assert r["completion_tokens"] == 4
    # decode window = 1.3 - 1.0 = 0.3; (completion-1)/window = 3 / 0.3 = 10
    assert abs(r["decode_tps"] - 10.0) < 1e-6
    # prefill_tps = prompt_tokens / ttft = 100 / 0.2 = 500
    assert abs(r["prefill_tps"] - 500.0) < 1e-6
    # ITL = mean of gaps (0.1, 0.1, 0.1) = 0.1
    assert abs(r["inter_token_latency"] - 0.1) < 1e-9


def test_single_content_chunk_no_decode_tps():
    """A single content chunk cannot define a decode interval."""
    events = [
        (2.0, _content("only")),
        (2.05, _usage(50, 1)),
        (2.1, "data: [DONE]"),
    ]
    r = parse_streaming_chunks(events, t_send=1.9)

    assert r["pp_tg_source"] == "client_stream"
    assert abs(r["ttft"] - 0.1) < 1e-9
    assert r["decode_tps"] is None  # < 2 content chunks
    assert r["inter_token_latency"] is None  # < 2 content chunks
    # prefill still derivable (prompt_tokens / ttft)
    assert r["prefill_tps"] is not None


def test_no_usage_chunk_falls_back_to_chunk_count():
    """Without a usage chunk, completion_tokens == number of content chunks."""
    events = [
        (1.0, _content("a")),
        (1.2, _content("b")),
        (1.4, _content("c")),
    ]
    r = parse_streaming_chunks(events, t_send=0.9)

    assert r["completion_tokens"] == 3
    assert r["prompt_tokens"] == 0
    # No prompt tokens -> prefill_tps cannot be computed.
    assert r["prefill_tps"] is None
    # decode window = 1.4 - 1.0 = 0.4; (3-1)/0.4 = 5
    assert abs(r["decode_tps"] - 5.0) < 1e-6


def test_time_gaps_reflected_in_decode_tps():
    """Larger gaps between chunks reduce decode throughput."""
    events = [
        (0.0, _content("a")),
        (1.0, _content("b")),
        (2.0, _content("c")),
        (2.0, _usage(10, 3)),
    ]
    r = parse_streaming_chunks(events, t_send=-0.1)
    # decode window = 2.0 - 0.0 = 2.0; (3-1)/2.0 = 1.0
    assert abs(r["decode_tps"] - 1.0) < 1e-6


def test_empty_stream():
    """No events at all yields zeros / None without raising."""
    r = parse_streaming_chunks([], t_send=0.0)
    assert r["completion_tokens"] == 0
    assert r["ttft"] is None
    assert r["decode_tps"] is None
    assert r["prefill_tps"] is None
    assert r["pp_tg_source"] == "client_stream"


# ---------------------------------------------------------------------------
# _compute_stats aggregation
# ---------------------------------------------------------------------------


def _streaming_result(rid: int, prefill: float, decode: float) -> dict:
    return {
        "request_id": rid,
        "duration": 1.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "inter_token_latency": 0.02,
        "ttft": 0.1,
        "prefill_tps": prefill,
        "decode_tps": decode,
        "pp_tg_source": "client_stream",
        "success": True,
        "streaming": True,
    }


def test_compute_stats_aggregates_pp_tg():
    results = [_streaming_result(i, 400.0 + i, 30.0 + i) for i in range(4)]
    stats = _compute_stats(
        results=results, total_time=2.0, context_length=32000,
        num_concurrent_users=4, prompt_type="classic", actual_prompt_tokens=100,
        gpu_stats=None, metrics_stats=None, cost_per_hour=None,
    )
    assert stats is not None
    assert stats["pp_tg_source"] == "client_stream"
    assert "prefill_tps_mean" in stats
    assert "prefill_tps_p50" in stats
    assert "prefill_tps_p90" in stats
    assert "prefill_tps_p99" in stats
    assert "decode_tps_mean" in stats
    assert "decode_tps_p50" in stats


def test_compute_stats_nonstreaming_unavailable_and_omits_split():
    """Non-streaming results set unavailable and omit the pp/tg numbers."""
    ns = {
        "request_id": 0,
        "duration": 1.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "inter_token_latency": 0.02,
        "ttft": None,
        "prefill_tps": None,
        "decode_tps": None,
        "pp_tg_source": "unavailable",
        "success": True,
        "streaming": False,
    }
    stats = _compute_stats(
        results=[ns], total_time=1.0, context_length=32000,
        num_concurrent_users=1, prompt_type="classic", actual_prompt_tokens=100,
        gpu_stats=None, metrics_stats=None, cost_per_hour=None,
    )
    assert stats is not None
    assert stats["pp_tg_source"] == "unavailable"
    assert "prefill_tps_mean" not in stats
    assert "decode_tps_mean" not in stats
