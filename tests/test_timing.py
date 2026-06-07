"""Tests proving duration measurements use a monotonic clock.

The streaming parser is a pure function that consumes ``perf_counter``
timestamps, so we can assert that a backwards wall-clock jump
(``time.time``) cannot corrupt a measured duration.

Author: amit
"""

import time

from vllm_benchmark.core import async_engine
from vllm_benchmark.core.async_engine import parse_streaming_chunks


def _sse(obj_json: str) -> str:
    return f"data: {obj_json}"


def test_duration_uses_perf_counter_not_wall_clock():
    """A duration derived from perf_counter timestamps is wall-clock immune."""
    # Two content chunks at monotonic times 100.0 and 100.5; send at 99.8.
    events = [
        (100.0, _sse('{"choices":[{"delta":{"content":"a"}}]}')),
        (100.5, _sse('{"choices":[{"delta":{"content":"b"}}],'
                     '"usage":{"prompt_tokens":10,"completion_tokens":2}}')),
        (100.6, "data: [DONE]"),
    ]
    result = parse_streaming_chunks(events, t_send=99.8, request_id=1)

    # duration = last_event(100.6) - t_send(99.8) == 0.8, regardless of time.time().
    assert abs(result["duration"] - 0.8) < 1e-9
    # ttft = first_content(100.0) - t_send(99.8) == 0.2
    assert abs(result["ttft"] - 0.2) < 1e-9


def test_backwards_wall_clock_jump_does_not_corrupt_duration():
    """Even if time.time() runs backwards, perf_counter-based math holds."""
    original_time = time.time

    # Simulate NTP stepping the wall clock backwards mid-request.
    fake_values = iter([1000.0, 5.0, 5.0])

    def jumpy_time():
        try:
            return next(fake_values)
        except StopIteration:
            return original_time()

    events = [
        (50.0, _sse('{"choices":[{"delta":{"content":"x"}}]}')),
        (50.4, _sse('{"choices":[{"delta":{"content":"y"}}],'
                    '"usage":{"prompt_tokens":4,"completion_tokens":2}}')),
    ]

    # parse_streaming_chunks never calls time.time(); patch it to prove it.
    time.time = jumpy_time
    try:
        result = parse_streaming_chunks(events, t_send=49.9, request_id=0)
    finally:
        time.time = original_time

    # Duration stays positive and correct (50.4 - 49.9 == 0.5).
    assert result["duration"] > 0
    assert abs(result["duration"] - 0.5) < 1e-9


def test_async_engine_module_uses_perf_counter():
    """The async request executors must reference time.perf_counter."""
    import inspect

    src = inspect.getsource(async_engine._async_streaming_request)
    assert "time.perf_counter()" in src
    assert "time.time()" not in src

    src_batch = inspect.getsource(async_engine._async_batch_request)
    assert "time.perf_counter()" in src_batch
    assert "time.time()" not in src_batch


def test_perf_counter_helper_is_monotonic():
    """Sanity: perf_counter never goes backwards across two reads."""
    a = time.perf_counter()
    b = time.perf_counter()
    assert b >= a
