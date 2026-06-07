"""Async benchmark engine using aiohttp.

Replaces threading with true async concurrency for accurate measurements.
Supports both burst mode (all-at-once) and sustained RPS mode.

Author: amit
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import time
from statistics import mean, stdev
from typing import Dict, List, Optional

import aiohttp
import numpy as np
from rich.console import Console

from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core.metrics import GPUMonitor, MetricsMonitor
from vllm_benchmark.core.prompts import get_prompt_generator

console = Console()


# ------------------------------------------------------------------
# Percentile helpers
# ------------------------------------------------------------------

def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# ------------------------------------------------------------------
# Streaming SSE parsing (pure, unit-testable)
# ------------------------------------------------------------------

def parse_streaming_chunks(
    events: List["tuple[float, str]"],
    t_send: float,
    request_id: int = 0,
) -> Dict:
    """Parse a sequence of timed SSE lines into request metrics.

    This is a pure function — it takes already-received SSE lines paired
    with the monotonic ``time.perf_counter()`` timestamp at which each
    line arrived, and computes timing/throughput metrics without any
    network I/O.  It is the single source of truth for streaming
    prefill/decode (pp/tg) accounting and is unit-tested directly.

    Args:
        events: List of ``(perf_counter_time, raw_line)`` tuples in
            arrival order.  ``raw_line`` is the decoded, stripped SSE
            line (e.g. ``"data: {...}"`` or ``"data: [DONE]"``).
        t_send: ``time.perf_counter()`` captured immediately before the
            request was sent.
        request_id: The request identifier to embed in the result.

    Returns:
        A result dict with timing fields.  ``prefill_tps`` / ``decode_tps``
        are ``None`` when they cannot be derived; ``pp_tg_source`` is
        ``"client_stream"``.
    """
    first_content_time: Optional[float] = None
    last_content_time: Optional[float] = None
    content_times: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0

    for now, line in events:
        if not line or not line.startswith("data: "):
            continue

        payload = line[6:]
        if payload.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content")

        if content is not None:
            content_times.append(now)
            if first_content_time is None:
                first_content_time = now
            last_content_time = now

    # Duration is measured from send to the last observed event (or send
    # time if nothing arrived).
    last_event_time = events[-1][0] if events else t_send
    duration = last_event_time - t_send

    # TTFT — time to first content chunk.
    ttft = (first_content_time - t_send) if first_content_time is not None else None

    # Inter-token latency over content chunks.
    if len(content_times) >= 2:
        inter_times = [content_times[i] - content_times[i - 1] for i in range(1, len(content_times))]
        avg_itl: Optional[float] = mean(inter_times)
    else:
        avg_itl = None

    if completion_tokens == 0:
        completion_tokens = len(content_times)

    # Prefill throughput: prompt tokens consumed during the prefill phase
    # (== time to first token).
    if ttft is not None and ttft > 0 and prompt_tokens > 0:
        prefill_tps: Optional[float] = prompt_tokens / ttft
    else:
        prefill_tps = None

    # Decode throughput: completion tokens emitted after the first content
    # chunk, requires at least two content chunks to define an interval.
    if (
        first_content_time is not None
        and last_content_time is not None
        and len(content_times) >= 2
        and completion_tokens >= 2
    ):
        decode_window = last_content_time - first_content_time
        decode_tps: Optional[float] = (completion_tokens - 1) / decode_window if decode_window > 0 else None
    else:
        decode_tps = None

    return {
        "request_id": request_id,
        "duration": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "inter_token_latency": avg_itl,
        "ttft": ttft,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "pp_tg_source": "client_stream",
        "success": True,
        "streaming": True,
    }


# ------------------------------------------------------------------
# Async request executors
# ------------------------------------------------------------------

async def _async_streaming_request(
    session: aiohttp.ClientSession,
    prompt: str,
    request_id: int,
    max_tokens: int,
    model_name: str,
    api_endpoint: str,
    request_timeout: int,
) -> Dict:
    """Async streaming request measuring true TTFT via SSE chunks.

    Collects each SSE line together with the monotonic
    ``time.perf_counter()`` at which it arrived, then delegates all metric
    derivation to :func:`parse_streaming_chunks`.
    """
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    try:
        start = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with session.post(api_endpoint, json=data, timeout=timeout) as response:
            if response.status != 200:
                return {
                    "request_id": request_id,
                    "duration": time.perf_counter() - start,
                    "success": False,
                    "error": f"HTTP {response.status}",
                }

            events: list[tuple[float, str]] = []
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                events.append((time.perf_counter(), line))

            return parse_streaming_chunks(events, start, request_id)

    except Exception as e:
        return {"request_id": request_id, "success": False, "error": str(e)}


async def _async_batch_request(
    session: aiohttp.ClientSession,
    prompt: str,
    request_id: int,
    max_tokens: int,
    model_name: str,
    api_endpoint: str,
    request_timeout: int,
) -> Dict:
    """Async non-streaming request."""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        start = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with session.post(api_endpoint, json=data, timeout=timeout) as response:
            duration = time.perf_counter() - start

            if response.status == 200:
                result = await response.json()
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                itl = duration / completion_tokens if completion_tokens > 0 else None

                # Non-streaming requests cannot separate prefill from decode;
                # do NOT fabricate a split.
                return {
                    "request_id": request_id,
                    "duration": duration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens", 0),
                    "inter_token_latency": itl,
                    "ttft": None,
                    "prefill_tps": None,
                    "decode_tps": None,
                    "pp_tg_source": "unavailable",
                    "success": True,
                    "streaming": False,
                }
            else:
                return {
                    "request_id": request_id,
                    "duration": duration,
                    "success": False,
                    "error": f"HTTP {response.status}",
                }
    except Exception as e:
        return {"request_id": request_id, "success": False, "error": str(e)}


async def _make_async_request(
    session: aiohttp.ClientSession,
    prompt: str,
    request_id: int,
    max_tokens: int,
    model_name: str,
    api_endpoint: str,
    request_timeout: int,
    streaming: bool,
) -> Dict:
    if streaming:
        return await _async_streaming_request(
            session, prompt, request_id, max_tokens,
            model_name, api_endpoint, request_timeout,
        )
    return await _async_batch_request(
        session, prompt, request_id, max_tokens,
        model_name, api_endpoint, request_timeout,
    )


# ------------------------------------------------------------------
# Burst mode (replaces old threading approach)
# ------------------------------------------------------------------

async def _warmup_connection(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    model_name: str,
) -> None:
    """Issue a single throwaway request to establish the connection pool.

    Guarded so that any failure is ignored; the warmup request is never
    counted in the measured results.
    """
    try:
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=config.request_timeout)
        async with session.post(config.api_endpoint, json=data, timeout=timeout) as response:
            await response.read()
    except Exception:
        pass


async def _run_burst(
    prompt: str,
    num_concurrent: int,
    config: BenchmarkConfig,
    model_name: str,
) -> List[Dict]:
    """Fire all requests simultaneously and collect results."""
    connector = aiohttp.TCPConnector(limit=num_concurrent + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Connection warmup — establish the pool before the measured burst.
        await _warmup_connection(session, config, model_name)
        tasks = [
            _make_async_request(
                session, prompt, i, config.output_tokens, model_name,
                config.api_endpoint, config.request_timeout, config.streaming,
            )
            for i in range(num_concurrent)
        ]
        return await asyncio.gather(*tasks)


# ------------------------------------------------------------------
# Sustained RPS mode
# ------------------------------------------------------------------

async def _run_sustained_rps(
    prompt: str,
    target_rps: float,
    duration_seconds: float,
    config: BenchmarkConfig,
    model_name: str,
) -> List[Dict]:
    """Sustain a target request rate for a given duration.

    Launches requests at the specified RPS using a token-bucket style
    scheduler. Tracks per-request timing and returns all results.
    """
    results: list[Dict] = []
    interval = 1.0 / target_rps
    request_id = 0
    pending: list[asyncio.Task] = []

    connector = aiohttp.TCPConnector(limit=int(target_rps * 10) + 50)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Connection warmup — establish the pool before the measured run.
        await _warmup_connection(session, config, model_name)

        start = time.perf_counter()
        next_send = start

        while time.perf_counter() - start < duration_seconds:
            now = time.perf_counter()
            if now >= next_send:
                task = asyncio.create_task(
                    _make_async_request(
                        session, prompt, request_id, config.output_tokens,
                        model_name, config.api_endpoint, config.request_timeout,
                        config.streaming,
                    )
                )
                pending.append(task)
                request_id += 1
                next_send += interval
            else:
                await asyncio.sleep(min(next_send - now, 0.01))

        # Wait for all in-flight requests to complete
        if pending:
            completed = await asyncio.gather(*pending, return_exceptions=True)
            for r in completed:
                if isinstance(r, dict):
                    results.append(r)
                else:
                    results.append({"request_id": -1, "success": False, "error": str(r)})

    return results


# ------------------------------------------------------------------
# Tokenizer-aware prompt token counting
# ------------------------------------------------------------------

_tokenizer_cache: dict = {}


def count_tokens(text: str, model_name: str = "") -> int:
    """Count tokens accurately using the model's tokenizer.

    Falls back to len(text)//4 if the tokenizer can't be loaded.
    """
    if model_name not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            try:
                from vllm_benchmark.config import DEFAULT_TOKENIZER
                _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
            except Exception:
                _tokenizer_cache[model_name] = None

    tokenizer = _tokenizer_cache[model_name]
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text) // 4


# ------------------------------------------------------------------
# Main async benchmark runner
# ------------------------------------------------------------------

def _compute_stats(
    results: List[Dict],
    total_time: float,
    context_length: int,
    num_concurrent_users: int,
    prompt_type: str,
    actual_prompt_tokens: int,
    gpu_stats: Optional[Dict],
    metrics_stats: Optional[Dict],
    cost_per_hour: Optional[float],
) -> Optional[Dict]:
    """Compute aggregate statistics from individual request results."""
    successful = [r for r in results if r.get("success", False)]
    failed = len(results) - len(successful)

    if not successful:
        return None

    durations = [r["duration"] for r in successful]
    completion_tokens_list = [r["completion_tokens"] for r in successful]
    prompt_tokens_list = [r["prompt_tokens"] for r in successful]

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0

    total_completion_tokens = sum(completion_tokens_list)
    avg_prompt_tokens = mean(prompt_tokens_list) if prompt_tokens_list else actual_prompt_tokens

    tokens_per_second = total_completion_tokens / total_time if total_time > 0 else 0
    requests_per_second = len(successful) / total_time if total_time > 0 else 0
    avg_tokens_per_request = mean(completion_tokens_list) if completion_tokens_list else 0

    ttft_values = [r.get("ttft") for r in successful if r.get("ttft") is not None]
    ttft_estimate = mean(ttft_values) if ttft_values else avg_duration * 0.1

    # TTFT percentiles
    ttft_percentiles = calculate_percentiles(ttft_values) if ttft_values else {}

    # Latency percentiles
    latency_percentiles = calculate_percentiles(durations)

    # ITL percentiles
    itl_values = [r.get("inter_token_latency") for r in successful if r.get("inter_token_latency") is not None]
    itl_percentiles = calculate_percentiles(itl_values) if itl_values else {}

    throughput_per_user = tokens_per_second / num_concurrent_users if num_concurrent_users > 0 else 0

    result_dict: Dict = {
        "context_length": context_length,
        "concurrent_users": num_concurrent_users,
        "prompt_type": prompt_type,
        "total_time": total_time,
        "successful": len(successful),
        "failed": failed,
        "total_requests": len(results),
        "avg_latency": avg_duration,
        "std_latency": std_duration,
        "min_latency": min(durations),
        "max_latency": max(durations),
        "latency_p50": latency_percentiles.get("p50", 0),
        "latency_p90": latency_percentiles.get("p90", 0),
        "latency_p95": latency_percentiles.get("p95", 0),
        "latency_p99": latency_percentiles.get("p99", 0),
        "ttft_estimate": ttft_estimate,
        "ttft_p50": ttft_percentiles.get("p50", 0),
        "ttft_p90": ttft_percentiles.get("p90", 0),
        "ttft_p95": ttft_percentiles.get("p95", 0),
        "ttft_p99": ttft_percentiles.get("p99", 0),
        "tokens_per_second": tokens_per_second,
        "requests_per_second": requests_per_second,
        "throughput_per_user": throughput_per_user,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_tokens_per_request,
    }

    # ITL percentiles
    if itl_percentiles:
        result_dict["itl_p50"] = itl_percentiles.get("p50", 0)
        result_dict["itl_p90"] = itl_percentiles.get("p90", 0)
        result_dict["itl_p95"] = itl_percentiles.get("p95", 0)
        result_dict["itl_p99"] = itl_percentiles.get("p99", 0)

    # Prefill / decode (pp/tg) throughput split — only present when the
    # client-side streaming parser could derive real per-phase numbers.
    sources = {r.get("pp_tg_source") for r in successful if r.get("pp_tg_source")}
    if sources:
        # If every successful request agrees on a source, surface it; if
        # they disagree (mixed modes), label as "mixed".
        result_dict["pp_tg_source"] = sources.pop() if len(sources) == 1 else "mixed"

    prefill_values = [r.get("prefill_tps") for r in successful if r.get("prefill_tps") is not None]
    if prefill_values:
        prefill_percentiles = calculate_percentiles(prefill_values)
        result_dict["prefill_tps_mean"] = mean(prefill_values)
        result_dict["prefill_tps_p50"] = prefill_percentiles.get("p50", 0)
        result_dict["prefill_tps_p90"] = prefill_percentiles.get("p90", 0)
        result_dict["prefill_tps_p99"] = prefill_percentiles.get("p99", 0)

    decode_values = [r.get("decode_tps") for r in successful if r.get("decode_tps") is not None]
    if decode_values:
        decode_percentiles = calculate_percentiles(decode_values)
        result_dict["decode_tps_mean"] = mean(decode_values)
        result_dict["decode_tps_p50"] = decode_percentiles.get("p50", 0)
        result_dict["decode_tps_p90"] = decode_percentiles.get("p90", 0)
        result_dict["decode_tps_p99"] = decode_percentiles.get("p99", 0)

    # Cost analysis
    if cost_per_hour is not None and cost_per_hour > 0:
        cost_per_second = cost_per_hour / 3600
        total_cost = cost_per_second * total_time
        cost_per_1m_tokens = (total_cost / total_completion_tokens * 1_000_000) if total_completion_tokens > 0 else 0
        result_dict["cost_per_hour"] = cost_per_hour
        result_dict["cost_per_1m_tokens"] = cost_per_1m_tokens
        result_dict["cost_total"] = total_cost

    # Metrics stats
    if metrics_stats:
        result_dict.update(metrics_stats)
        if metrics_stats.get("actual_prefill_time", 0) > 0:
            result_dict["prefill_speed"] = avg_prompt_tokens / metrics_stats["actual_prefill_time"]
        else:
            result_dict["prefill_speed"] = 0

    # GPU stats
    if gpu_stats:
        result_dict.update(gpu_stats)
        result_dict["watts_per_token"] = gpu_stats["avg_power"] / tokens_per_second if tokens_per_second > 0 else 0
        result_dict["tokens_per_watt"] = tokens_per_second / gpu_stats["avg_power"] if gpu_stats["avg_power"] > 0 else 0
        result_dict["energy_joules"] = gpu_stats["avg_power"] * total_time
        result_dict["energy_per_token"] = result_dict["energy_joules"] / total_completion_tokens if total_completion_tokens > 0 else 0

    return result_dict


async def run_benchmark_async(
    context_length: int,
    num_concurrent_users: int,
    config: BenchmarkConfig,
    model_name: str = None,
    live_display=None,
    gpu_monitor: GPUMonitor = None,
    prompt_type: str = "classic",
    cost_per_hour: float = None,
) -> Optional[Dict]:
    """Execute benchmark using async engine.

    Drop-in replacement for the threaded run_benchmark().
    """
    if not live_display:
        print(f"\n{'=' * 100}")
        print(f"Testing: {context_length:,} token context | {num_concurrent_users} concurrent users | {prompt_type} prompt")
        print(f"{'=' * 100}")

    prompt_gen = get_prompt_generator(prompt_type)
    prompt = prompt_gen(context_length, model_name or "")
    actual_prompt_tokens = count_tokens(prompt, model_name or "")

    local_monitor = gpu_monitor if gpu_monitor else GPUMonitor(config.gpu_poll_interval)
    if not gpu_monitor:
        local_monitor.start()

    metrics_monitor = MetricsMonitor(config.metrics_endpoint)
    metrics_monitor.start()

    start_time = time.perf_counter()
    results = await _run_burst(prompt, num_concurrent_users, config, model_name)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    gpu_stats = None
    if not gpu_monitor:
        gpu_stats = local_monitor.stop(window_start=start_time, window_end=end_time)
    metrics_stats = metrics_monitor.stop()

    return _compute_stats(
        list(results), total_time, context_length, num_concurrent_users,
        prompt_type, actual_prompt_tokens, gpu_stats, metrics_stats, cost_per_hour,
    )


async def run_sustained_benchmark(
    context_length: int,
    target_rps: float,
    duration_seconds: float,
    config: BenchmarkConfig,
    model_name: str = None,
    prompt_type: str = "classic",
    cost_per_hour: float = None,
) -> Optional[Dict]:
    """Run sustained RPS benchmark.

    Launches requests at a steady rate and measures how latency
    degrades under sustained load.
    """
    prompt_gen = get_prompt_generator(prompt_type)
    prompt = prompt_gen(context_length, model_name or "")
    actual_prompt_tokens = count_tokens(prompt, model_name or "")

    gpu_monitor = GPUMonitor(config.gpu_poll_interval)
    gpu_monitor.start()

    metrics_monitor = MetricsMonitor(config.metrics_endpoint)
    metrics_monitor.start()

    start_time = time.perf_counter()
    results = await _run_sustained_rps(
        prompt, target_rps, duration_seconds, config, model_name,
    )
    end_time = time.perf_counter()
    total_time = end_time - start_time

    gpu_stats = gpu_monitor.stop(window_start=start_time, window_end=end_time)
    metrics_stats = metrics_monitor.stop()

    effective_users = int(target_rps * (mean([r["duration"] for r in results if r.get("success")]) if results else 1))

    stats = _compute_stats(
        results, total_time, context_length, max(effective_users, 1),
        prompt_type, actual_prompt_tokens, gpu_stats, metrics_stats, cost_per_hour,
    )

    if stats:
        stats["mode"] = "sustained_rps"
        stats["target_rps"] = target_rps
        stats["duration_seconds"] = duration_seconds
        stats["actual_rps"] = len([r for r in results if r.get("success")]) / total_time if total_time > 0 else 0

        # Time-bucketed latency (10-second windows)
        successful = [r for r in results if r.get("success")]
        if successful:
            bucket_size = 10.0
            buckets: dict[int, list[float]] = {}
            for r in successful:
                bucket_id = int(r["duration"] // bucket_size)
                buckets.setdefault(bucket_id, []).append(r["duration"])
            stats["latency_over_time"] = {
                f"{k * int(bucket_size)}s": {
                    "avg": mean(v), "p99": float(np.percentile(v, 99)), "count": len(v),
                }
                for k, v in sorted(buckets.items())
            }

    return stats


# ------------------------------------------------------------------
# Sync wrapper for backward compatibility
# ------------------------------------------------------------------

def run_benchmark(
    context_length: int,
    num_concurrent_users: int,
    config: BenchmarkConfig,
    model_name: str = None,
    live_display=None,
    gpu_monitor: GPUMonitor = None,
    prompt_type: str = "classic",
    cost_per_hour: float = None,
) -> Optional[Dict]:
    """Sync wrapper around the async benchmark engine.

    Maintains backward compatibility with the existing CLI.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop — fall back to threaded execution
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                run_benchmark_async(
                    context_length, num_concurrent_users, config,
                    model_name, live_display, gpu_monitor, prompt_type, cost_per_hour,
                ),
            )
            return future.result()

    return asyncio.run(
        run_benchmark_async(
            context_length, num_concurrent_users, config,
            model_name, live_display, gpu_monitor, prompt_type, cost_per_hour,
        ),
    )
