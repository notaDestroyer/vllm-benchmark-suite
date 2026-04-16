"""Core benchmark execution engine.

Handles request dispatch, timing, metric collection, warmup, and
streaming-based TTFT measurement.

Author: amit
License: MIT
"""

from __future__ import annotations

import json
import threading
import time
from statistics import mean, stdev
from typing import Dict, List, Optional

import numpy as np
import requests
from rich.console import Console

from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core.metrics import GPUMonitor, MetricsMonitor
from vllm_benchmark.core.prompts import get_prompt_generator

console = Console()


# ------------------------------------------------------------------
# Percentile helpers
# ------------------------------------------------------------------

def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate P50, P90, P95, P99 percentiles using numpy."""
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
# Request executors
# ------------------------------------------------------------------

def make_request(
    prompt: str,
    request_id: int,
    results: List[Dict],
    max_tokens: int = 500,
    model_name: str = None,
    api_endpoint: str = "http://localhost:8000/v1/chat/completions",
    request_timeout: int = 900,
    streaming: bool = True,
) -> None:
    """Execute a single API request and record timing metrics.

    When *streaming* is True, uses SSE streaming to measure true
    Time-to-First-Token and inter-token latency.
    """
    if streaming:
        _make_streaming_request(
            prompt, request_id, results, max_tokens,
            model_name, api_endpoint, request_timeout,
        )
    else:
        _make_batch_request(
            prompt, request_id, results, max_tokens,
            model_name, api_endpoint, request_timeout,
        )


def _make_batch_request(
    prompt: str,
    request_id: int,
    results: List[Dict],
    max_tokens: int,
    model_name: str,
    api_endpoint: str,
    request_timeout: int,
) -> None:
    """Non-streaming request with estimated TTFT."""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        start = time.time()
        response = requests.post(api_endpoint, json=data, timeout=request_timeout)
        duration = time.time() - start

        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            itl = duration / completion_tokens if completion_tokens > 0 else 0
            prefill_estimate = duration * 0.15
            decode_estimate = duration - prefill_estimate

            results.append({
                "request_id": request_id,
                "duration": duration,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens", 0),
                "inter_token_latency": itl,
                "ttft": prefill_estimate,
                "prefill_time_estimate": prefill_estimate,
                "decode_time_estimate": decode_estimate,
                "success": True,
                "streaming": False,
            })
        else:
            results.append({
                "request_id": request_id,
                "duration": duration,
                "success": False,
                "error": f"HTTP {response.status_code}",
            })
    except Exception as e:
        results.append({"request_id": request_id, "success": False, "error": str(e)})


def _make_streaming_request(
    prompt: str,
    request_id: int,
    results: List[Dict],
    max_tokens: int,
    model_name: str,
    api_endpoint: str,
    request_timeout: int,
) -> None:
    """Streaming request that measures true TTFT via SSE chunks."""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    try:
        start = time.time()
        response = requests.post(
            api_endpoint, json=data, timeout=request_timeout, stream=True,
        )

        if response.status_code != 200:
            results.append({
                "request_id": request_id,
                "duration": time.time() - start,
                "success": False,
                "error": f"HTTP {response.status_code}",
            })
            return

        first_token_time = None
        chunk_times: list[float] = []
        completion_tokens = 0
        prompt_tokens = 0

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue

            payload = line[6:]  # strip "data: " prefix
            if payload.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            now = time.time()

            # Extract usage from final chunk if available
            if "usage" in chunk:
                usage = chunk["usage"]
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if choices[0].get("finish_reason"):
                pass  # finish_reason tracked via [DONE] sentinel

            if content is not None:
                chunk_times.append(now)
                if first_token_time is None:
                    first_token_time = now

        duration = time.time() - start

        # Calculate true TTFT
        ttft = (first_token_time - start) if first_token_time else duration * 0.15

        # Calculate true inter-token latency from chunk timestamps
        if len(chunk_times) >= 2:
            inter_times = [
                chunk_times[i] - chunk_times[i - 1]
                for i in range(1, len(chunk_times))
            ]
            avg_itl = mean(inter_times)
        else:
            avg_itl = duration / max(completion_tokens, 1)

        # If we didn't get token counts from usage, estimate from chunks
        if completion_tokens == 0:
            completion_tokens = len(chunk_times)

        results.append({
            "request_id": request_id,
            "duration": duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "inter_token_latency": avg_itl,
            "ttft": ttft,
            "prefill_time_estimate": ttft,
            "decode_time_estimate": duration - ttft,
            "success": True,
            "streaming": True,
        })

    except Exception as e:
        results.append({"request_id": request_id, "success": False, "error": str(e)})


# ------------------------------------------------------------------
# Warmup
# ------------------------------------------------------------------

def warmup_model(config: BenchmarkConfig, model_name: str, output_tokens: int = 100) -> bool:
    """Execute warmup inference to initialize GPU kernels and caches."""
    console.print("\n[yellow]Warming up model (1K context, single user)...[/yellow]")

    from vllm_benchmark.core.prompts import generate_prompt
    prompt = generate_prompt(1000)
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.7,
    }

    try:
        with console.status("[bold yellow]Executing warmup inference...", spinner="dots"):
            start = time.time()
            response = requests.post(config.api_endpoint, json=data, timeout=config.request_timeout)
            duration = time.time() - start

            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                console.print(
                    f"[green]OK[/green] Warmup complete: {duration:.2f}s, "
                    f"{completion_tokens} tokens generated"
                )
                return True
            else:
                console.print(f"[red]FAIL[/red] Warmup failed: HTTP {response.status_code}")
                return False
    except Exception as e:
        console.print(f"[red]FAIL[/red] Warmup error: {str(e)}")
        return False


# ------------------------------------------------------------------
# Main benchmark runner
# ------------------------------------------------------------------

def run_benchmark(
    context_length: int,
    num_concurrent_users: int,
    config: BenchmarkConfig,
    model_name: str = None,
    live_display=None,
    gpu_monitor: GPUMonitor = None,
    prompt_type: str = "classic",
) -> Optional[Dict]:
    """Execute benchmark for a specific context length and concurrency level.

    Returns:
        Dictionary containing performance metrics or None on failure.
    """
    if not live_display:
        print(f"\n{'=' * 100}")
        print(f"Testing: {context_length:,} token context | {num_concurrent_users} concurrent users | {prompt_type} prompt")
        print(f"{'=' * 100}")

    # Generate prompt
    prompt_gen = get_prompt_generator(prompt_type)
    prompt = prompt_gen(context_length, model_name or "")
    actual_prompt_tokens = len(prompt) // 4

    results: List[Dict] = []
    threads: List[threading.Thread] = []

    # Use existing or create local GPU monitor
    local_monitor = gpu_monitor if gpu_monitor else GPUMonitor(config.gpu_poll_interval)
    if not gpu_monitor:
        local_monitor.start()

    # Initialize metrics monitor
    metrics_monitor = MetricsMonitor(config.metrics_endpoint)
    metrics_monitor.start()

    start_time = time.time()

    # Launch concurrent requests
    for i in range(num_concurrent_users):
        t = threading.Thread(
            target=make_request,
            args=(prompt, i, results, config.output_tokens, model_name,
                  config.api_endpoint, config.request_timeout, config.streaming),
        )
        threads.append(t)
        t.start()

    # Wait for completion
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)

    for t in threads:
        t.join()

    total_time = time.time() - start_time

    # Stop monitors
    gpu_stats = None
    if not gpu_monitor:
        gpu_stats = local_monitor.stop()

    metrics_stats = metrics_monitor.stop()

    # Calculate statistics
    successful = [r for r in results if r.get("success", False)]
    failed = len(results) - len(successful)

    if not successful:
        if not live_display:
            print("\n[ERROR] All requests failed!")
            for r in results[:3]:
                if not r.get("success"):
                    print(f"  Error: {r.get('error', 'Unknown')}")
        return None

    durations = [r["duration"] for r in successful]
    completion_tokens_list = [r["completion_tokens"] for r in successful]
    prompt_tokens_list = [r["prompt_tokens"] for r in successful]

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0
    min_duration = min(durations)
    max_duration = max(durations)

    total_completion_tokens = sum(completion_tokens_list)
    avg_prompt_tokens = mean(prompt_tokens_list) if prompt_tokens_list else actual_prompt_tokens

    tokens_per_second = total_completion_tokens / total_time
    requests_per_second = len(successful) / total_time
    avg_tokens_per_request = mean(completion_tokens_list) if completion_tokens_list else 0

    # Use true TTFT if available from streaming, else estimate
    ttft_values = [r.get("ttft") for r in successful if r.get("ttft") is not None]
    ttft_estimate = mean(ttft_values) if ttft_values else avg_duration * 0.1

    throughput_per_user = tokens_per_second / num_concurrent_users if num_concurrent_users > 0 else 0

    # Build result dict
    result_dict: Dict = {
        "context_length": context_length,
        "concurrent_users": num_concurrent_users,
        "prompt_type": prompt_type,
        "total_time": total_time,
        "successful": len(successful),
        "failed": failed,
        "avg_latency": avg_duration,
        "std_latency": std_duration,
        "min_latency": min_duration,
        "max_latency": max_duration,
        "ttft_estimate": ttft_estimate,
        "tokens_per_second": tokens_per_second,
        "requests_per_second": requests_per_second,
        "throughput_per_user": throughput_per_user,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_tokens_per_request,
    }

    # Add metrics stats
    if metrics_stats:
        result_dict.update(metrics_stats)
        if metrics_stats.get("actual_prefill_time", 0) > 0:
            result_dict["prefill_speed"] = avg_prompt_tokens / metrics_stats["actual_prefill_time"]
        else:
            result_dict["prefill_speed"] = 0

    # Merge GPU stats and calculate energy efficiency
    if gpu_stats:
        result_dict.update(gpu_stats)
        result_dict["watts_per_token"] = (
            gpu_stats["avg_power"] / tokens_per_second if tokens_per_second > 0 else 0
        )
        result_dict["watts_per_token_per_user"] = (
            (gpu_stats["avg_power"] / tokens_per_second / num_concurrent_users)
            if (tokens_per_second > 0 and num_concurrent_users > 0) else 0
        )
        result_dict["throughput_per_user_per_watt"] = (
            throughput_per_user / gpu_stats["avg_power"] if gpu_stats["avg_power"] > 0 else 0
        )
        context_k = context_length / 1000
        result_dict["watts_per_token_per_user_per_1k_context"] = (
            result_dict["watts_per_token_per_user"] / context_k if context_k > 0 else 0
        )
        result_dict["tokens_per_watt"] = (
            tokens_per_second / gpu_stats["avg_power"] if gpu_stats["avg_power"] > 0 else 0
        )
        result_dict["energy_joules"] = gpu_stats["avg_power"] * total_time
        result_dict["energy_per_token"] = (
            result_dict["energy_joules"] / total_completion_tokens if total_completion_tokens > 0 else 0
        )
        result_dict["energy_watt_hours"] = result_dict["energy_joules"] / 3600
        result_dict["energy_per_token_per_1k_context"] = (
            result_dict["energy_per_token"] / context_k if context_k > 0 else 0
        )

    return result_dict
