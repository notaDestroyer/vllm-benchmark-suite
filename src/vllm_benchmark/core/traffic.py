"""Real-world traffic simulation for vLLM benchmarking.

Provides realistic load patterns instead of synchronized batch requests:
- Poisson arrival rates (bursty, real-world traffic)
- Multi-turn conversation simulation (growing context windows)
- Think-time between requests (user reading delays)

Author: amit
License: MIT
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict

import requests


@dataclass
class TrafficConfig:
    """Configuration for traffic simulation."""

    # Poisson arrival
    target_rps: float = 2.0  # Target requests per second
    duration_seconds: float = 60.0  # How long to run the simulation
    warmup_seconds: float = 5.0  # Ramp-up period

    # Multi-turn conversations
    multi_turn: bool = False
    turns_per_conversation: int = 5
    think_time_mean: float = 3.0  # Mean think time between turns (seconds)
    think_time_std: float = 1.0  # Std dev of think time
    context_growth_per_turn: int = 500  # Approximate tokens added per turn

    # Request parameters
    initial_context_tokens: int = 1000
    max_tokens: int = 500
    temperature: float = 0.7


@dataclass
class TrafficResult:
    """Results from a traffic simulation run."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    duration: float = 0.0
    actual_rps: float = 0.0

    # Latency stats
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p99_latency: float = 0.0
    avg_ttft: float = 0.0

    # Throughput
    total_tokens_generated: int = 0
    tokens_per_second: float = 0.0

    # Multi-turn specific
    conversations_completed: int = 0
    avg_turns_per_conversation: float = 0.0
    latency_by_turn: dict[int, list[float]] = field(default_factory=dict)


def _poisson_interarrival(rate: float) -> float:
    """Generate a Poisson-distributed inter-arrival time."""
    if rate <= 0:
        return 1.0
    return -math.log(1.0 - random.random()) / rate


def _make_single_request(
    prompt: str,
    model_name: str,
    api_endpoint: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    stream: bool = True,
) -> Dict:
    """Execute a single request and return timing metrics."""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    start = time.time()
    try:
        if stream:
            response = requests.post(api_endpoint, json=data, timeout=timeout, stream=True)
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}", "duration": time.time() - start}

            first_token_time = None
            completion_tokens = 0
            response_text = ""

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if "usage" in chunk:
                    completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)

                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        response_text += content

            duration = time.time() - start
            ttft = (first_token_time - start) if first_token_time else duration * 0.15
            if completion_tokens == 0:
                completion_tokens = len(response_text.split())  # rough estimate

            return {
                "success": True,
                "duration": duration,
                "ttft": ttft,
                "completion_tokens": completion_tokens,
                "response_text": response_text,
            }
        else:
            response = requests.post(api_endpoint, json=data, timeout=timeout)
            duration = time.time() - start
            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                return {
                    "success": True,
                    "duration": duration,
                    "ttft": duration * 0.15,
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "response_text": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                }
            return {"success": False, "error": f"HTTP {response.status_code}", "duration": duration}
    except Exception as e:
        return {"success": False, "error": str(e), "duration": time.time() - start}


def run_poisson_traffic(
    config: TrafficConfig,
    model_name: str,
    api_endpoint: str,
    prompt_generator,
    request_timeout: int = 900,
    on_request_complete=None,
) -> TrafficResult:
    """Run a Poisson-arrival traffic simulation.

    Sends requests at random intervals following a Poisson process,
    simulating real-world bursty traffic patterns.

    Args:
        config: Traffic simulation configuration.
        model_name: Model name for API requests.
        api_endpoint: vLLM API endpoint URL.
        prompt_generator: Callable(token_count, model_name) -> str
        request_timeout: Request timeout in seconds.
        on_request_complete: Optional callback(request_result_dict) for live updates.

    Returns:
        TrafficResult with aggregated statistics.
    """
    result = TrafficResult()
    all_latencies: list[float] = []
    all_ttfts: list[float] = []
    total_tokens = 0
    lock = threading.Lock()

    def fire_request():
        prompt = prompt_generator(config.initial_context_tokens, model_name)
        req_result = _make_single_request(
            prompt, model_name, api_endpoint,
            config.max_tokens, config.temperature, request_timeout,
        )
        with lock:
            if req_result["success"]:
                result.successful_requests += 1
                all_latencies.append(req_result["duration"])
                all_ttfts.append(req_result.get("ttft", 0))
                nonlocal total_tokens
                total_tokens += req_result.get("completion_tokens", 0)
            else:
                result.failed_requests += 1
            result.total_requests += 1

        if on_request_complete:
            on_request_complete(req_result)

    sim_start = time.time()
    threads: list[threading.Thread] = []

    # Ramp up: linearly increase rate during warmup
    while time.time() - sim_start < config.duration_seconds:
        elapsed = time.time() - sim_start

        # During warmup, scale rate linearly
        if elapsed < config.warmup_seconds and config.warmup_seconds > 0:
            current_rate = config.target_rps * (elapsed / config.warmup_seconds)
        else:
            current_rate = config.target_rps

        if current_rate > 0:
            wait = _poisson_interarrival(current_rate)
        else:
            wait = 0.5

        time.sleep(wait)

        t = threading.Thread(target=fire_request, daemon=True)
        t.start()
        threads.append(t)

    # Wait for in-flight requests (with timeout)
    for t in threads:
        t.join(timeout=request_timeout)

    result.duration = time.time() - sim_start
    result.actual_rps = result.total_requests / result.duration if result.duration > 0 else 0
    result.total_tokens_generated = total_tokens
    result.tokens_per_second = total_tokens / result.duration if result.duration > 0 else 0
    result.latencies = all_latencies
    result.ttfts = all_ttfts

    if all_latencies:
        import numpy as np
        arr = np.array(all_latencies)
        result.avg_latency = float(np.mean(arr))
        result.p50_latency = float(np.percentile(arr, 50))
        result.p90_latency = float(np.percentile(arr, 90))
        result.p99_latency = float(np.percentile(arr, 99))

    if all_ttfts:
        result.avg_ttft = mean(all_ttfts)

    return result


def run_multiturn_traffic(
    config: TrafficConfig,
    model_name: str,
    api_endpoint: str,
    prompt_generator,
    request_timeout: int = 900,
    on_request_complete=None,
) -> TrafficResult:
    """Run a multi-turn conversation simulation.

    Simulates realistic chat workloads where context grows with each turn,
    with think-time delays between turns to simulate user reading.

    Args:
        config: Traffic simulation configuration.
        model_name: Model name for API requests.
        api_endpoint: vLLM API endpoint URL.
        prompt_generator: Callable(token_count, model_name) -> str
        request_timeout: Request timeout in seconds.
        on_request_complete: Optional callback for live updates.

    Returns:
        TrafficResult with per-turn latency breakdowns.
    """
    result = TrafficResult()
    all_latencies: list[float] = []
    all_ttfts: list[float] = []
    total_tokens = 0
    latency_by_turn: dict[int, list[float]] = {}
    lock = threading.Lock()
    conversations_done = 0

    def run_conversation():
        nonlocal conversations_done, total_tokens

        conversation_history: list[dict] = []
        current_context_size = config.initial_context_tokens

        for turn in range(config.turns_per_conversation):
            # Generate prompt for this turn
            user_prompt = prompt_generator(current_context_size, model_name)

            # Build messages with conversation history
            messages = list(conversation_history)
            messages.append({"role": "user", "content": user_prompt})

            data = {
                "model": model_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "stream": True,
            }

            start = time.time()
            try:
                response = requests.post(api_endpoint, json=data, timeout=request_timeout, stream=True)
                if response.status_code != 200:
                    with lock:
                        result.failed_requests += 1
                        result.total_requests += 1
                    continue

                first_token_time = None
                response_text = ""
                completion_tokens = 0

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if "usage" in chunk:
                        completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)

                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            response_text += content

                duration = time.time() - start
                ttft = (first_token_time - start) if first_token_time else duration * 0.15

                with lock:
                    result.successful_requests += 1
                    result.total_requests += 1
                    all_latencies.append(duration)
                    all_ttfts.append(ttft)
                    total_tokens += completion_tokens
                    latency_by_turn.setdefault(turn, []).append(duration)

                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_prompt})
                conversation_history.append({"role": "assistant", "content": response_text[:200]})

                # Grow context for next turn
                current_context_size += config.context_growth_per_turn

                if on_request_complete:
                    on_request_complete({"success": True, "duration": duration, "turn": turn, "ttft": ttft})

            except Exception:
                with lock:
                    result.failed_requests += 1
                    result.total_requests += 1

            # Think time between turns
            if turn < config.turns_per_conversation - 1:
                think = max(0.5, random.gauss(config.think_time_mean, config.think_time_std))
                time.sleep(think)

        with lock:
            nonlocal conversations_done
            conversations_done += 1

    # Launch concurrent conversations based on target RPS
    # Each conversation has N turns, so we need enough conversations
    # to sustain the target RPS over the duration
    avg_turn_time = 5.0  # rough estimate
    total_conversation_time = config.turns_per_conversation * (avg_turn_time + config.think_time_mean)
    concurrent_conversations = max(1, int(config.target_rps * total_conversation_time / config.turns_per_conversation))
    concurrent_conversations = min(concurrent_conversations, 50)  # cap at 50

    sim_start = time.time()
    threads: list[threading.Thread] = []

    for _ in range(concurrent_conversations):
        t = threading.Thread(target=run_conversation, daemon=True)
        t.start()
        threads.append(t)
        # Stagger conversation starts
        time.sleep(_poisson_interarrival(config.target_rps))

    for t in threads:
        t.join(timeout=request_timeout * config.turns_per_conversation)

    result.duration = time.time() - sim_start
    result.actual_rps = result.total_requests / result.duration if result.duration > 0 else 0
    result.total_tokens_generated = total_tokens
    result.tokens_per_second = total_tokens / result.duration if result.duration > 0 else 0
    result.latencies = all_latencies
    result.ttfts = all_ttfts
    result.conversations_completed = conversations_done
    result.avg_turns_per_conversation = (
        result.total_requests / conversations_done if conversations_done > 0 else 0
    )
    result.latency_by_turn = latency_by_turn

    if all_latencies:
        import numpy as np
        arr = np.array(all_latencies)
        result.avg_latency = float(np.mean(arr))
        result.p50_latency = float(np.percentile(arr, 50))
        result.p90_latency = float(np.percentile(arr, 90))
        result.p99_latency = float(np.percentile(arr, 99))

    if all_ttfts:
        result.avg_ttft = mean(all_ttfts)

    return result


def format_traffic_report(result: TrafficResult, config: TrafficConfig) -> str:
    """Format traffic simulation results as a Rich-compatible string."""
    lines: list[str] = []
    lines.append("")
    mode = "Multi-Turn Conversations" if config.multi_turn else "Poisson Arrivals"
    lines.append(f"  [bold cyan]Traffic Simulation: {mode}[/]")
    lines.append("")
    lines.append(f"  Duration:        {result.duration:.1f}s")
    lines.append(f"  Target RPS:      {config.target_rps:.1f}")
    lines.append(f"  Actual RPS:      {result.actual_rps:.1f}")
    lines.append(f"  Total Requests:  {result.total_requests}")
    lines.append(f"  Successful:      {result.successful_requests}")
    lines.append(f"  Failed:          {result.failed_requests}")
    lines.append("")
    lines.append(f"  Throughput:      {result.tokens_per_second:.1f} tok/s")
    lines.append(f"  Avg Latency:     {result.avg_latency:.2f}s")
    lines.append(f"  P50 Latency:     {result.p50_latency:.2f}s")
    lines.append(f"  P90 Latency:     {result.p90_latency:.2f}s")
    lines.append(f"  P99 Latency:     {result.p99_latency:.2f}s")
    lines.append(f"  Avg TTFT:        {result.avg_ttft * 1000:.0f}ms")

    if config.multi_turn and result.latency_by_turn:
        lines.append("")
        lines.append("  [bold]Latency by Turn:[/]")
        for turn in sorted(result.latency_by_turn.keys()):
            lats = result.latency_by_turn[turn]
            avg = mean(lats) if lats else 0
            lines.append(f"    Turn {turn + 1}: {avg:.2f}s avg ({len(lats)} requests)")
        lines.append(f"  Conversations:   {result.conversations_completed}")

    lines.append("")
    return "\n".join(lines)
