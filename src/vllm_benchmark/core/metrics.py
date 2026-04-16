"""GPU and vLLM metrics monitoring.

Provides real-time GPU telemetry via nvidia-smi polling and vLLM
Prometheus metrics tracking for cache hit rates, prefill/decode times.

Author: amit
License: MIT
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from statistics import mean
from typing import Dict, Optional

import requests


class MetricsMonitor:
    """vLLM metrics monitoring system.

    Queries the vLLM Prometheus metrics endpoint to track:
    - Prefix cache hit rate (queries and hits)
    - Prefill time (input processing)
    - Decode time (output generation)
    """

    def __init__(self, metrics_url: str) -> None:
        """Initialize metrics monitor.

        Args:
            metrics_url: Full URL to the vLLM ``/metrics`` endpoint,
                e.g. ``"http://localhost:8000/metrics"``.
        """
        self.metrics_url = metrics_url
        self.baseline_queries: float = 0
        self.baseline_hits: float = 0
        self.baseline_prefill_time: float = 0.0
        self.baseline_decode_time: float = 0.0
        self.available: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_metrics(self) -> Optional[Dict]:
        """Query vLLM metrics endpoint for current statistics.

        Returns:
            Dictionary containing metrics, or ``None`` if unavailable.
        """
        try:
            response = requests.get(self.metrics_url, timeout=2)

            if response.status_code != 200:
                return None

            metrics_text = response.text
            queries = 0.0
            hits = 0.0
            prefill_time = 0.0
            decode_time = 0.0

            for line in metrics_text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue

                # Extract prefix cache queries
                if "vllm:prefix_cache_queries_total" in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            queries = float(parts[-1])
                    except Exception:
                        pass

                # Extract prefix cache hits
                if "vllm:prefix_cache_hits_total" in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            hits = float(parts[-1])
                    except Exception:
                        pass

                # Extract prefill time sum
                if "vllm:request_prefill_time_seconds_sum" in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            prefill_time = float(parts[-1])
                    except Exception:
                        pass

                # Extract decode time sum
                if "vllm:request_decode_time_seconds_sum" in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            decode_time = float(parts[-1])
                    except Exception:
                        pass

            return {
                "cache_queries": queries,
                "cache_hits": hits,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "timestamp": time.time(),
            }

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public start / stop interface
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Capture baseline metrics at start of test.

        Returns:
            ``True`` if metrics are available, ``False`` otherwise.
        """
        stats = self.get_metrics()
        if stats:
            self.baseline_queries = stats["cache_queries"]
            self.baseline_hits = stats["cache_hits"]
            self.baseline_prefill_time = stats["prefill_time"]
            self.baseline_decode_time = stats["decode_time"]
            self.available = True
            return True
        else:
            self.available = False
            return False

    def stop(self) -> Optional[Dict]:
        """Calculate metrics deltas since start.

        Returns:
            Dictionary containing cache hit rate, prefill/decode times,
            or ``None`` if unavailable.
        """
        if not self.available:
            return None

        stats = self.get_metrics()
        if not stats:
            return None

        # Calculate cache deltas
        delta_queries = stats["cache_queries"] - self.baseline_queries
        delta_hits = stats["cache_hits"] - self.baseline_hits
        hit_rate = (delta_hits / delta_queries * 100) if delta_queries > 0 else 0

        # Calculate time deltas
        delta_prefill = stats["prefill_time"] - self.baseline_prefill_time
        delta_decode = stats["decode_time"] - self.baseline_decode_time

        return {
            "cache_hit_rate": hit_rate,
            "cache_queries_delta": delta_queries,
            "cache_hits_delta": delta_hits,
            "total_cache_queries": stats["cache_queries"],
            "total_cache_hits": stats["cache_hits"],
            "actual_prefill_time": delta_prefill,
            "actual_decode_time": delta_decode,
        }


class GPUMonitor:
    """Real-time GPU performance monitoring system.

    Polls ``nvidia-smi`` at regular intervals to collect GPU utilization,
    memory usage, temperature, power draw, and clock frequencies during
    benchmark execution.
    """

    def __init__(self, poll_interval: float = 0.1) -> None:
        """Initialize GPU monitor.

        Args:
            poll_interval: Polling interval in seconds.
        """
        self.monitoring: bool = False
        self.stats: list[Dict] = []
        self.thread: Optional[threading.Thread] = None
        self.poll_interval = poll_interval

    def get_gpu_stats(self) -> Optional[Dict]:
        """Query nvidia-smi for current GPU statistics.

        Returns:
            Dictionary containing GPU metrics or ``None`` if query fails.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw,clocks.gr,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "gpu_util": float(values[0]),
                    "mem_used": float(values[1]),
                    "mem_total": float(values[2]),
                    "temperature": float(values[3]),
                    "power_draw": float(values[4]),
                    "gpu_clock": float(values[5]),
                    "mem_clock": float(values[6]),
                    "timestamp": time.time(),
                }
        except Exception as e:
            print(f"[WARNING] GPU monitoring error: {e}", file=sys.stderr)
        return None

    def monitor_loop(self) -> None:
        """Background thread loop for continuous GPU monitoring."""
        while self.monitoring:
            stats = self.get_gpu_stats()
            if stats:
                self.stats.append(stats)
            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start monitoring in background thread."""
        self.monitoring = True
        self.stats = []
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Optional[Dict]:
        """Stop monitoring and return aggregated statistics.

        Returns:
            Dictionary containing averaged and peak GPU metrics.
        """
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.stats:
            return None

        return {
            "avg_gpu_util": mean([s["gpu_util"] for s in self.stats]),
            "max_gpu_util": max([s["gpu_util"] for s in self.stats]),
            "avg_mem_used": mean([s["mem_used"] for s in self.stats]),
            "max_mem_used": max([s["mem_used"] for s in self.stats]),
            "avg_temperature": mean([s["temperature"] for s in self.stats]),
            "max_temperature": max([s["temperature"] for s in self.stats]),
            "avg_power": mean([s["power_draw"] for s in self.stats]),
            "max_power": max([s["power_draw"] for s in self.stats]),
            "avg_gpu_clock": mean([s["gpu_clock"] for s in self.stats]),
            "max_gpu_clock": max([s["gpu_clock"] for s in self.stats]),
            "avg_mem_clock": mean([s["mem_clock"] for s in self.stats]),
            "samples": len(self.stats),
        }
