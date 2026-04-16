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
    benchmark execution.  Supports multi-GPU setups — when more than one
    GPU is detected, per-GPU data is collected and aggregated.
    """

    def __init__(self, poll_interval: float = 0.1) -> None:
        """Initialize GPU monitor.

        Args:
            poll_interval: Polling interval in seconds.
        """
        self.monitoring: bool = False
        self.stats: list[Dict] = []
        self.per_gpu_stats: list[list[Dict]] = []  # per-GPU snapshots
        self.thread: Optional[threading.Thread] = None
        self.poll_interval = poll_interval
        self.gpu_count: int = self._detect_gpu_count()

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gpu_count() -> int:
        """Detect the number of available NVIDIA GPUs.

        Returns:
            Number of GPUs found, or 1 as a safe fallback.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=count",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # nvidia-smi prints the total count on every line (one per GPU),
                # so the first non-empty line contains the value we need.
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if line:
                        return int(line)
        except Exception:
            pass
        return 1

    # ------------------------------------------------------------------
    # Per-GPU and aggregated queries
    # ------------------------------------------------------------------

    def _parse_gpu_line(self, line: str) -> Optional[Dict]:
        """Parse a single CSV line from nvidia-smi into a stats dict.

        Expected columns (in order):
            index, utilization.gpu, memory.used, memory.total,
            temperature.gpu, power.draw, clocks.gr, clocks.mem
        """
        try:
            values = [v.strip() for v in line.split(",")]
            if len(values) < 8:
                return None
            return {
                "gpu_index": int(values[0]),
                "gpu_util": float(values[1]),
                "mem_used": float(values[2]),
                "mem_total": float(values[3]),
                "temperature": float(values[4]),
                "power_draw": float(values[5]),
                "gpu_clock": float(values[6]),
                "mem_clock": float(values[7]),
                "timestamp": time.time(),
            }
        except (ValueError, IndexError):
            return None

    def get_all_gpu_stats(self) -> list[Dict]:
        """Query nvidia-smi for per-GPU statistics across all GPUs.

        Returns:
            List of dicts, one per GPU, each containing GPU metrics.
            Empty list if the query fails.
        """
        try:
            gpu_ids = ",".join(str(i) for i in range(self.gpu_count))
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_ids}",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw,clocks.gr,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode != 0:
                return []

            per_gpu: list[Dict] = []
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed = self._parse_gpu_line(line)
                if parsed is not None:
                    per_gpu.append(parsed)
            return per_gpu
        except Exception as e:
            print(f"[WARNING] GPU monitoring error: {e}", file=sys.stderr)
            return []

    def get_gpu_stats(self) -> Optional[Dict]:
        """Query nvidia-smi for current GPU statistics.

        When multiple GPUs are present the values are **averaged** across
        all GPUs so that callers that expect a single dict continue to
        work unchanged (backward-compatible).

        Returns:
            Dictionary containing GPU metrics or ``None`` if query fails.
        """
        per_gpu = self.get_all_gpu_stats()
        if not per_gpu:
            return None

        # Single GPU fast-path — drop the index key for compatibility
        if len(per_gpu) == 1:
            stats = dict(per_gpu[0])
            stats.pop("gpu_index", None)
            return stats

        # Multi-GPU: aggregate by averaging across GPUs
        return {
            "gpu_util": mean([g["gpu_util"] for g in per_gpu]),
            "mem_used": mean([g["mem_used"] for g in per_gpu]),
            "mem_total": mean([g["mem_total"] for g in per_gpu]),
            "temperature": mean([g["temperature"] for g in per_gpu]),
            "power_draw": mean([g["power_draw"] for g in per_gpu]),
            "gpu_clock": mean([g["gpu_clock"] for g in per_gpu]),
            "mem_clock": mean([g["mem_clock"] for g in per_gpu]),
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Background monitoring loop
    # ------------------------------------------------------------------

    def monitor_loop(self) -> None:
        """Background thread loop for continuous GPU monitoring."""
        while self.monitoring:
            if self.gpu_count > 1:
                per_gpu = self.get_all_gpu_stats()
                if per_gpu:
                    self.per_gpu_stats.append(per_gpu)
                    # Also store an aggregated snapshot for backward compat
                    agg = self.get_gpu_stats()
                    if agg:
                        self.stats.append(agg)
            else:
                stats = self.get_gpu_stats()
                if stats:
                    self.stats.append(stats)
            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start monitoring in background thread."""
        self.monitoring = True
        self.stats = []
        self.per_gpu_stats = []
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Optional[Dict]:
        """Stop monitoring and return aggregated statistics.

        Returns:
            Dictionary containing averaged and peak GPU metrics.
            Includes ``gpu_count`` and a ``per_gpu`` list of breakdowns
            when multiple GPUs are present.
        """
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.stats:
            return None

        result = {
            "gpu_count": self.gpu_count,
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

        # Include per-GPU breakdowns for multi-GPU setups
        if self.gpu_count > 1 and self.per_gpu_stats:
            result["per_gpu"] = self._aggregate_per_gpu()

        return result

    # ------------------------------------------------------------------
    # Per-GPU aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_per_gpu(self) -> list[Dict]:
        """Aggregate per-GPU statistics across all snapshots.

        Returns:
            List of dicts (one per GPU) with avg/max stats for the
            monitoring period.
        """
        # Collect samples grouped by GPU index
        by_index: Dict[int, list[Dict]] = {}
        for snapshot in self.per_gpu_stats:
            for gpu in snapshot:
                idx = gpu.get("gpu_index", 0)
                by_index.setdefault(idx, []).append(gpu)

        per_gpu_agg: list[Dict] = []
        for idx in sorted(by_index.keys()):
            samples = by_index[idx]
            per_gpu_agg.append({
                "gpu_index": idx,
                "avg_gpu_util": mean([s["gpu_util"] for s in samples]),
                "max_gpu_util": max([s["gpu_util"] for s in samples]),
                "avg_mem_used": mean([s["mem_used"] for s in samples]),
                "max_mem_used": max([s["mem_used"] for s in samples]),
                "avg_temperature": mean([s["temperature"] for s in samples]),
                "max_temperature": max([s["temperature"] for s in samples]),
                "avg_power": mean([s["power_draw"] for s in samples]),
                "max_power": max([s["power_draw"] for s in samples]),
                "avg_gpu_clock": mean([s["gpu_clock"] for s in samples]),
                "max_gpu_clock": max([s["gpu_clock"] for s in samples]),
                "avg_mem_clock": mean([s["mem_clock"] for s in samples]),
                "samples": len(samples),
            })
        return per_gpu_agg
