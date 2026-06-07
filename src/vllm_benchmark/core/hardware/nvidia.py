"""NVIDIA GPU hardware monitor.

Polls ``nvidia-smi`` at regular intervals to collect GPU utilization,
memory, temperature, power, and clocks during benchmark execution.
Supports multi-GPU setups via per-device snapshots that are aggregated
on ``stop()``.

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

from vllm_benchmark.core.hardware.base import HardwareMonitor


class NvidiaMonitor(HardwareMonitor):
    """Real-time NVIDIA GPU performance monitor.

    Polls ``nvidia-smi`` at regular intervals to collect GPU utilization,
    memory usage, temperature, power draw, and clock frequencies during
    benchmark execution.  Supports multi-GPU setups — when more than one
    GPU is detected, per-GPU data is collected and aggregated.
    """

    def __init__(self, poll_interval: float = 0.1) -> None:
        """Initialize the NVIDIA monitor.

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
                "perf_counter": time.perf_counter(),
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

        When multiple GPUs are present the values are aggregated so that
        callers expecting a single dict continue to work unchanged.  Power
        is **summed** across devices (total board draw) while utilization,
        memory, temperature, and clocks are averaged — preserving the
        historical backward-compatible semantics.

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

        # Multi-GPU: sum power (total board draw), average the rest
        return {
            "gpu_util": mean([g["gpu_util"] for g in per_gpu]),
            "mem_used": mean([g["mem_used"] for g in per_gpu]),
            "mem_total": mean([g["mem_total"] for g in per_gpu]),
            "temperature": mean([g["temperature"] for g in per_gpu]),
            "power_draw": sum([g["power_draw"] for g in per_gpu]),
            "gpu_clock": mean([g["gpu_clock"] for g in per_gpu]),
            "mem_clock": mean([g["mem_clock"] for g in per_gpu]),
            "timestamp": time.time(),
            "perf_counter": time.perf_counter(),
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

    def stop(
        self,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None,
    ) -> Optional[Dict]:
        """Stop monitoring and return aggregated statistics.

        Args:
            window_start: Optional ``time.perf_counter()`` value; samples
                taken before this are excluded.  ``None`` means no lower
                bound (include from the beginning).
            window_end: Optional ``time.perf_counter()`` value; samples
                taken after this are excluded.  ``None`` means no upper
                bound (include to the end).

        Returns:
            Dictionary containing averaged and peak GPU metrics.
            Includes ``gpu_count`` and a ``per_gpu`` list of breakdowns
            when multiple GPUs are present.  ``None`` if no samples fall
            within the window.
        """
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

        stats = self._filter_window(self.stats, window_start, window_end)
        if not stats:
            return None

        result = {
            "gpu_count": self.gpu_count,
            "avg_gpu_util": mean([s["gpu_util"] for s in stats]),
            "max_gpu_util": max([s["gpu_util"] for s in stats]),
            "avg_mem_used": mean([s["mem_used"] for s in stats]),
            "max_mem_used": max([s["mem_used"] for s in stats]),
            "avg_temperature": mean([s["temperature"] for s in stats]),
            "max_temperature": max([s["temperature"] for s in stats]),
            "avg_power": mean([s["power_draw"] for s in stats]),
            "max_power": max([s["power_draw"] for s in stats]),
            "avg_gpu_clock": mean([s["gpu_clock"] for s in stats]),
            "max_gpu_clock": max([s["gpu_clock"] for s in stats]),
            "avg_mem_clock": mean([s["mem_clock"] for s in stats]),
            "samples": len(stats),
        }

        # Include per-GPU breakdowns for multi-GPU setups
        if self.gpu_count > 1 and self.per_gpu_stats:
            result["per_gpu"] = self._aggregate_per_gpu(window_start, window_end)

        return result

    # ------------------------------------------------------------------
    # Window filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _in_window(
        sample: Dict,
        window_start: Optional[float],
        window_end: Optional[float],
    ) -> bool:
        """Return whether a sample falls within the perf_counter window."""
        if window_start is None and window_end is None:
            return True
        pc = sample.get("perf_counter")
        if pc is None:
            # Samples without a perf_counter timestamp are always kept so
            # that legacy callers / fixtures are not silently dropped.
            return True
        if window_start is not None and pc < window_start:
            return False
        if window_end is not None and pc > window_end:
            return False
        return True

    @classmethod
    def _filter_window(
        cls,
        samples: list[Dict],
        window_start: Optional[float],
        window_end: Optional[float],
    ) -> list[Dict]:
        """Filter a list of samples to those within the window."""
        if window_start is None and window_end is None:
            return samples
        return [s for s in samples if cls._in_window(s, window_start, window_end)]

    # ------------------------------------------------------------------
    # Per-GPU aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_per_gpu(
        self,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None,
    ) -> list[Dict]:
        """Aggregate per-GPU statistics across all snapshots.

        Args:
            window_start: Optional perf_counter lower bound.
            window_end: Optional perf_counter upper bound.

        Returns:
            List of dicts (one per GPU) with avg/max stats for the
            monitoring period.
        """
        # Collect samples grouped by GPU index
        by_index: Dict[int, list[Dict]] = {}
        for snapshot in self.per_gpu_stats:
            for gpu in snapshot:
                if not self._in_window(gpu, window_start, window_end):
                    continue
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
