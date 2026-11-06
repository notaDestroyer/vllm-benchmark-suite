"""GPU and vLLM metrics monitoring.

Provides real-time GPU telemetry via nvidia-smi polling and vLLM
Prometheus metrics tracking for cache hit rates, prefill/decode times.

Author: amit
License: MIT
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import requests

from vllm_benchmark.core.hardware.nvidia import NvidiaMonitor


def _parse_prom_line(line: str) -> Optional[tuple[str, float]]:
    """Parse one Prometheus exposition line into ``(metric_name, value)``.

    Handles both ``name value`` and ``name{labels} value`` forms.  Returns
    ``None`` for comments, blank lines, or unparsable values.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        head, value_str = line.rsplit(" ", 1)
    except ValueError:
        return None
    name = head.split("{", 1)[0].strip()
    if not name:
        return None
    try:
        return name, float(value_str)
    except ValueError:
        return None


#: Metric-name aliases, backend-agnostic.  Each canonical key maps to the
#: exposition names (vLLM first, then SGLang) that carry it.
_GAUGE_ALIASES: dict[str, tuple[str, ...]] = {
    # Fraction of KV-cache blocks in use (0..1).
    "gpu_cache_usage_perc": ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc", "sglang:token_usage"),
    "num_requests_waiting": ("vllm:num_requests_waiting", "sglang:num_queue_reqs"),
    "num_requests_running": ("vllm:num_requests_running", "sglang:num_running_reqs"),
}
_COUNTER_ALIASES: dict[str, tuple[str, ...]] = {
    "cache_queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    "cache_hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    "prefill_time": ("vllm:request_prefill_time_seconds_sum",),
    "decode_time": ("vllm:request_decode_time_seconds_sum",),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
}


def parse_metrics_text(metrics_text: str) -> Dict:
    """Parse a Prometheus ``/metrics`` payload into the canonical dict.

    Counters are summed across label sets (e.g. several served models);
    gauges take the maximum across label sets.  Missing metrics default to
    ``0.0`` for counters and ``None`` for gauges.
    """
    counters: Dict[str, float] = {k: 0.0 for k in _COUNTER_ALIASES}
    gauges: Dict[str, Optional[float]] = {k: None for k in _GAUGE_ALIASES}
    counter_lookup = {alias: key for key, aliases in _COUNTER_ALIASES.items() for alias in aliases}
    gauge_lookup = {alias: key for key, aliases in _GAUGE_ALIASES.items() for alias in aliases}

    for line in metrics_text.split("\n"):
        parsed = _parse_prom_line(line)
        if parsed is None:
            continue
        name, value = parsed
        if name in counter_lookup:
            counters[counter_lookup[name]] += value
        elif name in gauge_lookup:
            key = gauge_lookup[name]
            gauges[key] = value if gauges[key] is None else max(gauges[key], value)

    out: Dict = dict(counters)
    out.update(gauges)
    out["timestamp"] = time.time()
    return out


class MetricsMonitor:
    """Inference-server Prometheus metrics monitoring.

    Queries the server's ``/metrics`` endpoint (vLLM or SGLang names) to
    track, over a measurement window:

    - Prefix cache hit rate (queries and hits deltas)
    - Prefill / decode time deltas
    - **Peak** KV-cache usage, requests waiting and requests running
      (sampled by a background poller during the window)
    - Preemptions that occurred during the window (counter delta)

    The peak gauges and the preemption delta are what the bottleneck
    classifier needs to detect ``kv_capacity`` / ``queue`` regimes; they
    are attached to every result cell by ``_compute_stats``.
    """

    def __init__(self, metrics_url: str, poll_interval: float = 1.0) -> None:
        """Initialize metrics monitor.

        Args:
            metrics_url: Full URL to the ``/metrics`` endpoint,
                e.g. ``"http://localhost:8000/metrics"``.
            poll_interval: Seconds between background samples while the
                monitor is running.
        """
        self.metrics_url = metrics_url
        self.poll_interval = poll_interval
        self.baseline: Optional[Dict] = None
        self.available: bool = False
        self._peaks: Dict[str, Optional[float]] = {k: None for k in _GAUGE_ALIASES}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # Backward-compatible scalar accessors (older callers / tests).
    @property
    def baseline_queries(self) -> float:
        return float((self.baseline or {}).get("cache_queries", 0.0))

    @property
    def baseline_hits(self) -> float:
        return float((self.baseline or {}).get("cache_hits", 0.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_metrics(self) -> Optional[Dict]:
        """Query the metrics endpoint for current statistics.

        Returns:
            Dictionary containing metrics, or ``None`` if unavailable.
        """
        try:
            response = requests.get(self.metrics_url, timeout=2)
            if response.status_code != 200:
                return None
            return parse_metrics_text(response.text)
        except Exception:
            return None

    def _record_peaks(self, stats: Dict) -> None:
        for key in _GAUGE_ALIASES:
            val = stats.get(key)
            if val is None:
                continue
            cur = self._peaks.get(key)
            self._peaks[key] = val if cur is None else max(cur, val)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            stats = self.get_metrics()
            if stats:
                self._record_peaks(stats)
            self._stop_event.wait(self.poll_interval)

    # ------------------------------------------------------------------
    # Public start / stop interface
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Capture baseline metrics and start the background poller.

        Returns:
            ``True`` if metrics are available, ``False`` otherwise.
        """
        stats = self.get_metrics()
        if not stats:
            self.available = False
            return False
        self.baseline = stats
        self.available = True
        self._peaks = {k: None for k in _GAUGE_ALIASES}
        self._record_peaks(stats)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> Optional[Dict]:
        """Stop polling and calculate metrics deltas / peaks since start.

        Returns:
            Dictionary containing cache hit rate, prefill/decode time
            deltas, peak ``gpu_cache_usage_perc`` / ``num_requests_waiting``
            / ``num_requests_running`` and the ``preemptions`` delta, or
            ``None`` if unavailable.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.poll_interval, 2.0) + 2.0)
            self._thread = None

        if not self.available or self.baseline is None:
            return None

        stats = self.get_metrics()
        if not stats:
            return None
        self._record_peaks(stats)

        delta_queries = stats["cache_queries"] - self.baseline["cache_queries"]
        delta_hits = stats["cache_hits"] - self.baseline["cache_hits"]
        hit_rate = (delta_hits / delta_queries * 100) if delta_queries > 0 else 0

        delta_prefill = stats["prefill_time"] - self.baseline["prefill_time"]
        delta_decode = stats["decode_time"] - self.baseline["decode_time"]
        delta_preemptions = max(0.0, stats["preemptions"] - self.baseline["preemptions"])

        out: Dict = {
            "cache_hit_rate": hit_rate,
            "cache_queries_delta": delta_queries,
            "cache_hits_delta": delta_hits,
            "total_cache_queries": stats["cache_queries"],
            "total_cache_hits": stats["cache_hits"],
            "actual_prefill_time": delta_prefill,
            "actual_decode_time": delta_decode,
            "preemptions": int(delta_preemptions),
        }
        for key in _GAUGE_ALIASES:
            if self._peaks.get(key) is not None:
                out[key] = self._peaks[key]
        return out


class GPUMonitor(NvidiaMonitor):
    """Backward-compatible alias for :class:`NvidiaMonitor`.

    The GPU monitoring implementation moved to
    :class:`vllm_benchmark.core.hardware.nvidia.NvidiaMonitor`.  This thin
    subclass preserves the historical ``GPUMonitor`` import path and
    behavior unchanged for existing callers and tests.
    """
