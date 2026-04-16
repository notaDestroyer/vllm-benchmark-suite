"""Tests for GPU and vLLM metrics monitoring.

Author: amit
"""

from unittest.mock import patch, MagicMock

from vllm_benchmark.core.metrics import GPUMonitor, MetricsMonitor


# ---------------------------------------------------------------------------
# GPUMonitor — GPU count detection
# ---------------------------------------------------------------------------


def test_detect_gpu_count_single():
    """Single-GPU nvidia-smi output returns 1."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = "1\n"
    with patch("subprocess.run", return_value=fake):
        count = GPUMonitor._detect_gpu_count()
    assert count == 1


def test_detect_gpu_count_multi():
    """Multi-GPU nvidia-smi output (count printed per GPU line) returns correct count."""
    fake = MagicMock()
    fake.returncode = 0
    # nvidia-smi --query-gpu=count prints the total on every GPU line
    fake.stdout = "4\n4\n4\n4\n"
    with patch("subprocess.run", return_value=fake):
        count = GPUMonitor._detect_gpu_count()
    assert count == 4


def test_detect_gpu_count_failure():
    """Falls back to 1 when nvidia-smi is unavailable."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        count = GPUMonitor._detect_gpu_count()
    assert count == 1


# ---------------------------------------------------------------------------
# GPUMonitor — single-line parsing
# ---------------------------------------------------------------------------


def test_parse_gpu_line_valid():
    monitor = GPUMonitor.__new__(GPUMonitor)
    line = "0, 72, 34567, 81920, 62, 285.30, 1980, 1215"
    parsed = monitor._parse_gpu_line(line)
    assert parsed is not None
    assert parsed["gpu_index"] == 0
    assert parsed["gpu_util"] == 72.0
    assert parsed["mem_used"] == 34567.0
    assert parsed["mem_total"] == 81920.0
    assert parsed["temperature"] == 62.0
    assert parsed["power_draw"] == 285.30
    assert parsed["gpu_clock"] == 1980.0
    assert parsed["mem_clock"] == 1215.0
    assert "timestamp" in parsed


def test_parse_gpu_line_insufficient_columns():
    monitor = GPUMonitor.__new__(GPUMonitor)
    assert monitor._parse_gpu_line("0, 72, 34567") is None


def test_parse_gpu_line_non_numeric():
    monitor = GPUMonitor.__new__(GPUMonitor)
    assert monitor._parse_gpu_line("0, N/A, 34567, 81920, 62, 285, 1980, 1215") is None


# ---------------------------------------------------------------------------
# GPUMonitor — get_all_gpu_stats
# ---------------------------------------------------------------------------


_TWO_GPU_OUTPUT = (
    "0, 85, 40000, 81920, 64, 290.00, 1980, 1215\n"
    "1, 78, 38000, 81920, 61, 275.00, 1950, 1200\n"
)


def test_get_all_gpu_stats_multi():
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = _TWO_GPU_OUTPUT

    with patch("subprocess.run", return_value=fake):
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_count = 2
        stats = monitor.get_all_gpu_stats()

    assert len(stats) == 2
    assert stats[0]["gpu_index"] == 0
    assert stats[1]["gpu_index"] == 1
    assert stats[0]["gpu_util"] == 85.0
    assert stats[1]["gpu_util"] == 78.0


def test_get_all_gpu_stats_failure():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_count = 1
        stats = monitor.get_all_gpu_stats()
    assert stats == []


# ---------------------------------------------------------------------------
# GPUMonitor — get_gpu_stats (aggregated / backward-compat)
# ---------------------------------------------------------------------------


def test_get_gpu_stats_single_gpu():
    """Single GPU returns stats without gpu_index."""
    one_line = "0, 90, 40000, 81920, 65, 300.00, 2000, 1250\n"
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = one_line

    with patch("subprocess.run", return_value=fake):
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_count = 1
        stats = monitor.get_gpu_stats()

    assert stats is not None
    assert "gpu_index" not in stats
    assert stats["gpu_util"] == 90.0


def test_get_gpu_stats_multi_gpu_averages():
    """Multi-GPU get_gpu_stats returns averaged values."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = _TWO_GPU_OUTPUT

    with patch("subprocess.run", return_value=fake):
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_count = 2
        stats = monitor.get_gpu_stats()

    assert stats is not None
    # (85 + 78) / 2 = 81.5
    assert abs(stats["gpu_util"] - 81.5) < 0.01
    # (40000 + 38000) / 2 = 39000
    assert abs(stats["mem_used"] - 39000.0) < 0.01


# ---------------------------------------------------------------------------
# GPUMonitor — stop() includes gpu_count and per-GPU breakdown
# ---------------------------------------------------------------------------


def test_stop_includes_gpu_count():
    monitor = GPUMonitor.__new__(GPUMonitor)
    monitor.gpu_count = 1
    monitor.monitoring = False
    monitor.thread = None
    monitor.stats = [
        {
            "gpu_util": 80, "mem_used": 30000, "mem_total": 81920,
            "temperature": 60, "power_draw": 250, "gpu_clock": 1900,
            "mem_clock": 1200, "timestamp": 0,
        },
    ]
    monitor.per_gpu_stats = []
    result = monitor.stop()
    assert result is not None
    assert result["gpu_count"] == 1
    assert "per_gpu" not in result  # single GPU — no per_gpu breakdown


def test_stop_multi_gpu_per_gpu_breakdown():
    monitor = GPUMonitor.__new__(GPUMonitor)
    monitor.gpu_count = 2
    monitor.monitoring = False
    monitor.thread = None
    monitor.stats = [
        {
            "gpu_util": 81.5, "mem_used": 39000, "mem_total": 81920,
            "temperature": 62.5, "power_draw": 282.5, "gpu_clock": 1965,
            "mem_clock": 1207.5, "timestamp": 0,
        },
    ]
    monitor.per_gpu_stats = [
        [
            {
                "gpu_index": 0, "gpu_util": 85, "mem_used": 40000,
                "mem_total": 81920, "temperature": 64, "power_draw": 290,
                "gpu_clock": 1980, "mem_clock": 1215, "timestamp": 0,
            },
            {
                "gpu_index": 1, "gpu_util": 78, "mem_used": 38000,
                "mem_total": 81920, "temperature": 61, "power_draw": 275,
                "gpu_clock": 1950, "mem_clock": 1200, "timestamp": 0,
            },
        ],
    ]
    result = monitor.stop()
    assert result is not None
    assert result["gpu_count"] == 2
    assert "per_gpu" in result
    assert len(result["per_gpu"]) == 2
    assert result["per_gpu"][0]["gpu_index"] == 0
    assert result["per_gpu"][1]["gpu_index"] == 1
    assert result["per_gpu"][0]["avg_gpu_util"] == 85.0
    assert result["per_gpu"][1]["avg_gpu_util"] == 78.0


# ---------------------------------------------------------------------------
# GPUMonitor — stop() with no data
# ---------------------------------------------------------------------------


def test_stop_no_data_returns_none():
    monitor = GPUMonitor.__new__(GPUMonitor)
    monitor.gpu_count = 1
    monitor.monitoring = False
    monitor.thread = None
    monitor.stats = []
    monitor.per_gpu_stats = []
    result = monitor.stop()
    assert result is None


# ---------------------------------------------------------------------------
# MetricsMonitor — basic start / stop
# ---------------------------------------------------------------------------


def test_metrics_monitor_unavailable():
    """MetricsMonitor.start() returns False when endpoint is unreachable."""
    with patch("requests.get", side_effect=ConnectionError):
        mon = MetricsMonitor("http://localhost:9999/metrics")
        assert mon.start() is False
        assert mon.available is False


def test_metrics_monitor_stop_without_start():
    mon = MetricsMonitor("http://localhost:9999/metrics")
    assert mon.stop() is None
