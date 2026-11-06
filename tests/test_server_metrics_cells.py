"""Server-side metrics reach result cells so kv_capacity / queue /
interconnect bottleneck classes are reachable from a real run.

Before this fix ``classify_run`` read ``gpu_cache_usage_perc``,
``num_requests_waiting``, ``preemptions`` and ``tensor_parallel`` from cells
but nothing ever wrote them there.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from vllm_benchmark.analysis.bottleneck import classify_run
from vllm_benchmark.analysis.model_intel import ModelProfile
from vllm_benchmark.core.async_engine import _compute_stats
from vllm_benchmark.core.backends.base import ServerInfo
from vllm_benchmark.core.metrics import MetricsMonitor, _parse_prom_line, parse_metrics_text

VLLM_METRICS = """
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="m"} 0.97
vllm:num_requests_waiting{model_name="m"} 3.0
vllm:num_requests_running{model_name="m"} 16.0
vllm:num_preemptions_total{model_name="m"} 5.0
vllm:prefix_cache_queries_total{model_name="m"} 1000.0
vllm:prefix_cache_hits_total{model_name="m"} 800.0
vllm:request_prefill_time_seconds_sum{model_name="m"} 12.5
vllm:request_decode_time_seconds_sum{model_name="m"} 40.0
"""

SGLANG_METRICS = """
sglang:token_usage{model_name="m"} 0.5
sglang:num_queue_reqs{model_name="m"} 2.0
sglang:num_running_reqs{model_name="m"} 8.0
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_prom_line_forms():
    assert _parse_prom_line('vllm:x{a="b"} 1.5') == ("vllm:x", 1.5)
    assert _parse_prom_line("vllm:y 2") == ("vllm:y", 2.0)
    assert _parse_prom_line("# comment") is None
    assert _parse_prom_line("") is None
    assert _parse_prom_line("vllm:z notanumber") is None


def test_parse_metrics_text_vllm():
    m = parse_metrics_text(VLLM_METRICS)
    assert m["gpu_cache_usage_perc"] == 0.97
    assert m["num_requests_waiting"] == 3.0
    assert m["num_requests_running"] == 16.0
    assert m["preemptions"] == 5.0
    assert m["cache_queries"] == 1000.0 and m["cache_hits"] == 800.0
    assert m["prefill_time"] == 12.5 and m["decode_time"] == 40.0


def test_parse_metrics_text_sglang_aliases():
    m = parse_metrics_text(SGLANG_METRICS)
    assert m["gpu_cache_usage_perc"] == 0.5
    assert m["num_requests_waiting"] == 2.0
    assert m["num_requests_running"] == 8.0
    assert m["preemptions"] == 0.0


def test_parse_metrics_text_multi_label_sets():
    text = 'vllm:gpu_cache_usage_perc{m="a"} 0.2\nvllm:gpu_cache_usage_perc{m="b"} 0.8\n' \
           'vllm:num_preemptions_total{m="a"} 1\nvllm:num_preemptions_total{m="b"} 2\n'
    m = parse_metrics_text(text)
    assert m["gpu_cache_usage_perc"] == 0.8  # gauges: max
    assert m["preemptions"] == 3.0  # counters: sum


def test_parse_metrics_text_missing_gauges_are_none():
    m = parse_metrics_text("")
    assert m["gpu_cache_usage_perc"] is None
    assert m["num_requests_waiting"] is None


# ---------------------------------------------------------------------------
# Monitor window: peaks + deltas
# ---------------------------------------------------------------------------

def _resp(text: str):
    return SimpleNamespace(status_code=200, text=text)


def test_monitor_reports_peaks_and_preemption_delta():
    baseline = VLLM_METRICS.replace("0.97", "0.10").replace("3.0\n", "0.0\n").replace("5.0\n", "2.0\n")
    peak = VLLM_METRICS  # 0.97 usage, 3 waiting, 5 preemptions
    final = VLLM_METRICS.replace("0.97", "0.05").replace("3.0\n", "0.0\n")
    responses = iter([_resp(baseline), _resp(peak), _resp(final)] + [_resp(final)] * 10)
    with patch("requests.get", side_effect=lambda *a, **k: next(responses)):
        mon = MetricsMonitor("http://x/metrics", poll_interval=0.01)
        assert mon.start() is True
        # Let the poller sample the peak at least once.
        import time
        deadline = time.time() + 2
        while mon._peaks.get("gpu_cache_usage_perc", 0) < 0.97 and time.time() < deadline:
            time.sleep(0.01)
        out = mon.stop()
    assert out is not None
    assert out["gpu_cache_usage_perc"] == 0.97
    assert out["num_requests_waiting"] == 3.0
    assert out["num_requests_running"] == 16.0
    assert out["preemptions"] == 3  # 5 - 2
    assert out["cache_hit_rate"] == 0  # no delta in queries


def test_monitor_stop_without_start_is_none():
    assert MetricsMonitor("http://x/metrics").stop() is None


def test_monitor_unavailable_endpoint():
    with patch("requests.get", side_effect=ConnectionError):
        mon = MetricsMonitor("http://x/metrics")
        assert mon.start() is False
        assert mon.stop() is None


# ---------------------------------------------------------------------------
# Cells carry the metrics; classifier reaches the gated classes
# ---------------------------------------------------------------------------

def _reqs(n):
    return [{"request_id": i, "success": True, "duration": 2.0, "prompt_tokens": 1000,
             "completion_tokens": 100, "ttft": 0.2, "prefill_tps": 5000.0, "decode_tps": 50.0,
             "pp_tg_source": "streaming"} for i in range(n)]


def _profile():
    return ModelProfile(name="dense", is_moe=False, total_params=8e9, active_params=8e9,
                        kv_bytes_per_token=131072, confidence="confirmed")


_H100 = {"hbm_bandwidth_gbps": 3350, "peak_flops_tflops": {"bf16": 989}}


def test_compute_stats_attaches_server_metrics_to_cell():
    metrics = {"cache_hit_rate": 50.0, "actual_prefill_time": 1.0, "actual_decode_time": 2.0,
               "gpu_cache_usage_perc": 0.98, "num_requests_waiting": 4.0, "preemptions": 2}
    cell = _compute_stats(_reqs(4), 2.2, 32_000, 4, "classic", 1000, None, metrics, None)
    assert cell["gpu_cache_usage_perc"] == 0.98
    assert cell["num_requests_waiting"] == 4.0
    assert cell["preemptions"] == 2


def test_classify_run_reaches_kv_capacity_from_cell_metrics():
    cell = _compute_stats(_reqs(4), 2.2, 32_000, 4, "classic", 1000, None,
                          {"gpu_cache_usage_perc": 0.99, "preemptions": 0}, None)
    [v] = classify_run([cell], _profile(), _H100)
    assert v.primary == "kv_capacity"


def test_classify_run_reaches_queue_from_waiting_requests():
    cell = _compute_stats(_reqs(4), 2.2, 32_000, 4, "classic", 1000, None,
                          {"gpu_cache_usage_perc": 0.3, "num_requests_waiting": 2.0, "preemptions": 0}, None)
    [v] = classify_run([cell], _profile(), _H100)
    assert v.primary == "queue"
    assert v.confidence == "high"


def test_classify_run_tensor_parallel_from_server_info():
    # Saturating scaling below B* with TP>1 -> interconnect.
    cells = []
    for users, tps in ((1, 100.0), (4, 110.0), (8, 112.0)):
        c = _compute_stats(_reqs(users), 2.2, 32_000, users, "classic", 1000, None, None, None)
        c["tokens_per_second"] = tps
        cells.append(c)
    info = ServerInfo(backend="vllm", tensor_parallel=4)
    verdicts = classify_run(cells, _profile(), _H100, server_info=info)
    assert any(v.primary == "interconnect" for v in verdicts)
    # Without TP information the same cells never claim interconnect.
    verdicts_no_tp = classify_run(cells, _profile(), _H100)
    assert all(v.primary != "interconnect" for v in verdicts_no_tp)


def test_classify_run_tensor_parallel_from_dict_and_cell():
    c = _compute_stats(_reqs(4), 2.2, 32_000, 4, "classic", 1000, None, None, None)
    # explicit kwarg, dict server_info and cell key are all accepted
    classify_run([c], _profile(), _H100, tensor_parallel=2)
    classify_run([c], _profile(), _H100, server_info={"tensor_parallel": 2})
    c["tensor_parallel"] = 2
    classify_run([c], _profile(), _H100)
