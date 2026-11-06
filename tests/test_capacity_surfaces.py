"""Telemetry columns, KV-capacity parsing, chart, panel, HTML, share and bundle surfaces."""

from __future__ import annotations

from vllm_benchmark.analysis import capacity as cap
from vllm_benchmark.analysis.report.bundle import allowed_numbers, build_bundle
from vllm_benchmark.core.hardware.nvidia import NvidiaMonitor, parse_throttle_reasons
from vllm_benchmark.core.metrics import parse_metrics_text, parse_vllm_cache_config
from vllm_benchmark.reports.capacity_chart import plot_capacity
from vllm_benchmark.reports.html_report import generate_html_report
from vllm_benchmark.reports.share import build_share_markdown
from vllm_benchmark.reports.terminal import render_capacity_panel

from .test_capacity import PRO6000, _cell, _llama8b, _server  # noqa: F401

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_parse_throttle_reasons():
    assert parse_throttle_reasons("0x0000000000000004") == {"power_capped": True, "thermal": False}
    assert parse_throttle_reasons("0x0000000000000060") == {"power_capped": False, "thermal": True}
    assert parse_throttle_reasons("0x0000000000000084") == {"power_capped": True, "thermal": False}
    assert parse_throttle_reasons("0x0") == {"power_capped": False, "thermal": False}
    assert parse_throttle_reasons("[N/A]") == {"power_capped": None, "thermal": None}
    assert parse_throttle_reasons(None) == {"power_capped": None, "thermal": None}


def test_parse_gpu_line_with_optional_columns():
    mon = NvidiaMonitor.__new__(NvidiaMonitor)
    line = "0, 99, 60000, 97887, 70, 299.5, 1950, 14001, 2617, 300.00, 0x0000000000000004"
    s = mon._parse_gpu_line(line)
    assert s["max_sm_clock"] == 2617 and s["power_limit"] == 300.0 and s["power_capped"] is True and s["thermal"] is False
    # Old 8-column form still parses, without the optional keys.
    s8 = mon._parse_gpu_line("0, 72, 34567, 81920, 62, 285.30, 1980, 1215")
    assert "max_sm_clock" not in s8


def test_power_truth_aggregation():
    stats = [
        {"gpu_clock": 1900, "max_sm_clock": 2617, "power_limit": 300.0, "power_capped": True, "thermal": False},
        {"gpu_clock": 2000, "max_sm_clock": 2617, "power_limit": 300.0, "power_capped": True, "thermal": False},
        {"gpu_clock": 2100, "max_sm_clock": 2617, "power_limit": 300.0, "power_capped": False, "thermal": True},
        {"gpu_clock": 2000},  # legacy sample without optional columns
    ]
    out = NvidiaMonitor._power_truth(stats)
    assert out["max_sm_clock"] == 2617 and out["power_limit_w"] == 300.0
    assert abs(out["sustained_clock_pct"] - 2000 / 2617) < 0.01
    assert abs(out["power_cap_frac"] - 2 / 3) < 0.01 and abs(out["thermal_frac"] - 1 / 3) < 0.01
    assert NvidiaMonitor._power_truth([{"gpu_clock": 1}]) == {}


def test_parse_vllm_cache_config():
    text = ('# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n'
            'vllm:cache_config_info{block_size="16",cache_dtype="auto",gpu_memory_utilization="0.9",'
            'num_gpu_blocks="27584",num_cpu_blocks="2048",swap_space="4"} 1.0\n')
    assert parse_vllm_cache_config(text) == 27584 * 16
    assert parse_metrics_text(text)["kv_capacity_tokens"] == 441_344
    assert parse_vllm_cache_config("vllm:num_requests_waiting 0\n") is None
    assert parse_vllm_cache_config('vllm:cache_config_info{block_size="x",num_gpu_blocks="1"} 1\n') is None


def test_parse_sglang_capacity_gauge():
    assert parse_metrics_text('sglang:max_total_num_tokens{m="a"} 123456\n')["kv_capacity_tokens"] == 123456.0


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

def _capacity():
    cells = [_cell(32_000, 1, 60.0, prefill=15_000.0, kv_capacity_tokens=441_344, avg_gpu_clock=2500.0,
                   power_cap_frac=0.0, thermal_frac=0.0, power_limit_w=600.0),
             _cell(32_000, 4, 50.0), _cell(32_000, 16, 20.0), _cell(128_000, 1, 40.0)]
    return cap.analyze_capacity(cells, _llama8b(), PRO6000, _server(), {"gpu_count": 1},
                                context_lengths=[32_000, 128_000], concurrency_levels=[1, 4, 16], output_tokens=500,
                                bottlenecks=[{"critical_batch": 280.0}]), cells


def test_plot_capacity_writes_png(tmp_path):
    capacity, _ = _capacity()
    path = plot_capacity(capacity, str(tmp_path / "cap.png"))
    assert open(path, "rb").read(8).startswith(b"\x89PNG")
    # Empty input still produces a placeholder figure.
    assert open(plot_capacity({}, str(tmp_path / "empty.png")), "rb").read(8).startswith(b"\x89PNG")


def test_render_capacity_panel():
    from rich.console import Console
    capacity, _ = _capacity()
    console = Console(record=True, width=140)
    console.print(render_capacity_panel(capacity))
    text = console.export_text()
    assert "KV capacity" in text and "441,344 tokens (server)" in text
    assert "Suggested serve command" in text and "vllm serve" in text
    assert "Operating point" in text
    console.print(render_capacity_panel(None))


def test_html_and_share_include_capacity(tmp_path):
    capacity, cells = _capacity()
    meta = {"server_info": {"model_name": "m"}, "system_info": {}, "advisory": {"capacity": capacity}}
    html = open(generate_html_report(cells, meta, None, None, str(tmp_path)), encoding="utf-8").read()
    assert "Capacity &amp; Ceilings" in html and "441,344 tokens" in html and "vllm serve" in html
    assert "Roofline: an insightful visual performance model" in html
    md = build_share_markdown(cells, meta)
    assert "**KV capacity:** 441,344 tokens (server)" in md
    assert "**Roofline attainment:**" in md and "**Operating point:**" in md


def test_bundle_carries_capacity_facts():
    capacity, cells = _capacity()
    meta = {"advisory": {"capacity": capacity, "tips": []}}
    b = build_bundle(cells, meta)
    c = b["advisory"]["capacity"]
    assert c["kv_capacity_tokens"] == 441_344 and c["ceilings"][0]["bound_by"] == "bandwidth"
    assert 441344 in allowed_numbers(b)
    assert build_bundle(cells, {"advisory": {}})["advisory"]["capacity"] is None


def test_visualize_results_renders_on_current_matplotlib(tmp_path):
    """Regression: boxplot(labels=) was renamed in matplotlib 3.9 and crashed every charted run."""
    from vllm_benchmark.reports.charts import visualize_results
    cells = []
    for ctx in (32_000, 64_000):
        for users in (1, 4):
            cells.append({"context_length": ctx, "concurrent_users": users, "prompt_type": "classic",
                          "tokens_per_second": 100.0 * users, "avg_latency": 2.0, "latency_p99": 3.0,
                          "ttft_estimate": 0.2, "throughput_per_user": 100.0, "min_latency": 1.5, "max_latency": 3.5,
                          "latency_p50": 2.0, "latency_p90": 2.8, "latency_p95": 2.9})
    path = visualize_results(cells, "m", {}, {}, 500, str(tmp_path))
    assert open(path, "rb").read(8).startswith(b"\x89PNG")
