"""Tests for PR5 charts: bottleneck map/grid and roofline.

``bottleneck_grid`` correctness on synthetic verdicts, plus smoke tests
that ``plot_bottleneck_map``, ``plot_roofline`` and ``plot_quant_compare``
produce a PNG without raising on both normal and empty input.

Author: amit
"""

from __future__ import annotations

from pathlib import Path

from vllm_benchmark.reports.charts import (
    bottleneck_grid,
    plot_bottleneck_map,
    plot_quant_compare,
    plot_roofline,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _verdict(ctx: int, users: int, primary: str, **extra) -> dict:
    v = {"cell": [ctx, users, "classic"], "primary": primary}
    v.update(extra)
    return v


def _verdicts() -> list[dict]:
    return [
        _verdict(32000, 1, "decode_weight_bandwidth", mbu=0.4, critical_batch=16),
        _verdict(32000, 8, "decode_weight_bandwidth", critical_batch=16),
        _verdict(32000, 32, "decode_compute", critical_batch=16),
        _verdict(64000, 1, "decode_kv_bandwidth", critical_batch=16),
        _verdict(64000, 8, "kv_capacity", critical_batch=16),
        _verdict(64000, 32, "queue", critical_batch=16),
    ]


def test_bottleneck_grid_dimensions_and_values() -> None:
    contexts, concurrencies, grid = bottleneck_grid(_verdicts())
    assert contexts == [32000, 64000]
    assert concurrencies == [1, 8, 32]
    assert len(grid) == 2
    assert len(grid[0]) == 3
    # Row 0 = ctx 32000 across users 1/8/32.
    assert grid[0] == ["decode_weight_bandwidth", "decode_weight_bandwidth", "decode_compute"]
    assert grid[1] == ["decode_kv_bandwidth", "kv_capacity", "queue"]


def test_bottleneck_grid_tie_breaks_by_precedence() -> None:
    # Same cell, two different primaries, one each -> precedence wins.
    verdicts = [
        _verdict(32000, 1, "queue"),
        _verdict(32000, 1, "kv_capacity"),
    ]
    _, _, grid = bottleneck_grid(verdicts)
    # kv_capacity precedes queue in the color/precedence ordering.
    assert grid[0][0] == "kv_capacity"


def test_bottleneck_grid_empty() -> None:
    contexts, concurrencies, grid = bottleneck_grid([])
    assert contexts == []
    assert concurrencies == []
    assert grid == []


def test_plot_bottleneck_map_png(tmp_path: Path) -> None:
    out = tmp_path / "bn.png"
    path = plot_bottleneck_map(_verdicts(), str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC


def test_plot_bottleneck_map_empty(tmp_path: Path) -> None:
    out = tmp_path / "bn_empty.png"
    path = plot_bottleneck_map([], str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC


def test_plot_roofline_png(tmp_path: Path) -> None:
    out = tmp_path / "roof.png"
    results = [
        {"context_length": 32000, "concurrent_users": 1,
         "prefill_tps": 9000.0, "tokens_per_second": 120.0},
    ]
    profile = {"active_params": 8_000_000_000}
    gpu_spec = {"hbm_bandwidth_gbps": 3350.0, "peak_flops_tflops": {"bf16": 989.0}}
    path = plot_roofline(results, profile, gpu_spec, str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC


def test_plot_roofline_empty(tmp_path: Path) -> None:
    out = tmp_path / "roof_empty.png"
    path = plot_roofline([], None, None, str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC


def test_plot_quant_compare_png(tmp_path: Path) -> None:
    out = tmp_path / "quant.png"
    comparison = {
        "runs": [
            {"label": "m [fp8]", "tokens_per_second_mean": 2000.0,
             "peak_mem_mean": 40000.0, "quality": 92.0},
            {"label": "m [int4]", "tokens_per_second_mean": 900.0,
             "peak_mem_mean": 20000.0, "quality": 80.0},
        ],
    }
    path = plot_quant_compare(comparison, str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC


def test_plot_quant_compare_empty(tmp_path: Path) -> None:
    out = tmp_path / "quant_empty.png"
    path = plot_quant_compare({}, str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC
