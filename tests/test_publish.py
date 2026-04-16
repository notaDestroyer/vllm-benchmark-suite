"""Tests for result publishing and leaderboard generation.

Uses mock data only -- no real files or servers required.
"""

import json
from types import SimpleNamespace

from vllm_benchmark.analysis.publish import (
    create_result_entry,
    format_leaderboard_row,
    generate_leaderboard_md,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bench_result(**overrides):
    base = {
        "tokens_per_second": 800.0,
        "avg_latency": 1.2,
        "ttft_estimate": 0.09,
        "context_length": 32000,
        "concurrent_users": 4,
        "cost_per_1m_tokens": 0.0312,
    }
    base.update(overrides)
    return base


def _make_metadata():
    return {
        "system_info": {
            "gpu_name": "NVIDIA H100",
            "gpu_count": 1,
            "total_vram_gb": 80,
            "cpu_model": "AMD EPYC 9654",
            "total_ram_gb": 512,
        },
        "server_info": {
            "model_name": "meta-llama/Llama-3-70B",
            "version": "0.4.0",
            "quantization": "AWQ",
            "tensor_parallel_size": 1,
        },
    }


def _make_entry():
    """Full round-trip: create_result_entry -> entry dict."""
    results = [
        _make_bench_result(tokens_per_second=800),
        _make_bench_result(tokens_per_second=1200, concurrent_users=8),
    ]
    return create_result_entry(results, _make_metadata())


# ---------------------------------------------------------------------------
# create_result_entry
# ---------------------------------------------------------------------------


def test_create_result_entry_structure():
    entry = _make_entry()
    assert "hardware" in entry
    assert "software" in entry
    assert "results" in entry
    assert "test_matrix" in entry
    assert "fingerprint" in entry
    assert "submitted_at" in entry


def test_create_result_entry_peak_throughput():
    entry = _make_entry()
    assert entry["results"]["peak_throughput_tps"] == 1200.0


def test_create_result_entry_hardware_info():
    entry = _make_entry()
    assert entry["hardware"]["gpu"] == "NVIDIA H100"
    assert entry["hardware"]["gpu_count"] == 1


def test_create_result_entry_with_score():
    score = SimpleNamespace(
        overall=4500,
        grade="A",
        throughput=5000,
        latency=4000,
        efficiency=4500,
        energy=4200,
        consistency=4800,
    )
    results = [_make_bench_result()]
    entry = create_result_entry(results, _make_metadata(), score=score)
    assert "score" in entry
    assert entry["score"]["overall"] == 4500
    assert entry["score"]["grade"] == "A"


def test_create_result_entry_empty_results():
    entry = create_result_entry([], _make_metadata())
    assert entry == {}


def test_create_result_entry_no_tps():
    """Results without tokens_per_second should be skipped, returning empty."""
    entry = create_result_entry([{"avg_latency": 2.0}], _make_metadata())
    assert entry == {}


def test_create_result_entry_cost_field():
    entry = _make_entry()
    assert entry["results"]["cost_per_1m_tokens_usd"] is not None
    assert entry["results"]["cost_per_1m_tokens_usd"] > 0


def test_create_result_entry_fingerprint_is_deterministic():
    e1 = _make_entry()
    e2 = _make_entry()
    assert e1["fingerprint"] == e2["fingerprint"]


# ---------------------------------------------------------------------------
# format_leaderboard_row
# ---------------------------------------------------------------------------


def test_format_leaderboard_row_pipe_separated():
    entry = _make_entry()
    row = format_leaderboard_row(entry)
    assert row.startswith("|")
    assert row.endswith("|")
    # Should contain GPU, model, throughput, TTFT, cost columns
    assert "NVIDIA H100" in row
    assert "Llama-3-70B" in row


def test_format_leaderboard_row_truncates_long_model():
    entry = _make_entry()
    entry["software"]["model"] = "a" * 50  # longer than 30 chars
    row = format_leaderboard_row(entry)
    assert "..." in row


def test_format_leaderboard_row_no_cost():
    entry = _make_entry()
    entry["results"]["cost_per_1m_tokens_usd"] = None
    row = format_leaderboard_row(entry)
    assert "N/A" in row


def test_format_leaderboard_row_no_score():
    entry = _make_entry()
    # No "score" key at all
    entry.pop("score", None)
    row = format_leaderboard_row(entry)
    assert "- (0)" in row


# ---------------------------------------------------------------------------
# generate_leaderboard_md
# ---------------------------------------------------------------------------


def test_generate_leaderboard_md_empty(tmp_path):
    md = generate_leaderboard_md(str(tmp_path))
    assert "No benchmark results found" in md


def test_generate_leaderboard_md_with_files(tmp_path):
    entry = _make_entry()
    path = tmp_path / "result_test_model_20260101_000000.json"
    path.write_text(json.dumps(entry))

    md = generate_leaderboard_md(str(tmp_path))
    assert "# vLLM Benchmark Leaderboard" in md
    assert "| GPU |" in md
    assert "NVIDIA H100" in md


def test_generate_leaderboard_md_sorted_by_throughput(tmp_path):
    slow = _make_entry()
    slow["results"]["peak_throughput_tps"] = 500
    fast = _make_entry()
    fast["results"]["peak_throughput_tps"] = 2000

    (tmp_path / "result_slow_20260101_000000.json").write_text(json.dumps(slow))
    (tmp_path / "result_fast_20260101_000001.json").write_text(json.dumps(fast))

    md = generate_leaderboard_md(str(tmp_path))
    lines = md.splitlines()
    data_rows = [row for row in lines if row.startswith("|") and "GPU" not in row and "---" not in row]
    assert len(data_rows) == 2
    # First row should have the higher throughput
    assert "2,000" in data_rows[0]


def test_generate_leaderboard_md_skips_bad_json(tmp_path):
    (tmp_path / "result_bad_20260101_000000.json").write_text("{bad json")
    md = generate_leaderboard_md(str(tmp_path))
    assert "No benchmark results found" in md
