"""publish.py must not crash on partial cells and must read the
``tensor_parallel`` key that ServerInfo actually emits."""

from __future__ import annotations

from datetime import datetime, timezone

from vllm_benchmark.analysis.publish import create_result_entry, format_leaderboard_row, generate_leaderboard_md


def _meta(**server):
    return {"system_info": {"gpu_name": "NVIDIA H100", "gpu_count": 1},
            "server_info": {"model_name": "m", "backend": "vllm", **server}}


def test_tensor_parallel_read_from_serverinfo_key():
    e = create_result_entry([{"tokens_per_second": 10.0, "avg_latency": 1.0}], _meta(tensor_parallel=4))
    assert e["software"]["tensor_parallel"] == 4
    assert e["software"]["backend"] == "vllm"


def test_tensor_parallel_legacy_key_still_accepted():
    e = create_result_entry([{"tokens_per_second": 10.0, "avg_latency": 1.0}], _meta(tensor_parallel_size=2))
    assert e["software"]["tensor_parallel"] == 2


def test_missing_avg_latency_does_not_raise():
    e = create_result_entry([{"tokens_per_second": 10.0}], _meta())
    assert e["results"]["best_latency_s"] is None
    assert e["results"]["peak_throughput_tps"] == 10.0


def test_none_ttft_does_not_raise():
    cells = [{"tokens_per_second": 10.0, "avg_latency": 1.0, "ttft_estimate": None},
             {"tokens_per_second": 12.0, "avg_latency": 1.0, "ttft_estimate": 0.25}]
    e = create_result_entry(cells, _meta())
    assert e["results"]["best_ttft_ms"] == 250.0


def test_all_ttft_missing_gives_none_not_999s():
    e = create_result_entry([{"tokens_per_second": 10.0, "avg_latency": 1.0}], _meta())
    assert e["results"]["best_ttft_ms"] is None


def test_leaderboard_row_handles_none_ttft():
    e = create_result_entry([{"tokens_per_second": 10.0, "avg_latency": 1.0}], _meta())
    row = format_leaderboard_row(e)
    assert "| N/A |" in row


def test_leaderboard_timestamp_is_utc(tmp_path, monkeypatch):
    (tmp_path / "result_x.json").write_text('{"results": {"peak_throughput_tps": 1}}')
    md = generate_leaderboard_md(str(tmp_path))
    stamp = md.split("*Generated ")[1].split(" UTC*")[0]
    parsed = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 120
