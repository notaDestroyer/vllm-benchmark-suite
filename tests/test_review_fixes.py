"""Regression tests for the findings from the PR review."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_benchmark import cli
from vllm_benchmark.analysis.head_to_head import head_to_head
from vllm_benchmark.analysis.model_intel import match_gpu_spec
from vllm_benchmark.analysis.publish import create_result_entry
from vllm_benchmark.analysis.scoring import VLLMScore
from vllm_benchmark.analysis.statistics import is_finite_number, is_number
from vllm_benchmark.config import GPU_BASELINES, GPU_PRICING, BenchmarkConfig
from vllm_benchmark.core import async_engine
from vllm_benchmark.core.backends.base import ServerInfo
from vllm_benchmark.core.hardware.gpu_names import match_gpu_key
from vllm_benchmark.core.metrics import MetricsMonitor, _parse_prom_line, parse_metrics_text

# ---------------------------------------------------------------------------
# 1. GPU matcher: family prefix alone never matches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, spec_key", [
    ("NVIDIA GeForce RTX 3090", "RTX 3090"),
    ("NVIDIA GeForce RTX 4080", "RTX 4080"),
    ("NVIDIA RTX A6000", "RTX A6000"),
    ("Quadro RTX 6000", None),             # Turing-era card: never the Ada or Blackwell entry
    ("NVIDIA GeForce RTX 3080 Ti", "RTX 3080 Ti"),
    ("NVIDIA A40", "A40"),
    ("NVIDIA A30", "A30"),
])
def test_family_prefix_never_claims_a_different_card(name, spec_key):
    # None of these are RTX 4090s: no score baseline, no price ...
    assert match_gpu_key(name, GPU_BASELINES.keys()) is None
    assert match_gpu_key(name, GPU_PRICING.keys()) is None
    assert VLLMScore._match_gpu_baseline(name) == (None, None)
    # ... and the roofline table resolves them to their own entry or to nothing.
    spec = match_gpu_spec(name)
    assert (spec["_matched_key"] if spec else None) == spec_key


@pytest.mark.parametrize("name, expected", [
    ("NVIDIA GeForce RTX 4090", "RTX 4090"),
    ("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada"),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000"),
    ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
    ("NVIDIA H100 80GB HBM3", "H100"),
    ("NVIDIA H100", "H100"),            # no memory token: A100-80GB-style keys still allowed to miss
])
def test_known_cards_still_match(name, expected):
    assert match_gpu_key(name, GPU_BASELINES.keys()) == expected


def test_memory_token_missing_is_penalty_not_rejection():
    # "NVIDIA A100" (no size) still resolves to an A100 key rather than nothing.
    assert match_gpu_key("NVIDIA A100", GPU_PRICING.keys()) in ("A100-80GB", "A100-40GB")


# ---------------------------------------------------------------------------
# 2. --prompts-file honoured in --traffic mode
# ---------------------------------------------------------------------------

def test_custom_prompt_generator_cycles_file(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(3)) + "\n")
    async_engine._custom_prompt_cache.clear()
    gen = async_engine.custom_prompt_generator(BenchmarkConfig(prompts_file=str(path)))
    assert [gen(32_000, "m") for _ in range(5)] == ["p0", "p1", "p2", "p0", "p1"]


def test_custom_prompt_generator_requires_file():
    with pytest.raises(ValueError):
        async_engine.custom_prompt_generator(BenchmarkConfig())


def test_traffic_mode_uses_custom_prompts(tmp_path, monkeypatch):
    """Drive cli.main() with --traffic and assert the simulator got the file's prompts."""
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"prompt": "FROM-FILE"}) + "\n")
    seen: list[str] = []

    def _fake_poisson(cfg, model, endpoint, gen, timeout, *a, **k):
        seen.append(gen(cfg.initial_context_tokens, model))
        from vllm_benchmark.core.traffic import TrafficResult
        return TrafficResult()

    monkeypatch.setattr("vllm_benchmark.core.traffic.run_poisson_traffic", _fake_poisson)
    monkeypatch.setattr("vllm_benchmark.core.traffic.format_traffic_report", lambda *_a, **_k: "report")
    monkeypatch.setattr(cli.SystemInfo, "get_system_info", staticmethod(lambda: {"gpu_name": None}))

    class _B:
        name = "vllm"
        def server_info(self, cfg):
            return ServerInfo(backend="vllm", model_name="m")
    monkeypatch.setattr("vllm_benchmark.core.backends.detect_backend", lambda *_a, **_k: _B())
    async_engine._custom_prompt_cache.clear()

    argv = ["--prompts-file", str(path), "--traffic", "poisson", "-y", "--traffic-duration", "0"]
    with patch("sys.argv", ["vllm-bench", *argv]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0
    assert seen == ["FROM-FILE"]


# ---------------------------------------------------------------------------
# 3. Head-to-head: per-request tps is its own metric; direction guard
# ---------------------------------------------------------------------------

def _cell(tps_point: float, per_req: list[float]) -> dict:
    return {"context_length": 1, "concurrent_users": 16, "prompt_type": "c",
            "tokens_per_second": tps_point, "request_tps_samples": per_req}


def test_failed_requests_cannot_win_aggregate_throughput():
    # A: all 16 succeed at ~50 tok/s (aggregate 800).  B: 8 succeed at ~60 (aggregate 480).
    a = _cell(800.0, [50.0 + (i % 4) * 0.5 for i in range(16)])
    b = _cell(480.0, [60.0 + (i % 4) * 0.5 for i in range(8)])
    out = head_to_head([a], {}, [b], {})
    m = out["cells"][0]["metrics"]
    # B is faster per request ...
    assert m["request_tps"]["winner"] == "b"
    # ... but the aggregate metric is a point-only tie, never a B win.
    assert m["tokens_per_second"]["tie"]
    assert m["tokens_per_second"]["winner"] is None
    assert m["tokens_per_second"]["point_a"] == 800.0


def test_direction_conflict_withholds_winner():
    # Cross-iteration samples say B is higher, but the cell point estimates say A.
    a = {"context_length": 1, "concurrent_users": 1, "prompt_type": "c",
         "tokens_per_second": 900.0, "tokens_per_second_samples": [100.0, 101.0, 99.0, 100.5, 99.5]}
    b = {"context_length": 1, "concurrent_users": 1, "prompt_type": "c",
         "tokens_per_second": 400.0, "tokens_per_second_samples": [200.0, 201.0, 199.0, 200.5, 199.5]}
    out = head_to_head([a], {}, [b], {})
    v = out["cells"][0]["metrics"]["tokens_per_second"]
    assert v["tie"] and v["winner"] is None
    assert v["tie_reason"] == "direction_conflict"


def test_agreeing_direction_still_wins():
    a = {"context_length": 1, "concurrent_users": 1, "prompt_type": "c",
         "tokens_per_second": 100.0, "tokens_per_second_samples": [100.0, 101.0, 99.0, 100.5, 99.5]}
    b = {"context_length": 1, "concurrent_users": 1, "prompt_type": "c",
         "tokens_per_second": 200.0, "tokens_per_second_samples": [200.0, 201.0, 199.0, 200.5, 199.5]}
    v = head_to_head([a], {}, [b], {})["cells"][0]["metrics"]["tokens_per_second"]
    assert v["winner"] == "b" and v["tie_reason"] is None


# ---------------------------------------------------------------------------
# 4. Cost multiplier follows the server's parallelism
# ---------------------------------------------------------------------------

def test_billable_gpus_prefers_server_parallelism():
    host = {"gpu_count": 8}
    assert cli._billable_gpus(ServerInfo(backend="vllm", tensor_parallel=1), host) == 1
    assert cli._billable_gpus(ServerInfo(backend="vllm", tensor_parallel=2, pipeline_parallel=2), host) == 4
    assert cli._billable_gpus(ServerInfo(backend="vllm", pipeline_parallel=2), host) == 2
    # Unknown parallelism: fall back to the host count, then to 1.
    assert cli._billable_gpus(ServerInfo(backend="vllm"), host) == 8
    assert cli._billable_gpus(ServerInfo(backend="vllm"), {}) == 1
    assert cli._billable_gpus(ServerInfo(backend="vllm"), {"gpu_count": "x"}) == 1


# ---------------------------------------------------------------------------
# 5. Prometheus parsing with timestamps / awkward labels
# ---------------------------------------------------------------------------

def test_parse_prom_line_with_timestamp_suffix():
    assert _parse_prom_line('vllm:gpu_cache_usage_perc{m="a"} 0.42 1700000000000') == ("vllm:gpu_cache_usage_perc", 0.42)
    assert _parse_prom_line("vllm:num_requests_waiting 3 1700000000000") == ("vllm:num_requests_waiting", 3.0)


def test_parse_prom_line_labels_with_spaces_and_braces():
    line = 'vllm:num_requests_running{model_name="my model {v2}",engine="0"} 7'
    assert _parse_prom_line(line) == ("vllm:num_requests_running", 7.0)
    assert _parse_prom_line("broken{unterminated 1") is None
    assert _parse_prom_line("name_only") is None


def test_parse_metrics_text_timestamped_payload_does_not_trip_kv_capacity():
    text = 'vllm:gpu_cache_usage_perc{m="a"} 0.10 1700000000000\n'
    assert parse_metrics_text(text)["gpu_cache_usage_perc"] == 0.10


# ---------------------------------------------------------------------------
# 6. Poller cannot leak into a later window
# ---------------------------------------------------------------------------

def test_orphaned_poller_writes_only_into_its_own_window():
    gate = threading.Event()
    calls = {"n": 0}

    def _get(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:  # start() #1 baseline
            return SimpleNamespace(status_code=200, text='vllm:num_requests_waiting{m="a"} 0\n')
        if calls["n"] == 2:  # window-1 poller: stalls, then reports a huge value
            gate.wait(5)
            return SimpleNamespace(status_code=200, text='vllm:num_requests_waiting{m="a"} 99\n')
        return SimpleNamespace(status_code=200, text='vllm:num_requests_waiting{m="a"} 0\n')

    mon = MetricsMonitor("http://x/metrics", poll_interval=0.01)
    with patch("requests.get", side_effect=_get):
        assert mon.start() is True
        deadline = time.time() + 2
        while calls["n"] < 2 and time.time() < deadline:
            time.sleep(0.005)
        first_peaks = mon._peaks
        out = mon.stop()  # bounded join; the window-1 poller is still blocked
        assert out is not None
        assert mon._thread is not None and mon._thread.is_alive()
        # Window 2 gets its own peaks dict and event.
        assert mon.start() is True
        second_peaks = mon._peaks
        assert second_peaks is not first_peaks
        gate.set()  # release the orphan: its 99 must not land in window 2
        time.sleep(0.1)
        assert second_peaks.get("num_requests_waiting") == 0.0
        assert first_peaks.get("num_requests_waiting") in (None, 0.0)  # stopped event: not recorded
        mon.stop()


# ---------------------------------------------------------------------------
# 7 + 9. Finite-number predicate is shared and rejects NaN
# ---------------------------------------------------------------------------

def test_is_finite_number_contract():
    assert is_number(1) and is_number(1.5) and is_number(math.nan)
    assert not is_number(True) and not is_number("1") and not is_number(None)
    assert is_finite_number(1.5) and not is_finite_number(math.nan) and not is_finite_number(math.inf)


def test_publish_ignores_nan_latency_and_ttft():
    cells = [
        {"tokens_per_second": 10.0, "avg_latency": math.nan, "ttft_estimate": math.nan},
        {"tokens_per_second": 12.0, "avg_latency": 1.5, "ttft_estimate": 0.2},
    ]
    e = create_result_entry(cells, {"system_info": {}, "server_info": {}})
    assert e["results"]["best_latency_s"] == 1.5
    assert e["results"]["best_ttft_ms"] == 200.0


def test_publish_reuses_statistics_predicate():
    from vllm_benchmark.analysis import publish, statistics
    assert publish.is_finite_number is statistics.is_finite_number


# ---------------------------------------------------------------------------
# 8. Prompt token counts are cached per file
# ---------------------------------------------------------------------------

def test_prompt_token_counts_cached_across_cells(monkeypatch):
    calls: list[str] = []

    def _count(text, model=""):
        calls.append(text)
        return len(text)

    monkeypatch.setattr(async_engine, "count_tokens", _count)
    async_engine._custom_prompt_token_cache.clear()
    prompts = ["aa", "bbbb", "cccccc", "d"]
    # First cell: 2 concurrent -> only two prompts counted.
    assert async_engine._mean_prompt_tokens(prompts, "m", cache_key="f", n_used=2) == 3
    assert calls == ["aa", "bbbb"]
    # Second cell with 3 concurrent: only the new prompt is tokenized.
    async_engine._mean_prompt_tokens(prompts, "m", cache_key="f", n_used=3)
    assert calls == ["aa", "bbbb", "cccccc"]
    # Repeat is free.
    async_engine._mean_prompt_tokens(prompts, "m", cache_key="f", n_used=3)
    assert len(calls) == 3
    # n_used larger than the file counts every prompt exactly once.
    async_engine._mean_prompt_tokens(prompts, "m", cache_key="f", n_used=100)
    assert len(calls) == 4


def test_sustained_mode_counts_only_prompts_sent(monkeypatch, tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": "x" * (i + 1)}) for i in range(50)) + "\n")
    counted: list[str] = []
    monkeypatch.setattr(async_engine, "count_tokens", lambda t, m="": counted.append(t) or len(t))

    async def _fake_sustained(prompts, *_a, **_k):
        return [{"request_id": i, "success": True, "duration": 1.0, "prompt_tokens": 0,
                 "completion_tokens": 5} for i in range(3)]

    class _No:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            return False

        def stop(self, **_k):
            return None

    monkeypatch.setattr(async_engine, "_run_sustained_rps", _fake_sustained)
    monkeypatch.setattr(async_engine, "GPUMonitor", _No)
    monkeypatch.setattr(async_engine, "MetricsMonitor", _No)
    async_engine._custom_prompt_cache.clear()
    async_engine._custom_prompt_token_cache.clear()
    cfg = BenchmarkConfig(prompts_file=str(path))
    stats = asyncio.run(async_engine.run_sustained_benchmark(32_000, 1.0, 0.1, cfg, model_name="m"))
    assert stats is not None
    assert len(counted) == 3  # not all 50


# ---------------------------------------------------------------------------
# 10. The CLI primes the engine cache so the file is parsed once
# ---------------------------------------------------------------------------

def test_prime_custom_prompts_skips_second_parse(tmp_path, monkeypatch):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"prompt": "only"}) + "\n")
    async_engine._custom_prompt_cache.clear()
    async_engine.prime_custom_prompts(str(path), ["only"])
    monkeypatch.setattr(async_engine, "load_custom_prompts", lambda _p: (_ for _ in ()).throw(AssertionError("re-parsed")))
    prompts, label, source = async_engine.resolve_prompts(BenchmarkConfig(prompts_file=str(path)), "classic", 1, "")
    assert prompts == ["only"] and source == "file"
