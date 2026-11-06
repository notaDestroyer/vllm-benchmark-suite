"""Fixes ported from PR #4 (tattrongvu): None-safe save_result, --tokenizer, warn-once."""

from __future__ import annotations

import io
import json
import sys

from vllm_benchmark import __version__, cli
from vllm_benchmark.analysis.publish import create_result_entry, save_result
from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core import async_engine
from vllm_benchmark.core.hardware import nvidia


def test_save_result_with_none_gpu_and_model(tmp_path):
    entry = create_result_entry([{"tokens_per_second": 1.0, "avg_latency": 1.0}],
                                {"system_info": {"gpu_name": None}, "server_info": {"model_name": None}})
    assert entry["hardware"]["gpu"] is None
    path = save_result(entry, str(tmp_path))
    assert path.endswith(".json") and "result_unknown_unknown_" in path
    json.load(open(path))


def test_save_result_missing_sections(tmp_path):
    path = save_result({"results": {}}, str(tmp_path))
    assert "result_unknown_unknown_" in path


def test_tokenizer_id_precedence():
    assert BenchmarkConfig().tokenizer_id("org/m") == "org/m"
    assert BenchmarkConfig(model_name="cfg/m").tokenizer_id(None) == "cfg/m"
    assert BenchmarkConfig(tokenizer="/local/tok", model_name="cfg/m").tokenizer_id("org/m") == "/local/tok"
    assert BenchmarkConfig().tokenizer_id(None) == ""


def test_cli_tokenizer_flag_and_env(monkeypatch):
    args = cli.build_parser().parse_args(["--tokenizer", "/tok"])
    assert args.tokenizer == "/tok"
    monkeypatch.setenv("VLLM_BENCH_TOKENIZER", "/env-tok")
    assert cli.build_parser().parse_args([]).tokenizer is None  # env is applied in main(), not argparse


def test_engine_generates_and_counts_with_override(monkeypatch):
    seen: dict[str, str] = {}

    def _gen(_type):
        def _g(n, tok):
            seen["gen"] = tok
            return "p"
        return _g

    monkeypatch.setattr(async_engine, "get_prompt_generator", _gen)
    monkeypatch.setattr(async_engine, "count_tokens", lambda text, tok="": seen.__setitem__("count", tok) or 1)
    cfg = BenchmarkConfig(tokenizer="/local/tok")
    prompts, _, _ = async_engine.resolve_prompts(cfg, "classic", 10, "served/model")
    async_engine._mean_prompt_tokens(prompts, cfg.tokenizer_id("served/model"))
    assert seen == {"gen": "/local/tok", "count": "/local/tok"}


def test_gpu_monitoring_warning_printed_once(monkeypatch):
    monkeypatch.setattr(nvidia, "_gpu_monitoring_warned", False)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    for _ in range(5):
        nvidia._warn_gpu_monitoring_once(FileNotFoundError("nvidia-smi"))
    assert buf.getvalue().count("nvidia-smi not found") == 1


def test_get_all_gpu_stats_warns_once_on_missing_binary(monkeypatch):
    monkeypatch.setattr(nvidia, "_gpu_monitoring_warned", False)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    def _raise(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(nvidia.subprocess, "run", _raise)
    mon = nvidia.NvidiaMonitor(poll_interval=0.01)
    for _ in range(10):
        assert mon.get_all_gpu_stats() == []
    assert buf.getvalue().count("[WARNING]") == 1


def test_version_matches_pyproject():
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert __version__ == re.search(r'^version = "([^"]+)"', text, re.M).group(1)
