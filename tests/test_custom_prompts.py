"""Tests that ``--prompts-file`` prompts are actually sent by the engine.

Before this fix ``config.prompts_file`` was parsed and stored but never read,
so custom prompts were silently replaced by synthetic ones.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core import async_engine
from vllm_benchmark.core.async_engine import (
    CUSTOM_PROMPT_TYPE,
    _prompt_for,
    resolve_prompts,
)


@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "prompts.jsonl"
    lines = [{"prompt": f"custom prompt number {i}"} for i in range(3)]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return str(path)


@pytest.fixture(autouse=True)
def _clear_cache():
    async_engine._custom_prompt_cache.clear()
    yield
    async_engine._custom_prompt_cache.clear()


# ---------------------------------------------------------------------------
# _prompt_for
# ---------------------------------------------------------------------------

def test_prompt_for_string_is_shared():
    assert _prompt_for("hello", 0) == "hello"
    assert _prompt_for("hello", 17) == "hello"


def test_prompt_for_list_cycles():
    prompts = ["a", "b", "c"]
    assert [_prompt_for(prompts, i) for i in range(7)] == ["a", "b", "c", "a", "b", "c", "a"]


def test_prompt_for_empty_list_raises():
    with pytest.raises(ValueError):
        _prompt_for([], 0)


# ---------------------------------------------------------------------------
# resolve_prompts
# ---------------------------------------------------------------------------

def test_resolve_prompts_uses_file(prompts_file):
    cfg = BenchmarkConfig(prompts_file=prompts_file)
    prompts, label, source = resolve_prompts(cfg, "classic", 32_000, "")
    assert prompts == [f"custom prompt number {i}" for i in range(3)]
    assert label == CUSTOM_PROMPT_TYPE
    assert source == "file"


def test_resolve_prompts_caches_file(prompts_file, monkeypatch):
    cfg = BenchmarkConfig(prompts_file=prompts_file)
    calls = []
    real = async_engine.load_custom_prompts

    def _counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(async_engine, "load_custom_prompts", _counting)
    resolve_prompts(cfg, "classic", 1, "")
    resolve_prompts(cfg, "random", 2, "")
    assert calls == [prompts_file]


def test_resolve_prompts_generated_when_no_file(monkeypatch):
    cfg = BenchmarkConfig()
    monkeypatch.setattr(async_engine, "get_prompt_generator", lambda _t: (lambda n, m: f"gen:{n}"))
    prompts, label, source = resolve_prompts(cfg, "classic", 123, "m")
    assert prompts == ["gen:123"]
    assert label == "classic"
    assert source == "generated"


def test_resolve_prompts_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    cfg = BenchmarkConfig(prompts_file=str(path))
    with pytest.raises(ValueError):
        resolve_prompts(cfg, "classic", 1, "")


def test_resolve_prompts_missing_file_raises(tmp_path):
    cfg = BenchmarkConfig(prompts_file=str(tmp_path / "nope.jsonl"))
    with pytest.raises(FileNotFoundError):
        resolve_prompts(cfg, "classic", 1, "")


def test_resolve_prompts_custom_label_without_file_raises():
    with pytest.raises(ValueError):
        resolve_prompts(BenchmarkConfig(), CUSTOM_PROMPT_TYPE, 1, "")


# ---------------------------------------------------------------------------
# _run_burst sends the custom prompts
# ---------------------------------------------------------------------------

def test_run_burst_cycles_custom_prompts(monkeypatch):
    sent: list[tuple[int, str]] = []

    async def _fake_request(session, prompt, request_id, *_a, **_k):
        sent.append((request_id, prompt))
        return {"request_id": request_id, "success": True, "duration": 0.1,
                "prompt_tokens": 1, "completion_tokens": 1}

    async def _noop_warmup(*_a, **_k):
        return None

    monkeypatch.setattr(async_engine, "_make_async_request", _fake_request)
    monkeypatch.setattr(async_engine, "_warmup_connection", _noop_warmup)

    cfg = BenchmarkConfig()
    results = asyncio.run(async_engine._run_burst(["p0", "p1"], 5, cfg, "m"))
    assert len(results) == 5
    assert sorted(sent) == [(0, "p0"), (1, "p1"), (2, "p0"), (3, "p1"), (4, "p0")]


# ---------------------------------------------------------------------------
# run_benchmark_async end-to-end with the file
# ---------------------------------------------------------------------------

def test_run_benchmark_async_uses_prompts_file(prompts_file, monkeypatch):
    seen_prompts: list[str] = []

    async def _fake_burst(prompts, n, cfg, model):
        for i in range(n):
            seen_prompts.append(_prompt_for(prompts, i))
        return [
            {"request_id": i, "success": True, "duration": 1.0,
             "prompt_tokens": 10, "completion_tokens": 20, "ttft": 0.1}
            for i in range(n)
        ]

    class _NoMetrics:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            return False

        def stop(self):
            return None

    class _NoGPU:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            pass

        def stop(self, **_k):
            return None

    monkeypatch.setattr(async_engine, "_run_burst", _fake_burst)
    monkeypatch.setattr(async_engine, "MetricsMonitor", _NoMetrics)
    monkeypatch.setattr(async_engine, "GPUMonitor", _NoGPU)
    monkeypatch.setattr(async_engine, "count_tokens", lambda text, model="": len(text.split()))

    cfg = BenchmarkConfig(prompts_file=prompts_file)
    stats = asyncio.run(async_engine.run_benchmark_async(32_000, 4, cfg, model_name="m", prompt_type="classic"))

    assert stats is not None
    assert stats["prompt_type"] == CUSTOM_PROMPT_TYPE
    assert stats["prompt_source"] == "file"
    assert stats["prompts_file"] == prompts_file
    assert stats["num_custom_prompts"] == 3
    assert seen_prompts == ["custom prompt number 0", "custom prompt number 1",
                            "custom prompt number 2", "custom prompt number 0"]


def test_run_benchmark_async_generated_source(monkeypatch):
    async def _fake_burst(prompts, n, cfg, model):
        assert isinstance(prompts, list) and len(prompts) == 1
        return [{"request_id": 0, "success": True, "duration": 1.0,
                 "prompt_tokens": 10, "completion_tokens": 20}]

    class _No:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            return False

        def stop(self, **_k):
            return None

    monkeypatch.setattr(async_engine, "_run_burst", _fake_burst)
    monkeypatch.setattr(async_engine, "MetricsMonitor", _No)
    monkeypatch.setattr(async_engine, "GPUMonitor", _No)
    monkeypatch.setattr(async_engine, "count_tokens", lambda text, model="": 5)
    monkeypatch.setattr(async_engine, "get_prompt_generator", lambda _t: (lambda n, m: "gen"))

    stats = asyncio.run(async_engine.run_benchmark_async(32_000, 1, BenchmarkConfig(), model_name="m"))
    assert stats["prompt_source"] == "generated"
    assert stats["prompt_type"] == "classic"
    assert "prompts_file" not in stats


# ---------------------------------------------------------------------------
# CLI parser accepts the flag (smoke)
# ---------------------------------------------------------------------------

def test_cli_parser_has_prompts_file():
    from vllm_benchmark.cli import build_parser
    args = build_parser().parse_args(["--prompts-file", "x.jsonl", "-y"])
    assert args.prompts_file == "x.jsonl"
