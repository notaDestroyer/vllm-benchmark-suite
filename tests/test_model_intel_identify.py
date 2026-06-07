"""Tests for model identification: name parsing, KB fallback, HF fetch.

Covers the name-parser table, the offline KB-fallback path, a mocked
live HuggingFace fetch path, provenance/confidence flags, and the
guarantee that :func:`build_profile` never raises on network errors.

Author: amit
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_benchmark.analysis import model_intel as mi
from vllm_benchmark.core.backends.base import ServerInfo

FIXTURES = Path(__file__).parent / "fixtures" / "configs"


def _server(model_name: str, **kw) -> ServerInfo:
    return ServerInfo(backend="vllm", model_name=model_name, **kw)


# ---------------------------------------------------------------------------
# Name parser table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,checks",
    [
        ("meta-llama/Llama-3-8B-Instruct", {"family": "llama", "variant": "instruct"}),
        ("meta-llama/Llama-3.1-70B", {"family": "llama", "total_params": 70_000_000_000}),
        ("mistralai/Mixtral-8x7B-v0.1", {"family": "mixtral", "is_moe": True, "num_experts": 8}),
        ("mistralai/Mixtral-8x22B", {"is_moe": True, "num_experts": 8}),
        ("Qwen/Qwen3-30B-A3B", {"is_moe": True, "total_params": 30_000_000_000, "active_params": 3_000_000_000}),
        ("neuralmagic/Llama-3-8B-FP8", {"quant": "FP8"}),
        ("TheBloke/Llama-2-7B-AWQ", {"quant": "AWQ"}),
        ("some/Model-7B-GPTQ", {"quant": "GPTQ"}),
    ],
)
def test_parse_model_name_table(name: str, checks: dict) -> None:
    parsed = mi.parse_model_name(name)
    for key, expected in checks.items():
        assert parsed.get(key) == expected, f"{name}: {key}"


def test_parse_model_name_empty() -> None:
    assert mi.parse_model_name(None) == {"name": ""}
    assert mi.parse_model_name("")["name"] == ""


def test_bytes_per_param_table() -> None:
    assert mi.bytes_per_param("fp8") == 1.0
    assert mi.bytes_per_param("INT8") == 1.0
    assert mi.bytes_per_param("int4") == 0.5
    assert mi.bytes_per_param("AWQ") == 0.5
    assert mi.bytes_per_param("bf16") == 2.0
    assert mi.bytes_per_param("float16") == 2.0
    assert mi.bytes_per_param(None) == 2.0
    assert mi.bytes_per_param("unknown-dtype") == 2.0
    assert mi.bytes_per_param("fp8_e4m3") == 1.0  # substring fallback


# ---------------------------------------------------------------------------
# KB-fallback path (no network)
# ---------------------------------------------------------------------------

def test_build_profile_kb_fallback() -> None:
    info = _server("mistralai/Mixtral-8x7B-Instruct-v0.1", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=False)
    assert profile.source == "kb"
    assert profile.confidence == "inferred"
    assert profile.is_moe is True
    assert profile.num_experts == 8
    assert profile.attention_type == "GQA"
    assert profile.kv_bytes_per_token is not None
    assert profile.active_params == 12_900_000_000


def test_build_profile_name_only_when_unknown() -> None:
    info = _server("acme/SuperSecret-13B-Instruct")
    profile = mi.build_profile(info, allow_network=False)
    assert profile.source == "name"
    assert profile.confidence == "heuristic"
    assert profile.param_source == "name_heuristic"
    assert profile.total_params == 13_000_000_000


# ---------------------------------------------------------------------------
# Live HF fetch path (mocked)
# ---------------------------------------------------------------------------

def test_build_profile_hf_live(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = json.loads((FIXTURES / "llama3_8b.json").read_text())

    def fake_fetch(repo_id, hf_token):
        # Mimic config.json + safetensors total_size (~16GB for 8B bf16).
        return cfg, 16_060_000_000

    monkeypatch.setattr(mi, "_hf_fetch_config", fake_fetch)

    info = _server("meta-llama/Meta-Llama-3-8B", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=True)

    assert profile.source == "hf_live"
    assert profile.confidence == "confirmed"
    assert profile.param_source == "config_estimate"
    assert profile.num_layers == 32
    assert profile.num_kv_heads == 8
    assert profile.attention_type == "GQA"
    assert profile.head_dim == 128
    assert profile.kv_bytes_per_token == 131072
    assert profile.weight_bytes == 16_060_000_000


def test_build_profile_hf_live_moe_active(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = json.loads((FIXTURES / "mixtral_8x7b.json").read_text())

    def fake_fetch(repo_id, hf_token):
        return cfg, None

    monkeypatch.setattr(mi, "_hf_fetch_config", fake_fetch)
    info = _server("mistralai/Mixtral-8x7B-v0.1", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=True)
    assert profile.is_moe is True
    # config has no total_params; KB cross-check should supply total/active.
    assert profile.total_params == 46_700_000_000
    assert profile.active_params is not None


# ---------------------------------------------------------------------------
# Robustness: never raise
# ---------------------------------------------------------------------------

def test_build_profile_network_error_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(repo_id, hf_token):
        raise RuntimeError("network down")

    monkeypatch.setattr(mi, "_hf_fetch_config", boom)
    info = _server("meta-llama/Llama-3-8B")
    # Should fall back to KB without raising.
    profile = mi.build_profile(info, allow_network=True)
    assert profile is not None
    assert profile.source in ("kb", "name")


def test_build_profile_gated_returns_kb(monkeypatch: pytest.MonkeyPatch) -> None:
    def gated(repo_id, hf_token):
        return None, None  # config unavailable (gated)

    monkeypatch.setattr(mi, "_hf_fetch_config", gated)
    info = _server("meta-llama/Llama-3-70B", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=True)
    assert profile.source == "kb"
    assert profile.num_layers == 80


def test_build_profile_unknown_model_no_crash() -> None:
    info = ServerInfo(backend="unknown")
    profile = mi.build_profile(info, allow_network=False)
    assert profile.name == "unknown"
    assert profile.confidence == "heuristic"


def test_active_param_disagreement_downgrades(monkeypatch: pytest.MonkeyPatch) -> None:
    # Config with a total_params that yields an active estimate far from KB.
    cfg = json.loads((FIXTURES / "mixtral_8x7b.json").read_text())
    cfg = dict(cfg)
    cfg["total_params"] = 46_700_000_000
    # Force per-expert estimate way off by shrinking intermediate_size so the
    # computed active diverges >10% from the KB's 12.9B.
    cfg["intermediate_size"] = 100

    def fake_fetch(repo_id, hf_token):
        return cfg, None

    monkeypatch.setattr(mi, "_hf_fetch_config", fake_fetch)
    info = _server("mistralai/Mixtral-8x7B-v0.1", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=True)
    # KB cross-check should win and a note should be recorded.
    assert profile.active_params == 12_900_000_000
    assert any("disagrees with" in n for n in profile.notes)
