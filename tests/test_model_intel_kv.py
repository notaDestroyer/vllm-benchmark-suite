"""Tests for KV-cache sizing, attention typing and MoE active-param math.

Parametrized over fixture ``config.json`` files (Llama-3-8B GQA,
Mixtral-8x7B MoE, Qwen3-30B-A3B MoE, a dense MHA model and an MQA model).
Asserts ``kv_bytes_per_token``, ``attention_type`` and ``active_params``
against hand-computed / published values and verifies MoE detection.

Author: amit
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_benchmark.analysis.model_intel import (
    attention_type_of,
    compute_active_params,
    kv_bytes_per_token,
)

FIXTURES = Path(__file__).parent / "fixtures" / "configs"


def _load(name: str) -> dict:
    with open(FIXTURES / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _head_dim(cfg: dict) -> int:
    if cfg.get("head_dim"):
        return cfg["head_dim"]
    return cfg["hidden_size"] // cfg["num_attention_heads"]


def _n_kv(cfg: dict) -> int:
    return cfg.get("num_key_value_heads") or cfg["num_attention_heads"]


# ---------------------------------------------------------------------------
# Attention typing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture,expected",
    [
        ("llama3_8b.json", "GQA"),
        ("mixtral_8x7b.json", "GQA"),
        ("qwen3_30b_a3b.json", "GQA"),
        ("dense_mha.json", "MHA"),
        ("mqa_model.json", "MQA"),
    ],
)
def test_attention_type(fixture: str, expected: str) -> None:
    cfg = _load(fixture)
    assert attention_type_of(cfg["num_attention_heads"], _n_kv(cfg)) == expected


# ---------------------------------------------------------------------------
# KV bytes per token (bf16 = 2 bytes)
# ---------------------------------------------------------------------------

def test_kv_bytes_llama3_8b() -> None:
    cfg = _load("llama3_8b.json")
    # 2 * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072
    val = kv_bytes_per_token(cfg["num_hidden_layers"], _n_kv(cfg), _head_dim(cfg), "bf16")
    assert val == 131072


def test_kv_bytes_mqa_smaller_than_mha() -> None:
    mqa = _load("mqa_model.json")
    mha = _load("dense_mha.json")
    mqa_val = kv_bytes_per_token(
        mqa["num_hidden_layers"], _n_kv(mqa), _head_dim(mqa), "fp16"
    )
    mha_val = kv_bytes_per_token(
        mha["num_hidden_layers"], _n_kv(mha), _head_dim(mha), "fp16"
    )
    # MQA has a single KV head -> dramatically smaller cache.
    assert mqa_val < mha_val


def test_kv_bytes_fp8_halves_bf16() -> None:
    cfg = _load("llama3_8b.json")
    bf16 = kv_bytes_per_token(cfg["num_hidden_layers"], _n_kv(cfg), _head_dim(cfg), "bf16")
    fp8 = kv_bytes_per_token(cfg["num_hidden_layers"], _n_kv(cfg), _head_dim(cfg), "fp8")
    assert fp8 == bf16 // 2


def test_kv_bytes_none_on_missing() -> None:
    assert kv_bytes_per_token(None, 8, 128, "bf16") is None
    assert kv_bytes_per_token(32, None, 128, "bf16") is None
    assert kv_bytes_per_token(32, 8, None, "bf16") is None


# ---------------------------------------------------------------------------
# MoE detection + active params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture,is_moe",
    [
        ("llama3_8b.json", False),
        ("mixtral_8x7b.json", True),
        ("qwen3_30b_a3b.json", True),
        ("dense_mha.json", False),
        ("mqa_model.json", False),
    ],
)
def test_moe_detection(fixture: str, is_moe: bool) -> None:
    cfg = _load(fixture)
    detected = bool(
        (cfg.get("num_experts") or cfg.get("num_local_experts"))
        and (cfg.get("experts_per_tok") or cfg.get("num_experts_per_tok"))
    )
    assert detected == is_moe


def test_active_params_mixtral_within_tolerance() -> None:
    cfg = _load("mixtral_8x7b.json")
    cfg = dict(cfg)
    cfg["total_params"] = 46_700_000_000
    cfg["num_layers"] = cfg["num_hidden_layers"]
    active = compute_active_params(cfg)
    # Published Mixtral-8x7B active params ~12.9B.
    assert active is not None
    assert abs(active - 12_900_000_000) / 12_900_000_000 < 0.10


def test_active_params_qwen3_within_tolerance() -> None:
    cfg = _load("qwen3_30b_a3b.json")
    cfg = dict(cfg)
    cfg["total_params"] = 30_500_000_000
    cfg["num_layers"] = cfg["num_hidden_layers"]
    active = compute_active_params(cfg)
    assert active is not None
    # Qwen3-30B-A3B advertises ~3.3B active; estimate uses moe_intermediate_size.
    assert active < 8_000_000_000  # far below total -> MoE sparsity captured
    assert active > 1_000_000_000


def test_active_params_dense_equals_total() -> None:
    cfg = _load("llama3_8b.json")
    cfg = dict(cfg)
    cfg["total_params"] = 8_030_000_000
    assert compute_active_params(cfg) == 8_030_000_000


def test_active_params_none_without_total() -> None:
    cfg = _load("llama3_8b.json")
    assert compute_active_params(dict(cfg)) is None


def test_active_params_uses_direct_kb_active() -> None:
    # When the entry already carries active_params, it is used verbatim.
    out = compute_active_params(
        {"num_experts": 8, "experts_per_tok": 2, "active_params": 12_900_000_000}
    )
    assert out == 12_900_000_000
