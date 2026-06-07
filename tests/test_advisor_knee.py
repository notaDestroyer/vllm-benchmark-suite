"""Tests for the advisor: Kneedle knee detection, operating points, tips.

A synthetic concave curve with a known knee, flat-curve handling, and
verification that ``config_tips`` fire only on the right model/GPU
profiles.

Author: amit
"""

from __future__ import annotations

from vllm_benchmark.analysis.advisor import (
    Advisory,
    build_advisory,
    config_tips,
    find_knee,
    recommend_operating_points,
    throughput_explanation,
)
from vllm_benchmark.analysis.model_intel import ModelProfile
from vllm_benchmark.core.backends.base import ServerInfo

H100 = {"hbm_bandwidth_gbps": 3350.0, "peak_flops_tflops": {"bf16": 989.0, "fp8": 1979.0}}
A100 = {"hbm_bandwidth_gbps": 2039.0, "peak_flops_tflops": {"bf16": 312.0, "fp8": None}}


# ---------------------------------------------------------------------------
# Kneedle
# ---------------------------------------------------------------------------

def test_find_knee_known_concave() -> None:
    # Concave-increasing curve that bends sharply around x=4.
    x = [1, 2, 4, 8, 16, 32]
    y = [100, 190, 360, 500, 560, 580]
    knee = find_knee(x, y, "concave", "increasing")
    assert knee in (4, 8)  # the bend region


def test_find_knee_flat_returns_none() -> None:
    assert find_knee([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_find_knee_too_few_points() -> None:
    assert find_knee([1, 2], [1, 2]) is None


def test_find_knee_convex_decreasing() -> None:
    # Latency curve rising convexly with concurrency.
    x = [1, 2, 4, 8, 16]
    y = [1.0, 1.1, 1.3, 2.0, 4.0]
    knee = find_knee(x, y, "convex", "increasing")
    assert knee is not None


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------

def _cell(conc: int, tps: float, lat: float) -> dict:
    return {
        "concurrent_users": conc,
        "tokens_per_second": tps,
        "latency_p95": lat,
        "avg_latency": lat,
        "context_length": 8192,
        "prompt_type": "classic",
    }


def test_recommend_operating_points() -> None:
    results = [
        _cell(1, 100, 0.5),
        _cell(2, 190, 0.6),
        _cell(4, 360, 0.9),
        _cell(8, 500, 1.8),
        _cell(16, 560, 4.0),
        _cell(32, 580, 9.0),
    ]
    points = recommend_operating_points(results)
    assert points["throughput_optimal"] is not None
    assert points["latency_optimal"] is not None
    # Latency-optimal should favour the elbow before latency explodes.
    assert points["latency_optimal"]["concurrent_users"] <= 8


def test_recommend_operating_points_empty() -> None:
    points = recommend_operating_points([])
    assert points == {"latency_optimal": None, "throughput_optimal": None}


# ---------------------------------------------------------------------------
# Throughput explanation bands
# ---------------------------------------------------------------------------

def test_throughput_explanation_moe_and_bands() -> None:
    p = ModelProfile(
        name="Mixtral-8x7B", is_moe=True,
        active_params=12_900_000_000, total_params=46_700_000_000,
    )
    top = _cell(8, 500, 1.8)
    text = throughput_explanation(p, top, mbu=0.8, mfu=0.3)
    assert "MoE" in text
    assert "near-optimal" in text  # MBU 0.8 > 0.7
    assert "below ceiling" in text  # MFU 0.3 < 0.4


def test_throughput_explanation_dense() -> None:
    p = ModelProfile(name="Llama-3-8B", is_moe=False, active_params=8_000_000_000)
    text = throughput_explanation(p, _cell(4, 300, 1.0), mbu=0.5, mfu=None)
    assert "Dense" in text
    assert "moderate" in text


def test_throughput_explanation_no_data() -> None:
    assert throughput_explanation(None, None, None, None)


# ---------------------------------------------------------------------------
# Config tips
# ---------------------------------------------------------------------------

def test_config_tips_moe() -> None:
    p = ModelProfile(name="Mixtral", is_moe=True, attention_type="GQA")
    info = ServerInfo(backend="vllm", quantization="FP16/BF16", prefix_caching=True)
    tips = config_tips(p, info, A100)
    assert any("expert parallelism" in t for t in tips)
    assert any("GQA" in t for t in tips)


def test_config_tips_fp8_suggestion() -> None:
    p = ModelProfile(name="Llama-3-8B", is_moe=False, attention_type="GQA")
    info = ServerInfo(backend="vllm", quantization="FP16/BF16", prefix_caching=True)
    tips = config_tips(p, info, H100)  # H100 is FP8-capable
    assert any("FP8" in t for t in tips)


def test_config_tips_no_fp8_on_incapable_gpu() -> None:
    p = ModelProfile(name="Llama-3-8B", is_moe=False, attention_type="MHA")
    info = ServerInfo(backend="vllm", quantization="FP16/BF16", prefix_caching=True)
    tips = config_tips(p, info, A100)  # A100 has no FP8 entry
    assert not any("supports FP8" in t for t in tips)


def test_config_tips_prefix_caching() -> None:
    p = ModelProfile(name="Llama-3-8B", is_moe=False, attention_type="MHA")
    info = ServerInfo(backend="vllm", quantization="FP8", prefix_caching=False)
    tips = config_tips(p, info, H100)
    assert any("prefix caching" in t.lower() for t in tips)


def test_config_tips_none_profile() -> None:
    info = ServerInfo(backend="vllm")
    assert config_tips(None, info, H100) == []


# ---------------------------------------------------------------------------
# Advisory aggregate
# ---------------------------------------------------------------------------

def test_build_advisory() -> None:
    results = [_cell(1, 100, 0.5), _cell(4, 360, 0.9), _cell(16, 560, 4.0)]
    p = ModelProfile(name="Llama-3-8B", is_moe=False, active_params=8e9,
                     confidence="confirmed", attention_type="GQA")
    info = ServerInfo(backend="vllm", quantization="FP16/BF16", prefix_caching=True)
    adv = build_advisory(results, p, info, H100, mbu=0.6, mfu=0.5)
    assert isinstance(adv, Advisory)
    assert adv.confidence == "high"  # confirmed profile
    assert adv.fitness is None  # extension point untouched
    assert adv.throughput_optimal is not None
    d = adv.to_dict()
    assert "explanation" in d and "tips" in d
