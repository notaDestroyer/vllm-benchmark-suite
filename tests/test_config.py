"""Tests for benchmark configuration."""

from vllm_benchmark.config import (
    GPU_PRICING,
    BenchmarkConfig,
    parse_concurrency,
    parse_context_lengths,
)

# ---------------------------------------------------------------------------
# GPU_PRICING
# ---------------------------------------------------------------------------


def test_gpu_pricing_exists():
    assert isinstance(GPU_PRICING, dict)
    assert len(GPU_PRICING) > 0


def test_gpu_pricing_expected_keys():
    expected = {"H100", "A100-80GB", "A100-40GB", "L40S", "RTX 4090"}
    assert expected.issubset(set(GPU_PRICING.keys()))


def test_gpu_pricing_values_positive():
    for gpu, price in GPU_PRICING.items():
        assert isinstance(price, (int, float)), f"{gpu} price is not numeric"
        assert price > 0, f"{gpu} price should be positive"


# ---------------------------------------------------------------------------
# BenchmarkConfig — cost_per_hour field
# ---------------------------------------------------------------------------


def test_config_has_cost_per_hour_field():
    cfg = BenchmarkConfig()
    assert hasattr(cfg, "cost_per_hour")
    assert cfg.cost_per_hour is None  # default is None


def test_config_cost_per_hour_settable():
    cfg = BenchmarkConfig(cost_per_hour=2.50)
    assert cfg.cost_per_hour == 2.50


def test_config_from_preset_with_cost():
    cfg = BenchmarkConfig.from_preset("quick", cost_per_hour=4.00)
    assert cfg.cost_per_hour == 4.00
    assert cfg.context_lengths == [32_000]


# ---------------------------------------------------------------------------
# BenchmarkConfig — properties
# ---------------------------------------------------------------------------


def test_config_api_endpoint():
    cfg = BenchmarkConfig(api_url="http://myhost:9000")
    assert cfg.api_endpoint == "http://myhost:9000/v1/chat/completions"


def test_config_total_tests():
    cfg = BenchmarkConfig(
        context_lengths=[32_000, 64_000],
        concurrency_levels=[1, 4],
        prompt_types=["classic"],
    )
    assert cfg.total_tests == 4  # 2 x 2 x 1


# ---------------------------------------------------------------------------
# parse_context_lengths / parse_concurrency
# ---------------------------------------------------------------------------


def test_parse_context_lengths_k_suffix():
    assert parse_context_lengths("32k,64k,128k") == [32000, 64000, 128000]


def test_parse_context_lengths_numeric():
    assert parse_context_lengths("1000,5000") == [1000, 5000]


def test_parse_concurrency():
    assert parse_concurrency("1,4,8,16") == [1, 4, 8, 16]
