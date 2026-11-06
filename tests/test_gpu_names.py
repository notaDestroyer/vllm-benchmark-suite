"""Shared GPU-name matcher: real nvidia-smi names resolve to table keys and
short keys never over-match."""

from __future__ import annotations

import pytest

from vllm_benchmark.analysis.model_intel import match_gpu_spec
from vllm_benchmark.analysis.scoring import VLLMScore
from vllm_benchmark.config import GPU_BASELINES, GPU_PRICING
from vllm_benchmark.core.hardware.gpu_names import match_gpu_key, normalize_gpu_name


def test_normalize_strips_count_prefix_and_noise():
    assert normalize_gpu_name("4x NVIDIA A100-SXM4-80GB") == ["a100", "sxm4", "80gb"]
    assert normalize_gpu_name("NVIDIA GeForce RTX 4090") == ["rtx", "4090"]
    assert normalize_gpu_name("Tesla T4") == ["t4"]
    assert normalize_gpu_name(None) == []
    assert normalize_gpu_name("") == []


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
        ("NVIDIA A100 80GB PCIe", "A100-80GB"),
        ("NVIDIA A100-PCIE-40GB", "A100-40GB"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA H100 PCIe", "H100"),
        ("4x NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L40S", "L40S"),
        ("NVIDIA GeForce RTX 4090", "RTX 4090"),
        ("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada"),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000"),
        ("NVIDIA L4", None),       # not in the scoring baselines
        ("Intel Arc A770", None),
        ("", None),
    ],
)
def test_scoring_baseline_match(name, expected):
    key, baseline = VLLMScore._match_gpu_baseline(name)
    assert key == expected
    assert (baseline is None) == (expected is None)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L4", "L4"),
        ("NVIDIA L40S", "L40S"),
        ("Tesla T4", "T4"),
        ("NVIDIA A10G", "A10G"),
        ("2x NVIDIA A10G", "A10G"),
    ],
)
def test_pricing_match(name, expected):
    assert match_gpu_key(name, GPU_PRICING.keys()) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA H100 80GB HBM3", "H100 SXM"),
        ("NVIDIA H100 PCIe", "H100 PCIe"),
        ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
        ("NVIDIA A100-PCIE-40GB", "A100-40GB"),
        ("NVIDIA L4", "L4"),
        ("NVIDIA L40S", "L40S"),
        ("8x NVIDIA H100 80GB HBM3", "H100 SXM"),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000 Blackwell"),
        ("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada"),
    ],
)
def test_spec_match(name, expected):
    spec = match_gpu_spec(name)
    assert spec is not None and spec["_matched_key"] == expected


def test_short_keys_do_not_overmatch():
    # "L4" is a prefix of "L40S" and "T4" of nothing relevant; exact tokens only.
    assert match_gpu_key("NVIDIA L40S", ["L4", "L40S"]) == "L40S"
    assert match_gpu_key("NVIDIA L4", ["L40S", "L4"]) == "L4"
    assert match_gpu_key("NVIDIA A100-SXM4-80GB", ["A100-40GB"]) is None  # contradicting memory


def test_calculate_reports_gpu_match_for_real_name():
    results = [{"tokens_per_second": 1000.0, "avg_latency": 1.0, "std_latency": 0.1, "throughput_per_user": 100.0}]
    score = VLLMScore().calculate(results, gpu_name="NVIDIA A100-SXM4-80GB")
    assert score.gpu_match == "A100-80GB"
    assert score.gpu_percentile is not None
    assert "A100-80GB" in GPU_BASELINES
