"""Tests for the vLLM-vs-SGLang head-to-head (``analysis/head_to_head.py``).

Synthetic two-endpoint result sets exercise the per-metric winner logic
(only on a real, CI-separated difference) and the honest-tie behaviour
when the difference is not real.

Author: amit
"""

from __future__ import annotations

from vllm_benchmark.analysis.head_to_head import head_to_head


def _cell(ctx: int, users: int, tps_samples: list[float]) -> dict:
    return {
        "context_length": ctx,
        "concurrent_users": users,
        "prompt_type": "classic",
        "tokens_per_second": sum(tps_samples) / len(tps_samples),
        "tokens_per_second_samples": tps_samples,
    }


def _meta(backend: str) -> dict:
    return {"server_info": {"backend": backend, "model_name": "llama-3-8b"}}


def test_winner_on_real_difference() -> None:
    a = [_cell(32000, 1, [2000, 2050, 2100, 2150, 2200])]
    b = [_cell(32000, 1, [800, 820, 840, 860, 880])]
    out = head_to_head(a, _meta("vllm"), b, _meta("sglang"))
    assert out["n_matched_cells"] == 1
    metric = out["cells"][0]["metrics"]["tokens_per_second"]
    assert metric["tie"] is False
    assert metric["winner"] == "a"  # vllm faster on tok/s (higher is better)
    assert out["summary"]["tokens_per_second"]["a"] == 1


def test_tie_when_not_real_difference() -> None:
    a = [_cell(32000, 1, [1000, 1010, 1005, 1008, 1002])]
    b = [_cell(32000, 1, [1001, 1009, 1004, 1007, 1003])]
    out = head_to_head(a, _meta("vllm"), b, _meta("sglang"))
    metric = out["cells"][0]["metrics"]["tokens_per_second"]
    assert metric["tie"] is True
    assert metric["winner"] is None
    assert out["summary"]["tokens_per_second"]["tie"] == 1


def test_only_matched_cells_compared() -> None:
    a = [_cell(32000, 1, [1000] * 5), _cell(64000, 1, [900] * 5)]
    b = [_cell(32000, 1, [1000] * 5)]  # only one cell in common
    out = head_to_head(a, _meta("vllm"), b, _meta("sglang"))
    assert out["n_matched_cells"] == 1
    assert out["cells"][0]["cell"] == [32000, 1, "classic"]


def test_lower_is_better_metric_winner() -> None:
    # TTFT: lower is better. Endpoint B has much lower TTFT -> B wins.
    a = [{
        "context_length": 32000, "concurrent_users": 1, "prompt_type": "classic",
        "ttft_samples": [0.50, 0.51, 0.49, 0.52, 0.48],
    }]
    b = [{
        "context_length": 32000, "concurrent_users": 1, "prompt_type": "classic",
        "ttft_samples": [0.10, 0.11, 0.09, 0.12, 0.08],
    }]
    out = head_to_head(a, _meta("vllm"), b, _meta("sglang"))
    metric = out["cells"][0]["metrics"]["ttft_estimate"]
    assert metric["winner"] == "b"


def test_no_matched_cells() -> None:
    a = [_cell(32000, 1, [1000] * 5)]
    b = [_cell(64000, 1, [1000] * 5)]
    out = head_to_head(a, _meta("vllm"), b, _meta("sglang"))
    assert out["n_matched_cells"] == 0
    assert out["cells"] == []
    assert out["label_a"] == "vllm:llama-3-8b"
