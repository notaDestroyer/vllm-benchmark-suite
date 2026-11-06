"""The engine emits per-request sample lists and head-to-head uses them.

Before this fix ``head_to_head`` looked for ``*_samples`` keys that nothing
produced, so every comparison ran on n=1 point estimates and was always a
tie.  ``latency_p99`` was also "sampled" from the raw latency list, which
compared mean latency under a p99 label.
"""

from __future__ import annotations

import random

from vllm_benchmark.analysis.head_to_head import _METRICS, _metric_sample, head_to_head
from vllm_benchmark.core.async_engine import _compute_stats


def _requests(n: int, *, latency: float, tps: float, ttft: float, jitter: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = latency * (1 + rng.uniform(-jitter, jitter))
        out.append({
            "request_id": i, "success": True, "duration": d,
            "prompt_tokens": 1000, "completion_tokens": 200,
            "ttft": ttft * (1 + rng.uniform(-jitter, jitter)),
            "prefill_tps": 1000 / (ttft * (1 + rng.uniform(-jitter, jitter))),
            "decode_tps": tps * (1 + rng.uniform(-jitter, jitter)),
            "pp_tg_source": "streaming",
        })
    return out


def _cell(n: int, *, latency: float, tps: float, ttft: float, jitter: float = 0.05, seed: int = 0) -> dict:
    reqs = _requests(n, latency=latency, tps=tps, ttft=ttft, jitter=jitter, seed=seed)
    return _compute_stats(reqs, total_time=latency * 1.1, context_length=32_000, num_concurrent_users=n,
                          prompt_type="classic", actual_prompt_tokens=1000, gpu_stats=None,
                          metrics_stats=None, cost_per_hour=None)


# ---------------------------------------------------------------------------
# Engine emits samples
# ---------------------------------------------------------------------------

def test_compute_stats_emits_sample_lists():
    cell = _cell(8, latency=2.0, tps=50.0, ttft=0.2)
    assert len(cell["latency_samples"]) == 8
    assert len(cell["ttft_samples"]) == 8
    assert len(cell["prefill_tps_samples"]) == 8
    assert len(cell["decode_tps_samples"]) == 8
    assert len(cell["request_tps_samples"]) == 8
    # Plain point-estimate aliases for the pp/tg split.
    assert abs(cell["prefill_tps"] - cell["prefill_tps_mean"]) < 1e-9
    assert abs(cell["decode_tps"] - cell["decode_tps_mean"]) < 1e-9
    # Per-request tps is tokens / duration.
    assert abs(cell["request_tps_samples"][0] - 200 / cell["latency_samples"][0]) < 1e-9


def test_compute_stats_omits_samples_without_data():
    reqs = [{"request_id": 0, "success": True, "duration": 1.0, "prompt_tokens": 1, "completion_tokens": 0}]
    cell = _compute_stats(reqs, 1.0, 1, 1, "classic", 1, None, None, None)
    assert cell["latency_samples"] == [1.0]
    assert "ttft_samples" not in cell
    assert "prefill_tps_samples" not in cell
    assert "request_tps_samples" not in cell  # zero completion tokens


def test_samples_are_json_serialisable_floats():
    import json
    cell = _cell(4, latency=1.0, tps=10.0, ttft=0.1)
    json.dumps(cell)
    assert all(isinstance(v, float) for v in cell["latency_samples"])


# ---------------------------------------------------------------------------
# Head-to-head consumes them
# ---------------------------------------------------------------------------

def test_metrics_compare_avg_latency_not_p99_under_wrong_label():
    assert "avg_latency" in _METRICS
    assert "latency_p99" not in _METRICS


def test_metric_sample_prefers_engine_lists():
    cell = _cell(6, latency=1.0, tps=10.0, ttft=0.1)
    assert _metric_sample(cell, "tokens_per_second") == cell["request_tps_samples"]
    assert _metric_sample(cell, "avg_latency") == cell["latency_samples"]
    assert _metric_sample(cell, "ttft_estimate") == cell["ttft_samples"]
    assert _metric_sample(cell, "prefill_tps") == cell["prefill_tps_samples"]
    assert _metric_sample(cell, "decode_tps") == cell["decode_tps_samples"]


def test_metric_sample_prefers_cross_iteration_lists():
    cell = {"tokens_per_second": 100.0, "tokens_per_second_samples": [98.0, 102.0],
            "request_tps_samples": [1.0, 2.0, 3.0]}
    assert _metric_sample(cell, "tokens_per_second") == [98.0, 102.0]


def test_metric_sample_point_fallback_for_pp_tg_mean_keys():
    cell = {"prefill_tps_mean": 5000.0, "decode_tps_p50": 40.0}
    assert _metric_sample(cell, "prefill_tps") == [5000.0]
    assert _metric_sample(cell, "decode_tps") == [40.0]


def test_head_to_head_finds_a_real_winner_from_engine_cells():
    # B is clearly faster in every dimension; A and B each have 16 requests.
    a = _cell(16, latency=4.0, tps=30.0, ttft=0.40, seed=1)
    b = _cell(16, latency=2.0, tps=60.0, ttft=0.20, seed=2)
    out = head_to_head([a], {"server_info": {"backend": "vllm"}}, [b], {"server_info": {"backend": "sglang"}})
    assert out["n_matched_cells"] == 1
    m = out["cells"][0]["metrics"]
    assert m["avg_latency"]["winner"] == "b"
    assert m["ttft_estimate"]["winner"] == "b"
    assert m["decode_tps"]["winner"] == "b"
    assert m["tokens_per_second"]["winner"] == "b"
    assert m["tokens_per_second"]["n_a"] == 16
    # point_* carry the aggregate cell values; mean_* the sample means.
    assert m["tokens_per_second"]["point_a"] == a["tokens_per_second"]
    assert m["tokens_per_second"]["point_b"] == b["tokens_per_second"]
    assert out["summary"]["avg_latency"]["b"] == 1


def test_head_to_head_identical_distributions_tie():
    a = _cell(16, latency=2.0, tps=60.0, ttft=0.2, seed=3)
    b = _cell(16, latency=2.0, tps=60.0, ttft=0.2, seed=4)
    out = head_to_head([a], {}, [b], {})
    for metric, verdict in out["cells"][0]["metrics"].items():
        assert verdict["tie"], metric


def test_head_to_head_point_estimates_only_still_tie():
    a = {"context_length": 1, "concurrent_users": 1, "prompt_type": "c", "tokens_per_second": 10.0}
    b = {"context_length": 1, "concurrent_users": 1, "prompt_type": "c", "tokens_per_second": 1000.0}
    out = head_to_head([a], {}, [b], {})
    v = out["cells"][0]["metrics"]["tokens_per_second"]
    assert v["tie"] and v["n_a"] == 1
