"""``aggregate_iterations`` must preserve the full result schema.

Before this fix only four metrics survived aggregation, so with
``--iterations > 1`` the bottleneck, fitness, scoring and diagnostics
stages silently lost prefill/decode throughput, percentiles, GPU stats,
cost and failure counts.
"""

from __future__ import annotations

from vllm_benchmark.analysis.statistics import aggregate_iterations


def _cell(it: int, **over) -> dict:
    base = {
        "context_length": 32_000, "concurrent_users": 4, "prompt_type": "classic",
        "iteration": it,
        "tokens_per_second": 100.0 + it, "avg_latency": 2.0 + it * 0.1,
        "ttft_estimate": 0.2, "throughput_per_user": 25.0 + it,
        "latency_p50": 1.9, "latency_p99": 3.0 + it,
        "prefill_tps_mean": 5000.0 + it, "decode_tps_mean": 40.0,
        "avg_gpu_util": 80.0 + it, "avg_power": 300.0,
        "cost_per_1m_tokens": 0.5, "successful": 4, "failed": it % 2,
        "pp_tg_source": "streaming", "prompt_source": "generated",
        "latency_samples": [2.0, 2.1], "request_tps_samples": [50.0, 51.0],
        "std_latency": 0.1,
    }
    base.update(over)
    return base


def test_all_numeric_metrics_preserved_as_median():
    agg = aggregate_iterations([[_cell(0)], [_cell(1)], [_cell(2)]])
    assert len(agg) == 1
    e = agg[0]
    assert e["prefill_tps_mean"] == 5001.0
    assert e["latency_p99"] == 4.0
    assert e["avg_gpu_util"] == 81.0
    assert e["cost_per_1m_tokens"] == 0.5
    assert e["decode_tps_mean"] == 40.0
    assert e["std_latency"] == 0.1


def test_integer_counts_stay_integers():
    agg = aggregate_iterations([[_cell(0)], [_cell(1)], [_cell(2)]])
    e = agg[0]
    assert e["successful"] == 4 and isinstance(e["successful"], int)
    assert e["failed"] == 0 and isinstance(e["failed"], int)


def test_iteration_key_dropped_and_n_iterations_added():
    agg = aggregate_iterations([[_cell(0)], [_cell(1)]])
    e = agg[0]
    assert "iteration" not in e
    assert e["n_iterations"] == 2


def test_non_numeric_fields_carried():
    agg = aggregate_iterations([[_cell(0)], [_cell(1)]])
    e = agg[0]
    assert e["pp_tg_source"] == "streaming"
    assert e["prompt_source"] == "generated"


def test_per_request_samples_pooled_and_cross_iteration_samples_added():
    agg = aggregate_iterations([[_cell(0)], [_cell(1)]])
    e = agg[0]
    assert e["latency_samples"] == [2.0, 2.1, 2.0, 2.1]
    assert e["tokens_per_second_samples"] == [100.0, 101.0]
    assert e["avg_latency_samples"] == [2.0, 2.1]
    assert "tokens_per_second_ci_lower" in e and "tokens_per_second_stats" in e


def test_missing_metric_in_some_iterations_uses_available():
    a = _cell(0)
    b = _cell(1)
    del b["avg_gpu_util"]
    agg = aggregate_iterations([[a], [b]])
    assert agg[0]["avg_gpu_util"] == 80.0


def test_none_values_ignored():
    a = _cell(0, decode_tps_mean=None)
    b = _cell(1, decode_tps_mean=42.0)
    agg = aggregate_iterations([[a], [b]])
    assert agg[0]["decode_tps_mean"] == 42.0


def test_downstream_bottleneck_sees_full_schema():
    """classify_run must get the same keys from aggregated cells as from raw ones."""
    from vllm_benchmark.analysis.bottleneck import classify_run
    from vllm_benchmark.analysis.model_intel import build_profile, match_gpu_spec
    from vllm_benchmark.core.backends.base import ServerInfo

    profile = build_profile(ServerInfo(backend="vllm", model_name="meta-llama/Llama-3-8B"), allow_network=False)
    spec = match_gpu_spec("NVIDIA H100 80GB HBM3")
    raw = [_cell(0, concurrent_users=u, tokens_per_second=100.0 * u) for u in (1, 4)]
    agg = aggregate_iterations([raw, raw])
    v_raw = classify_run(raw, profile, spec)
    v_agg = classify_run(agg, profile, spec)
    assert [v.primary for v in v_raw] == [v.primary for v in v_agg]
    assert all(v.primary != "unknown" for v in v_agg)
