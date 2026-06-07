"""Tests for governing-bottleneck classification.

Exercises the precedence hierarchy (capacity > queue > interconnect >
physics), the roofline-driven decode-regime choice, B* shifts with dtype
and context, analytic-vs-empirical disagreement lowering confidence, and
the pure :func:`batch_scaling_regime` on synthetic curves.

Author: amit
"""

from __future__ import annotations

from vllm_benchmark.analysis.bottleneck import (
    batch_scaling_regime,
    classify_cell,
    classify_run,
)
from vllm_benchmark.analysis.model_intel import ModelProfile

H100 = {"hbm_bandwidth_gbps": 3350.0, "peak_flops_tflops": {"bf16": 989.0, "fp8": 1979.0}}


def _profile(**kw) -> ModelProfile:
    base = dict(
        name="meta-llama/Llama-3-8B",
        is_moe=False,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        active_params=8_000_000_000,
        total_params=8_000_000_000,
        attention_type="GQA",
        kv_bytes_per_token=131072,
    )
    base.update(kw)
    return ModelProfile(**base)


def _cell(**kw) -> dict:
    base = dict(
        context_length=8192,
        concurrent_users=1,
        prompt_type="classic",
        tokens_per_second=50.0,
        avg_latency=1.0,
        std_latency=0.05,
        latency_p50=1.0,
        latency_p99=1.2,
        avg_prompt_tokens=8192,
        avg_completion_tokens=500,
        prefill_tps_mean=10000.0,
        decode_tps_mean=50.0,
    )
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# batch_scaling_regime (pure)
# ---------------------------------------------------------------------------

def test_regime_linear() -> None:
    # Perfectly linear: throughput tracks concurrency.
    assert batch_scaling_regime([1, 2, 4, 8], [100, 200, 400, 800]) == "linear"


def test_regime_saturating() -> None:
    # Throughput plateaus -> compute-bound.
    assert batch_scaling_regime([1, 2, 4, 8], [100, 150, 170, 175]) == "saturating"


def test_regime_collapsing() -> None:
    # Peak below the lowest-concurrency throughput -> overload.
    assert batch_scaling_regime([1, 2, 4, 8], [200, 150, 90, 50]) == "collapsing"


def test_regime_degenerate() -> None:
    assert batch_scaling_regime([1], [100]) == "linear"


def test_regime_zero_baseline_throughput() -> None:
    # t0 == 0 -> guarded fallback to linear.
    assert batch_scaling_regime([1, 2, 4], [0, 100, 200]) == "linear"


def test_prefill_frac_zero_tps_returns_none() -> None:
    from vllm_benchmark.analysis.bottleneck import _prefill_time_frac
    cell = _cell(prefill_tps_mean=0.0, decode_tps_mean=0.0)
    assert _prefill_time_frac(cell, _profile()) is None


# ---------------------------------------------------------------------------
# Precedence: capacity beats everything
# ---------------------------------------------------------------------------

def test_kv_capacity_precedence() -> None:
    # Even with queueing signals present, saturated cache wins.
    metrics = {"gpu_cache_usage_perc": 0.97, "num_requests_waiting": 5}
    v = classify_cell(_cell(latency_p99=10.0, latency_p50=1.0), _profile(), H100, metrics)
    assert v.primary == "kv_capacity"
    assert v.confidence == "high"
    assert "VRAM" in v.lever


def test_kv_capacity_from_preemptions() -> None:
    v = classify_cell(_cell(), _profile(), H100, {"preemptions": 3})
    assert v.primary == "kv_capacity"


# ---------------------------------------------------------------------------
# Precedence: queue
# ---------------------------------------------------------------------------

def test_queue_from_waiting() -> None:
    v = classify_cell(_cell(), _profile(), H100, {"num_requests_waiting": 4})
    assert v.primary == "queue"
    assert "max_num_seqs" in v.lever


def test_queue_from_tail_latency() -> None:
    # p99/p50 = 8 > 5 triggers queue even with no waiting metric.
    v = classify_cell(
        _cell(latency_p50=1.0, latency_p99=8.0), _profile(), H100, None
    )
    assert v.primary == "queue"


# ---------------------------------------------------------------------------
# Physics: decode regimes
# ---------------------------------------------------------------------------

def test_decode_weight_bandwidth_short_context() -> None:
    # Short context, B=1 < B*: weight bytes dominate -> weight bandwidth.
    cell = _cell(context_length=512, avg_prompt_tokens=512, concurrent_users=1)
    v = classify_cell(cell, _profile(), H100, None)
    assert v.primary == "decode_weight_bandwidth"
    assert v.critical_batch is not None and v.critical_batch > 1
    assert v.mbu is not None


def test_decode_kv_bandwidth_long_context() -> None:
    # Very long context so kv_bytes (kvbpt * seq) exceeds weight bytes, and a
    # long output so decode dominates wall-clock time.
    # weight = 8e9 * 2 = 1.6e10; kv = 131072 * seq. seq > ~122070 flips it.
    cell = _cell(
        context_length=200000, avg_prompt_tokens=200000,
        avg_completion_tokens=5000, prefill_tps_mean=10000.0,
        decode_tps_mean=50.0, concurrent_users=1,
    )
    v = classify_cell(cell, _profile(), H100, None)
    assert v.primary == "decode_kv_bandwidth"


def test_decode_compute_above_critical_batch() -> None:
    # Concurrency above B* (~295 for H100 bf16) -> compute-bound decode.
    cell = _cell(concurrent_users=512, prefill_tps_mean=None, decode_tps_mean=None,
                 avg_completion_tokens=500)
    v = classify_cell(cell, _profile(), H100, None)
    assert v.primary == "decode_compute"
    assert "add GPUs" in v.lever


def test_prefill_compute_when_prefill_dominates() -> None:
    # Huge prompt, tiny output, slow prefill -> prefill dominates time.
    cell = _cell(
        avg_prompt_tokens=100000, avg_completion_tokens=5,
        prefill_tps_mean=2000.0, decode_tps_mean=50.0,
    )
    v = classify_cell(cell, _profile(), H100, None)
    assert v.primary == "prefill_compute"
    assert v.prefill_time_frac is not None and v.prefill_time_frac >= 0.5
    assert "compute" in v.lever


# ---------------------------------------------------------------------------
# B* shifts with dtype and context
# ---------------------------------------------------------------------------

def test_bstar_shifts_with_weight_bytes() -> None:
    # Hold peak FLOPS fixed: halving bytes/param halves B*. (When FP8 *also*
    # doubles peak FLOPS the two effects cancel, which is physically correct;
    # this isolates the bytes/param dependence.)
    from vllm_benchmark.analysis.model_intel import critical_batch
    bf16 = critical_batch(989.0, 3350.0, 2.0)
    int4 = critical_batch(989.0, 3350.0, 0.5)
    assert bf16 is not None and int4 is not None
    assert int4 < bf16


def test_fp8_quant_note_changes_bytes_per_param() -> None:
    # An FP8 quant note flips the dtype used for peak FLOPS selection so MFU
    # is computed against the FP8 ceiling rather than bf16.
    p = _profile()
    p.notes.append("Server reports quantization=FP8.")
    cell = _cell(avg_prompt_tokens=100000, avg_completion_tokens=5,
                 prefill_tps_mean=2000.0, decode_tps_mean=50.0)
    v_fp8 = classify_cell(cell, p, H100, None)
    v_bf16 = classify_cell(cell, _profile(), H100, None)
    # Same measured prefill tps, but FP8 ceiling is ~2x -> lower MFU.
    assert v_fp8.mfu is not None and v_bf16.mfu is not None
    assert v_fp8.mfu < v_bf16.mfu


def test_headroom_to_ridge() -> None:
    v = classify_cell(_cell(concurrent_users=8), _profile(), H100, None)
    assert v.headroom_to_ridge is not None
    assert v.headroom_to_ridge == v.critical_batch - 8


# ---------------------------------------------------------------------------
# Interconnect + agreement
# ---------------------------------------------------------------------------

def test_interconnect_detected() -> None:
    # TP>1, saturating below ridge -> interconnect.
    cell = _cell(concurrent_users=4)
    v = classify_cell(cell, _profile(), H100, {"tensor_parallel": 4}, batch_scaling="saturating")
    assert v.primary == "interconnect"
    assert "NVLink" in v.lever


def test_disagreement_lowers_confidence() -> None:
    # Below B* analytic predicts bandwidth-bound (linear); empirical says
    # saturating -> disagreement -> low confidence.
    cell = _cell(concurrent_users=4, std_latency=0.01)  # low CV would be high conf
    v = classify_cell(cell, _profile(), H100, None, batch_scaling="saturating")
    assert v.analytic_vs_empirical_agree is False
    assert v.confidence == "low"


def test_agreement_when_consistent() -> None:
    cell = _cell(concurrent_users=4, std_latency=0.01)
    v = classify_cell(cell, _profile(), H100, None, batch_scaling="linear")
    assert v.analytic_vs_empirical_agree is True


# ---------------------------------------------------------------------------
# Unknown profile
# ---------------------------------------------------------------------------

def test_unknown_without_profile_or_gpu() -> None:
    v = classify_cell(_cell(), None, None, None)
    assert v.primary == "unknown"
    assert v.confidence == "low"


# ---------------------------------------------------------------------------
# classify_run
# ---------------------------------------------------------------------------

def test_classify_run_returns_one_per_cell() -> None:
    results = [
        _cell(concurrent_users=1, tokens_per_second=50),
        _cell(concurrent_users=4, tokens_per_second=180),
        _cell(concurrent_users=8, tokens_per_second=190),
    ]
    verdicts = classify_run(results, _profile(), H100)
    assert len(verdicts) == 3
    assert all(v.primary for v in verdicts)


def test_confidence_bands_medium_and_low() -> None:
    # CV ~0.2 -> medium; CV ~0.5 -> low (no batch_scaling so CV governs).
    med = classify_cell(_cell(avg_latency=1.0, std_latency=0.2), _profile(), H100, None)
    low = classify_cell(_cell(avg_latency=1.0, std_latency=0.5), _profile(), H100, None)
    assert med.confidence == "medium"
    assert low.confidence == "low"


def test_confidence_medium_without_latency_stats() -> None:
    cell = _cell()
    cell.pop("std_latency", None)
    cell.pop("avg_latency", None)
    cell["latency_p50"] = 1.0
    cell["latency_p99"] = 1.1
    v = classify_cell(cell, _profile(), H100, None)
    assert v.confidence == "medium"


def test_prefill_frac_none_without_measurements() -> None:
    cell = _cell(prefill_tps_mean=None, decode_tps_mean=None)
    cell.pop("decode_tps_p50", None)
    v = classify_cell(cell, _profile(), H100, None)
    assert v.prefill_time_frac is None


def test_contributions_normalize_empty() -> None:
    from vllm_benchmark.analysis.bottleneck import _normalize
    assert _normalize({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


def test_decode_compute_zero_kv_bytes() -> None:
    # Zero KV bytes with high concurrency still resolves to decode_compute.
    p = _profile(kv_bytes_per_token=0, active_params=8_000_000_000)
    cell = _cell(concurrent_users=512, prefill_tps_mean=None, decode_tps_mean=None)
    v = classify_cell(cell, p, H100, None)
    assert v.primary == "decode_compute"


def test_verdict_to_dict_roundtrip() -> None:
    v = classify_cell(_cell(), _profile(), H100, None)
    d = v.to_dict()
    assert d["primary"] == v.primary
    assert isinstance(d["contributions"], dict)
    assert isinstance(d["cell"], list)
