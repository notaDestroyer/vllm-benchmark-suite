"""Tests for analyst-report orchestration and fallback.

Verifies that :func:`generate_report` never raises, falls back to the
deterministic template on provider failure or on too many unsupported
numbers, and populates ``metadata["analyst_report"]`` either way.
"""

from __future__ import annotations

from vllm_benchmark.analysis.report import (
    AnalystReport,
    build_bundle,
    deterministic_report,
    generate_report,
)
from vllm_benchmark.analysis.report.providers import ProviderError


def _result_cell(**kw) -> dict:
    base = {
        "context_length": 32000,
        "concurrent_users": 4,
        "prompt_type": "classic",
        "tokens_per_second": 1234.0,
        "ttft_estimate": 0.25,
        "avg_latency": 1.2,
        "latency_p99": 2.4,
        "cost_per_1m_tokens": 0.5,
    }
    base.update(kw)
    return base


def _metadata() -> dict:
    return {
        "system_info": {"gpu_name": "NVIDIA H100", "total_vram_gb": 80.0},
        "server_info": {"model_name": "test/model", "backend": "vllm"},
        "model_profile": {
            "name": "test/model", "is_moe": False, "active_params": 8_000_000_000,
            "confidence": "confirmed", "source": "hf_live",
        },
        "bottlenecks": [{
            "cell": [32000, 1, "classic"], "primary": "decode_weight_bandwidth",
            "mbu": 0.78, "mfu": 0.42, "critical_batch": 256, "confidence": "high",
            "lever": "increase memory bandwidth",
        }],
        "advisory": {
            "explanation": "Dense model.",
            "tips": ["enable FP8"],
            "confidence": "high",
            "throughput_optimal": _result_cell(concurrent_users=16),
            "latency_optimal": _result_cell(concurrent_users=1),
            "fitness": {
                "verdict": "Best for: batch.",
                "profiles": {"interactive_chat": {"grade": "Marginal", "limiting_factor": "TTFT"}},
            },
        },
        "quality": {"mode": "probe", "status": "ok", "score": 87.0},
    }


# ---------------------------------------------------------------------------
# Deterministic report content
# ---------------------------------------------------------------------------

def test_deterministic_report_has_all_sections():
    bundle = build_bundle([_result_cell()], _metadata())
    md = deterministic_report(bundle)
    for heading in (
        "## Executive summary",
        "## Bottleneck analysis",
        "## Application fitness",
        "## Recommendations",
        "## Caveats & confidence",
    ):
        assert heading in md


def test_deterministic_report_only_uses_bundle_numbers():
    bundle = build_bundle([_result_cell()], _metadata())
    md = deterministic_report(bundle)
    # The deterministic template must itself pass the verifier (it draws only
    # from the bundle).
    from vllm_benchmark.analysis.report import allowed_numbers, verify_report
    res = verify_report(md, allowed_numbers(bundle))
    assert res["unsupported"] == []


# ---------------------------------------------------------------------------
# Fallback on provider failure
# ---------------------------------------------------------------------------

def test_provider_error_falls_back_without_raising(monkeypatch):
    def boom(name, params):
        raise ProviderError("nope")

    monkeypatch.setattr("vllm_benchmark.analysis.report.analyst.get_provider", boom)

    metadata = _metadata()
    report = generate_report([_result_cell()], metadata, provider="local")

    assert isinstance(report, AnalystReport)
    assert report.generated is False
    assert report.verification is None
    assert "## Executive summary" in report.markdown
    assert metadata.get("analyst_report") is None  # CLI wires this; generate_report doesn't


def test_unexpected_provider_exception_falls_back(monkeypatch):
    class _Exploding:
        def generate(self, system, user_text, params):
            raise RuntimeError("unexpected")

    monkeypatch.setattr(
        "vllm_benchmark.analysis.report.analyst.get_provider",
        lambda name, params: _Exploding(),
    )
    report = generate_report([_result_cell()], _metadata(), provider="local")
    assert report.generated is False
    assert "## Bottleneck analysis" in report.markdown


# ---------------------------------------------------------------------------
# Fallback on verification failure
# ---------------------------------------------------------------------------

def test_too_many_unsupported_numbers_falls_back(monkeypatch):
    class _Liar:
        def generate(self, system, user_text, params):
            # Many fabricated numbers, none in the bundle.
            return "Numbers: 11111, 22222, 33333, 44444, 55555 tok/s."

    monkeypatch.setattr(
        "vllm_benchmark.analysis.report.analyst.get_provider",
        lambda name, params: _Liar(),
    )
    report = generate_report([_result_cell()], _metadata(), provider="local")
    assert report.generated is False
    # Verification still recorded.
    assert report.verification is not None
    assert len(report.verification["unsupported"]) > 3
    # Fell back to deterministic markdown.
    assert "## Executive summary" in report.markdown


def test_grounded_draft_is_kept_and_redacted(monkeypatch):
    class _MostlyHonest:
        def generate(self, system, user_text, params):
            # One fabricated number (within threshold), rest grounded.
            return "Peak throughput was 1234 tok/s. (Bogus 99999.)"

    monkeypatch.setattr(
        "vllm_benchmark.analysis.report.analyst.get_provider",
        lambda name, params: _MostlyHonest(),
    )
    report = generate_report([_result_cell()], _metadata(), provider="local")
    assert report.generated is True
    assert report.verification["checked"] == 2
    assert "~~99999~~" in report.markdown  # redacted but kept
    assert "1234 tok/s" in report.markdown


def test_generate_report_never_raises_on_bad_inputs():
    # Empty results + minimal metadata must still produce a report.
    report = generate_report([], {"server_info": {}, "system_info": {}}, provider="local")
    assert isinstance(report, AnalystReport)
    # No server reachable in the test env -> deterministic fallback.
    assert report.generated is False


def test_bundle_sha_recorded():
    report = generate_report([_result_cell()], _metadata(), provider="local")
    assert len(report.bundle_sha256) == 64
