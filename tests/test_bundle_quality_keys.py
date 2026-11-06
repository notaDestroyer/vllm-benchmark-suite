"""Quality and score facts reach the analyst bundle under the keys the
quality modes actually emit (``mean_kl`` / ``top1_agreement`` for KL,
``perplexity`` for perplexity) and the full ScoreBreakdown is carried.
"""

from __future__ import annotations

from vllm_benchmark.analysis.report.analyst import deterministic_report
from vllm_benchmark.analysis.report.bundle import allowed_numbers, build_bundle
from vllm_benchmark.analysis.scoring import ScoreBreakdown


def _results():
    return [{"context_length": 32_000, "concurrent_users": 1, "prompt_type": "classic",
             "tokens_per_second": 100.0, "avg_latency": 2.0, "ttft_estimate": 0.2}]


def test_kl_quality_reaches_bundle():
    meta = {"quality": {"mode": "kl", "status": "ok", "mean_kl": 0.0123, "top1_agreement": 0.97,
                        "n_positions": 512, "tokenizer_mismatch": False}}
    q = build_bundle(_results(), meta)["quality"]
    assert q["mean_kl"] == 0.0123
    assert q["top1_agreement"] == 0.97
    assert q["n_positions"] == 512
    assert q["tokenizer_mismatch"] is False
    assert 0.0123 in allowed_numbers(build_bundle(_results(), meta))


def test_legacy_kl_divergence_key_aliased():
    meta = {"quality": {"mode": "kl", "status": "ok", "kl_divergence": 0.5}}
    assert build_bundle(_results(), meta)["quality"]["mean_kl"] == 0.5


def test_perplexity_quality_reaches_bundle():
    meta = {"quality": {"mode": "perplexity", "status": "ok", "perplexity": 7.25, "n_samples": 10, "n_tokens": 400}}
    q = build_bundle(_results(), meta)["quality"]
    assert q["perplexity"] == 7.25 and q["n_tokens"] == 400


def test_probe_quality_carries_ci():
    meta = {"quality": {"mode": "probe", "status": "ok", "score": 80.0, "passed": 8, "total": 10,
                        "overall_ci_low": 0.49, "overall_ci_high": 0.94}}
    q = build_bundle(_results(), meta)["quality"]
    assert q["passed"] == 8 and q["overall_ci_high"] == 0.94


def test_deterministic_report_mentions_quality_numbers():
    meta = {"quality": {"mode": "kl", "status": "ok", "mean_kl": 0.02, "top1_agreement": 0.95}}
    text = deterministic_report(build_bundle(_results(), meta))
    assert "mean KL 0.02" in text and "top-1 agreement 0.95" in text


def test_score_breakdown_to_dict_and_full_bundle():
    sb = ScoreBreakdown(overall=8123, throughput=9000, latency=8000, efficiency=7000, energy=6000,
                        consistency=5000, grade="A", gpu_match="H100", gpu_percentile=77)
    assert sb.to_dict()["consistency"] == 5000
    s = build_bundle(_results(), {}, score=sb)["score"]
    assert s["overall"] == 8123 and s["grade"] == "A"
    assert s["energy"] == 6000 and s["consistency"] == 5000
    assert s["gpu_match"] == "H100" and s["gpu_percentile"] == 77


def test_score_plain_object_without_to_dict():
    class _S:
        overall = 100
        grade = "F"
        throughput = 10
    s = build_bundle(_results(), {}, score=_S())["score"]
    assert s == {"overall": 100, "grade": "F", "throughput": 10}
