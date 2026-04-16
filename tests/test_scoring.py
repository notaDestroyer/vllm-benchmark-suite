"""Tests for composite benchmark scoring."""

from vllm_benchmark.analysis.scoring import ScoreBreakdown, VLLMScore


def _make_result(**overrides):
    base = {
        "tokens_per_second": 500,
        "avg_latency": 2.0,
        "std_latency": 0.3,
        "throughput_per_user": 125,
        "tokens_per_watt": 5.0,
    }
    base.update(overrides)
    return base


def test_score_range():
    scorer = VLLMScore()
    results = [_make_result()]
    score = scorer.calculate(results)
    assert 0 <= score.overall <= 10000
    assert 0 <= score.throughput <= 10000
    assert 0 <= score.latency <= 10000
    assert 0 <= score.efficiency <= 10000
    assert 0 <= score.energy <= 10000
    assert 0 <= score.consistency <= 10000


def test_grade_assignment():
    scorer = VLLMScore()
    # High performance → high score → good grade
    high = [_make_result(tokens_per_second=2000, avg_latency=0.5, throughput_per_user=200, tokens_per_watt=10)]
    score_high = scorer.calculate(high)
    assert score_high.grade in ("S", "A")

    # Low performance → low score → bad grade
    low = [_make_result(tokens_per_second=50, avg_latency=9.0, throughput_per_user=10, tokens_per_watt=0.5)]
    score_low = scorer.calculate(low)
    assert score_low.grade in ("D", "F")


def test_higher_throughput_higher_score():
    scorer = VLLMScore()
    low = scorer.calculate([_make_result(tokens_per_second=100)])
    high = scorer.calculate([_make_result(tokens_per_second=1000)])
    assert high.throughput > low.throughput


def test_single_result():
    scorer = VLLMScore()
    score = scorer.calculate([_make_result()])
    assert isinstance(score, ScoreBreakdown)
    assert score.grade in ("S", "A", "B", "C", "D", "F")


def test_empty_results():
    scorer = VLLMScore()
    score = scorer.calculate([])
    assert score.overall >= 0


def test_format_display():
    scorer = VLLMScore()
    score = scorer.calculate([_make_result()])
    display = scorer.format_score_display(score)
    assert "vLLM Benchmark Score" in display
    assert score.grade in display
