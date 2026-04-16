"""Tests for auto-diagnostics engine."""

from vllm_benchmark.analysis.diagnostics import DiagnosticEngine, Diagnostic


def _make_result(**overrides):
    base = {
        "context_length": 32000,
        "concurrent_users": 4,
        "tokens_per_second": 500,
        "avg_latency": 2.0,
        "max_latency": 3.0,
        "std_latency": 0.3,
        "ttft_estimate": 0.2,
        "throughput_per_user": 125,
        "successful": 4,
        "failed": 0,
    }
    base.update(overrides)
    return base


def test_no_issues():
    engine = DiagnosticEngine()
    results = [_make_result()]
    diagnostics = engine.analyze(results)
    severities = {d.severity for d in diagnostics}
    assert "critical" not in severities
    assert "warning" not in severities


def test_request_failures():
    engine = DiagnosticEngine()
    results = [_make_result(failed=3)]
    diagnostics = engine.analyze(results)
    assert any(d.severity == "critical" for d in diagnostics)
    assert any("failed" in d.message.lower() for d in diagnostics)


def test_high_latency_variance():
    engine = DiagnosticEngine()
    results = [_make_result(avg_latency=2.0, max_latency=10.0)]
    diagnostics = engine.analyze(results)
    assert any(d.title == "Extreme tail latency detected" for d in diagnostics)


def test_slow_ttft():
    engine = DiagnosticEngine()
    results = [_make_result(ttft_estimate=3.0)]  # 3000ms
    diagnostics = engine.analyze(results)
    assert any("Time to First Token" in d.title for d in diagnostics)


def test_excellent_throughput():
    engine = DiagnosticEngine()
    results = [_make_result(tokens_per_second=800)]
    diagnostics = engine.analyze(results)
    assert any(d.severity == "success" and "throughput" in d.title.lower() for d in diagnostics)


def test_empty_results():
    engine = DiagnosticEngine()
    diagnostics = engine.analyze([])
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == "info"
