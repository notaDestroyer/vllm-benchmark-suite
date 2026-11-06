"""Single implementations of helpers that used to be copy-pasted."""

from __future__ import annotations

from vllm_benchmark.analysis.bottleneck import top_verdict
from vllm_benchmark.analysis.statistics import cis_overlap, point_ci
from vllm_benchmark.reports.formatting import na


def test_top_verdict_prefers_confidence_then_highest_concurrency():
    verdicts = [
        {"cell": [32000, 1, "c"], "primary": "a", "confidence": "high"},
        {"cell": [32000, 16, "c"], "primary": "b", "confidence": "high"},
        {"cell": [32000, 32, "c"], "primary": "c", "confidence": "medium"},
    ]
    assert top_verdict(verdicts)["primary"] == "b"
    assert top_verdict([]) is None
    assert top_verdict(None) is None
    assert top_verdict([{"primary": "x"}])["primary"] == "x"  # no cell / confidence


def test_top_verdict_used_by_every_renderer():
    from vllm_benchmark.analysis.report import bundle
    from vllm_benchmark.reports import card, html_report, share
    for mod in (bundle, card, html_report, share):
        assert getattr(mod, "top_verdict") is top_verdict


def test_na_formatting():
    assert na(None) == "N/A"
    assert na(1.2345, " s", "{:.2f}") == "1.23 s"
    assert na("str", "", "{:.2f}") == "N/A"


def test_na_shared_by_renderers():
    from vllm_benchmark.reports import html_report, share, terminal
    for mod in (html_report, share, terminal):
        assert getattr(mod, "na") is na


def test_point_ci_and_overlap():
    assert point_ci([]) == (0.0, 0.0, 0.0)
    assert point_ci([3.0, 9.0]) == (3.0, 3.0, 3.0)
    assert cis_overlap((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) is True  # degenerate -> tie
    assert cis_overlap((1.0, 2.0, 3.0), (2.5, 3.0, 4.0)) is True
    assert cis_overlap((1.0, 2.0, 3.0), (3.5, 4.0, 5.0)) is False


def test_statistics_has_no_scipy_import_fallbacks():
    import inspect

    from vllm_benchmark.analysis import statistics
    src = inspect.getsource(statistics)
    assert "ImportError" not in src
    assert "_welch_ttest" not in src
