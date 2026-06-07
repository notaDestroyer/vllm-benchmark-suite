"""Tests for the quality probe mode and its pure grading core."""

import math

from vllm_benchmark.analysis import quality
from vllm_benchmark.analysis.quality import grade_probe_response, run_quality
from vllm_benchmark.analysis.statistics import wilson_interval

# ---------------------------------------------------------------------------
# grade_probe_response
# ---------------------------------------------------------------------------


class TestGradeProbeResponse:
    def test_exact_match(self):
        assert grade_probe_response("HELLO", "HELLO", "exact") is True
        assert grade_probe_response("  HELLO  ", "HELLO", "exact") is True
        assert grade_probe_response("hello", "HELLO", "exact") is False

    def test_regex_match(self):
        assert grade_probe_response("The answer is 43.", r"\b43\b", "regex") is True
        assert grade_probe_response("It is 430", r"\b43\b", "regex") is False

    def test_regex_case_insensitive_inline(self):
        assert grade_probe_response("paris", r"(?i)\bparis\b", "regex") is True
        assert grade_probe_response("PARIS", r"(?i)\bparis\b", "regex") is True

    def test_none_response(self):
        assert grade_probe_response(None, "x", "exact") is False

    def test_invalid_regex_is_false(self):
        assert grade_probe_response("abc", r"(", "regex") is False

    def test_unknown_mode_raises(self):
        try:
            grade_probe_response("a", "a", "fuzzy")
            assert False, "expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Wilson CI math (textbook value)
# ---------------------------------------------------------------------------


def test_wilson_textbook_value():
    low, point, high = wilson_interval(8, 10, confidence=0.95)
    assert abs(point - 0.8) < 1e-9
    assert abs(low - 0.490) < 0.01
    assert abs(high - 0.943) < 0.01


# ---------------------------------------------------------------------------
# Probe rollup (mocked HTTP)
# ---------------------------------------------------------------------------


class _Cfg:
    api_url = "http://localhost:8000"
    model_name = "test-model"
    request_timeout = 30


class _ServerInfo:
    backend = "vllm"
    model_name = "test-model"


def test_probe_rollup_and_temperature(monkeypatch):
    """Probe mode grades responses, rolls up per-category Wilson CIs, temp=0."""
    seen_bodies = []

    def fake_post_json(url, body, timeout):
        seen_bodies.append(body)
        # Echo back a correct answer derived from the prompt so all probes pass.
        prompt = body["messages"][0]["content"]
        answer = "fallback"
        if "17 + 26" in prompt:
            answer = "43"
        elif "12 * 12" in prompt:
            answer = "144"
        elif "100 - 37" in prompt:
            answer = "63"
        elif "capital of France" in prompt:
            answer = "Paris"
        elif "Red Planet" in prompt:
            answer = "Mars"
        elif "chemical formula for water" in prompt:
            answer = "H2O"
        elif '"answer"' in prompt:
            answer = '{"answer": "ok"}'
        elif '"count"' in prompt:
            answer = '{"count": 3}'
        elif "HELLO" in prompt:
            answer = "HELLO"
        elif "single word: yes" in prompt:
            answer = "yes"
        return {"choices": [{"message": {"content": answer}}]}

    monkeypatch.setattr(quality, "_post_json", fake_post_json)

    result = run_quality("probe", _Cfg(), _ServerInfo())

    assert result["mode"] == "probe"
    assert result["status"] == "ok"
    assert result["temperature"] == 0.0
    # All probes constructed to pass -> perfect score.
    assert result["passed"] == result["total"]
    assert math.isclose(result["score"], 100.0)

    # Per-category rollups present with Wilson CIs.
    assert "arithmetic" in result["categories"]
    arith = result["categories"]["arithmetic"]
    assert arith["total"] == 3
    assert arith["passed"] == 3
    assert 0.0 <= arith["ci_low"] <= arith["pass_rate"] <= arith["ci_high"] <= 1.0

    # Temperature=0 used for every request.
    assert all(b["temperature"] == 0.0 for b in seen_bodies)


def test_probe_partial_failures(monkeypatch):
    """A wrong answer lowers the score and the category pass rate."""

    def fake_post_json(url, body, timeout):
        # Always return the wrong thing.
        return {"choices": [{"message": {"content": "nope"}}]}

    monkeypatch.setattr(quality, "_post_json", fake_post_json)
    result = run_quality("probe", _Cfg(), _ServerInfo())
    assert result["status"] == "ok"
    assert result["passed"] == 0
    assert result["score"] == 0.0


def test_off_mode_disabled():
    result = run_quality("off", _Cfg(), _ServerInfo())
    assert result == {"mode": "off", "status": "disabled"}
