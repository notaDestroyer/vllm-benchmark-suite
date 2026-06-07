"""Tests for the numeric verifier.

Covers extraction (thousands separators, units), unit normalization
(``1.2s`` <-> ``1200ms``, ``0.78`` <-> ``78%``), catching fabricated
numbers, in-tolerance acceptance, redaction markup and the counters.
"""

from __future__ import annotations

from vllm_benchmark.analysis.report.verify import extract_numbers, verify_report

# ---------------------------------------------------------------------------
# extract_numbers
# ---------------------------------------------------------------------------

def test_extract_plain_and_units():
    nums = extract_numbers("Throughput 1234 tok/s, TTFT 250ms, MBU 78%.")
    raws = [r for r, _, _ in nums]
    assert "1234 tok/s" in raws
    assert "250ms" in raws
    assert "78%" in raws


def test_extract_thousands_separator():
    nums = extract_numbers("Peak was 1,234,567 tokens.")
    values = [v for _, v, _ in nums]
    assert 1234567.0 in values


def test_extract_decimal_and_seconds():
    nums = extract_numbers("Latency 1.2s observed.")
    assert any(abs(v - 1.2) < 1e-9 and unit == "s" for _, v, unit in nums)


# ---------------------------------------------------------------------------
# verify_report — basic supported / unsupported
# ---------------------------------------------------------------------------

def test_supported_number_passes():
    allowed = {1234.0}
    res = verify_report("Throughput was 1234 tok/s.", allowed)
    assert res["unsupported"] == []
    assert res["checked"] == 1
    assert "~~" not in res["redacted_text"]


def test_fabricated_number_caught_and_redacted():
    allowed = {1234.0}
    res = verify_report("Throughput was 1234 tok/s but latency was 9999.", allowed)
    assert "9999" in res["unsupported"]
    assert "~~9999~~" in res["redacted_text"]
    assert res["checked"] == 2
    assert len(res["unsupported"]) == 1


def test_in_tolerance_number_passes():
    # 1234.5 within 1% of an allowed 1234.0.
    allowed = {1234.0}
    res = verify_report("Throughput was 1234.5 tok/s.", allowed)
    assert res["unsupported"] == []


def test_out_of_tolerance_number_caught():
    allowed = {1234.0}
    res = verify_report("Throughput was 1300 tok/s.", allowed)
    assert "1300 tok/s" in res["unsupported"]


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

def test_seconds_to_ms_normalization():
    # allowed value is 1.2 (seconds); the draft renders it as 1200ms.
    allowed = {1.2}
    res = verify_report("TTFT was 1200ms.", allowed)
    assert res["unsupported"] == []


def test_ms_value_matches_seconds_allowed_both_directions():
    # allowed value is 1200 (ms-ish absolute); draft renders 1.2s.
    allowed = {1200.0}
    res = verify_report("Latency 1.2s.", allowed)
    # 1.2s normalizes to itself (1.2) — not 1200; but candidate includes raw 1.2
    # which won't match 1200. So this should be flagged. Confirm direction:
    assert "1.2s" in res["unsupported"]


def test_fraction_to_percent_normalization():
    allowed = {0.78}
    res = verify_report("Utilization reached 78%.", allowed)
    assert res["unsupported"] == []


def test_percent_value_unsupported_when_fraction_absent():
    allowed = {0.50}
    res = verify_report("Utilization reached 78%.", allowed)
    assert "78%" in res["unsupported"]


def test_thousands_separator_supported():
    allowed = {1234567.0}
    res = verify_report("Total tokens: 1,234,567.", allowed)
    assert res["unsupported"] == []


def test_billions_suffix_normalization():
    # allowed has 8e9 (raw param count); allowed_numbers would also carry 8.0
    allowed = {8.0}
    res = verify_report("The model has 8B active parameters.", allowed)
    assert res["unsupported"] == []


# ---------------------------------------------------------------------------
# Counters & multiple
# ---------------------------------------------------------------------------

def test_counters_with_multiple_numbers():
    allowed = {100.0, 200.0}
    res = verify_report("Values 100, 200, 300, 400.", allowed)
    assert res["checked"] == 4
    assert set(res["unsupported"]) == {"300", "400"}
    assert res["redacted_text"].count("~~") == 4  # two struck spans


def test_no_numbers_is_clean():
    res = verify_report("No figures here at all.", {1.0})
    assert res["checked"] == 0
    assert res["unsupported"] == []
    assert res["redacted_text"] == "No figures here at all."


def test_redacted_text_preserves_surrounding_prose():
    allowed = {5.0}
    res = verify_report("a 5 b 7 c", allowed)
    assert res["redacted_text"] == "a 5 b ~~7~~ c"
