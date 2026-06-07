"""Tests for the application-fitness profiles.

Drives synthetic metric sets through :func:`assess_fitness` and asserts
correct Good/Marginal/Poor grading for all eight profiles, that missing
signals yield ``N/A`` (never a fabricated grade), that the limiting factor
is attributed, and that :func:`fitness_verdict` produces a sensible
one-liner.  Also checks that :func:`build_advisory` populates
``Advisory.fitness``.
"""

from __future__ import annotations

from vllm_benchmark.analysis.advisor import build_advisory
from vllm_benchmark.analysis.fitness import (
    assess_fitness,
    fitness_verdict,
)
from vllm_benchmark.core.backends.base import ServerInfo


def _cell(**kw) -> dict:
    """Build a generative result cell with sensible defaults."""
    base = {
        "concurrent_users": 1,
        "context_length": 8000,
        "tokens_per_second": 1000.0,
        "ttft_estimate": 0.3,
        "avg_latency": 1.0,
        "latency_p50": 1.0,
        "latency_p99": 2.0,
        "decode_tps_mean": 50.0,
        "prefill_tps_mean": 1000.0,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# interactive_chat
# ---------------------------------------------------------------------------


def test_interactive_chat_good():
    grades = assess_fitness([_cell(ttft_estimate=0.10, decode_tps_mean=40)], {}, None, None)
    g = grades["interactive_chat"]
    assert g.grade == "Good"
    assert g.limiting_factor is None


def test_interactive_chat_marginal_ttft():
    grades = assess_fitness([_cell(ttft_estimate=0.5)], {}, None, None)
    g = grades["interactive_chat"]
    assert g.grade == "Marginal"
    assert g.limiting_factor == "TTFT"


def test_interactive_chat_poor_ttft():
    grades = assess_fitness([_cell(ttft_estimate=2.0)], {}, None, None)
    assert grades["interactive_chat"].grade == "Poor"


def test_interactive_chat_decode_floor_demotes():
    # Great TTFT but decode below the floor demotes Good -> Marginal.
    grades = assess_fitness([_cell(ttft_estimate=0.05, decode_tps_mean=5.0)], {}, None, None)
    g = grades["interactive_chat"]
    assert g.grade == "Marginal"
    assert g.limiting_factor == "decode throughput"


def test_interactive_chat_na_without_ttft():
    cell = _cell()
    del cell["ttft_estimate"]
    grades = assess_fitness([cell], {}, None, None)
    assert grades["interactive_chat"].grade == "N/A"


# ---------------------------------------------------------------------------
# rag_long_context
# ---------------------------------------------------------------------------


def test_rag_good():
    grades = assess_fitness([_cell(prefill_tps_mean=3000.0)], {}, None, None)
    assert grades["rag_long_context"].grade == "Good"


def test_rag_marginal():
    grades = assess_fitness([_cell(prefill_tps_mean=800.0)], {}, None, None)
    g = grades["rag_long_context"]
    assert g.grade == "Marginal"
    assert g.limiting_factor == "prefill throughput"


def test_rag_poor():
    grades = assess_fitness([_cell(prefill_tps_mean=100.0)], {}, None, None)
    assert grades["rag_long_context"].grade == "Poor"


def test_rag_na_without_prefill():
    cell = _cell()
    del cell["prefill_tps_mean"]
    grades = assess_fitness([cell], {}, None, None)
    assert grades["rag_long_context"].grade == "N/A"


# ---------------------------------------------------------------------------
# batch_offline
# ---------------------------------------------------------------------------


def test_batch_good():
    grades = assess_fitness([_cell(tokens_per_second=5000.0)], {}, None, None)
    assert grades["batch_offline"].grade == "Good"


def test_batch_marginal():
    grades = assess_fitness([_cell(tokens_per_second=900.0)], {}, None, None)
    assert grades["batch_offline"].grade == "Marginal"


def test_batch_poor():
    grades = assess_fitness([_cell(tokens_per_second=100.0)], {}, None, None)
    assert grades["batch_offline"].grade == "Poor"


def test_batch_cost_annotation():
    grades = assess_fitness(
        [_cell(tokens_per_second=5000.0, cost_per_1m_tokens=0.42)], {}, None, None
    )
    assert "$0.42/1M tok" in grades["batch_offline"].detail


# ---------------------------------------------------------------------------
# agentic_tooluse
# ---------------------------------------------------------------------------


def test_agentic_good():
    grades = assess_fitness(
        [_cell(ttft_estimate=0.2, latency_p50=1.0, latency_p99=1.5)], {}, None, None
    )
    assert grades["agentic_tooluse"].grade == "Good"


def test_agentic_tail_demotes_good():
    # Good TTFT but a wide tail demotes to Marginal.
    grades = assess_fitness(
        [_cell(ttft_estimate=0.2, latency_p50=1.0, latency_p99=3.0)], {}, None, None
    )
    g = grades["agentic_tooluse"]
    assert g.grade == "Marginal"
    assert g.limiting_factor == "latency tail (p99/p50)"


def test_agentic_extreme_tail_poor():
    grades = assess_fitness(
        [_cell(ttft_estimate=0.2, latency_p50=1.0, latency_p99=10.0)], {}, None, None
    )
    g = grades["agentic_tooluse"]
    assert g.grade == "Poor"
    assert g.limiting_factor == "latency tail (p99/p50)"


# ---------------------------------------------------------------------------
# code_completion
# ---------------------------------------------------------------------------


def test_code_good():
    grades = assess_fitness([_cell(ttft_estimate=0.05)], {}, None, None)
    assert grades["code_completion"].grade == "Good"


def test_code_marginal():
    grades = assess_fitness([_cell(ttft_estimate=0.2)], {}, None, None)
    assert grades["code_completion"].grade == "Marginal"


def test_code_poor():
    grades = assess_fitness([_cell(ttft_estimate=0.5)], {}, None, None)
    assert grades["code_completion"].grade == "Poor"


# ---------------------------------------------------------------------------
# structured_output_fc
# ---------------------------------------------------------------------------


def test_structured_good():
    pm = {"structured": {"schema_adherence_rate": 0.99, "best_ttft": 0.1}}
    grades = assess_fitness([_cell()], pm, None, None)
    assert grades["structured_output_fc"].grade == "Good"


def test_structured_marginal():
    pm = {"structured": {"schema_adherence_rate": 0.85}}
    grades = assess_fitness([_cell()], pm, None, None)
    g = grades["structured_output_fc"]
    assert g.grade == "Marginal"
    assert g.limiting_factor == "schema adherence"


def test_structured_poor():
    pm = {"structured": {"schema_adherence_rate": 0.5}}
    grades = assess_fitness([_cell()], pm, None, None)
    assert grades["structured_output_fc"].grade == "Poor"


def test_structured_ttft_demotes():
    pm = {"structured": {"schema_adherence_rate": 0.99, "best_ttft": 2.0}}
    grades = assess_fitness([_cell()], pm, None, None)
    g = grades["structured_output_fc"]
    assert g.grade == "Marginal"
    assert g.limiting_factor == "TTFT"


def test_structured_na_without_run():
    grades = assess_fitness([_cell()], {}, None, None)
    assert grades["structured_output_fc"].grade == "N/A"


# ---------------------------------------------------------------------------
# embeddings_rerank
# ---------------------------------------------------------------------------


def test_embeddings_good():
    pm = {"embeddings": {"peak_docs_per_second": 500.0, "best_avg_latency": 0.05}}
    grades = assess_fitness([], pm, None, None)
    assert grades["embeddings_rerank"].grade == "Good"


def test_embeddings_marginal():
    pm = {"embeddings": {"peak_docs_per_second": 80.0, "best_avg_latency": 0.2}}
    grades = assess_fitness([], pm, None, None)
    assert grades["embeddings_rerank"].grade == "Marginal"


def test_embeddings_poor():
    pm = {"embeddings": {"peak_docs_per_second": 10.0}}
    grades = assess_fitness([], pm, None, None)
    assert grades["embeddings_rerank"].grade == "Poor"


def test_embeddings_na_without_run():
    grades = assess_fitness([_cell()], {}, None, None)
    assert grades["embeddings_rerank"].grade == "N/A"


# ---------------------------------------------------------------------------
# speculative_decoding
# ---------------------------------------------------------------------------


def test_speculative_good():
    si = ServerInfo(backend="vllm", speculative={"acceptance_rate": 0.75})
    grades = assess_fitness([_cell()], {}, si, None)
    assert grades["speculative_decoding"].grade == "Good"


def test_speculative_marginal():
    si = ServerInfo(backend="vllm", speculative={"acceptance_rate": 0.4})
    grades = assess_fitness([_cell()], {}, si, None)
    assert grades["speculative_decoding"].grade == "Marginal"


def test_speculative_poor():
    si = ServerInfo(backend="vllm", speculative={"acceptance_rate": 0.1})
    grades = assess_fitness([_cell()], {}, si, None)
    assert grades["speculative_decoding"].grade == "Poor"


def test_speculative_na_without_config():
    si = ServerInfo(backend="vllm", speculative=None)
    grades = assess_fitness([_cell()], {}, si, None)
    assert grades["speculative_decoding"].grade == "N/A"


def test_speculative_na_without_acceptance():
    # Spec decoding configured but no acceptance rate reported -> N/A.
    si = ServerInfo(backend="vllm", speculative={"method": "ngram"})
    grades = assess_fitness([_cell()], {}, si, None)
    assert grades["speculative_decoding"].grade == "N/A"


# ---------------------------------------------------------------------------
# Coverage of all eight profiles + verdict
# ---------------------------------------------------------------------------


def test_all_eight_profiles_present():
    grades = assess_fitness([_cell()], {}, None, None)
    expected = {
        "interactive_chat",
        "rag_long_context",
        "batch_offline",
        "agentic_tooluse",
        "code_completion",
        "structured_output_fc",
        "embeddings_rerank",
        "speculative_decoding",
    }
    assert set(grades) == expected


def test_verdict_names_best_and_avoid():
    # Strong interactive (low TTFT) but weak batch throughput.
    cells = [_cell(ttft_estimate=0.05, tokens_per_second=100.0, prefill_tps_mean=100.0)]
    grades = assess_fitness(cells, {}, None, None)
    verdict = fitness_verdict(grades)
    assert verdict.startswith("Best for:")
    assert "avoid for:" in verdict
    assert "limited by" in verdict


def test_verdict_all_na():
    # No usable signals at all -> verdict reports insufficient data.
    grades = assess_fitness([], {}, None, None)
    verdict = fitness_verdict(grades)
    assert "insufficient data" in verdict


# ---------------------------------------------------------------------------
# build_advisory wiring
# ---------------------------------------------------------------------------


def test_build_advisory_populates_fitness():
    cells = [
        _cell(concurrent_users=1, ttft_estimate=0.1),
        _cell(concurrent_users=4, ttft_estimate=0.3),
        _cell(concurrent_users=8, ttft_estimate=0.5),
    ]
    si = ServerInfo(backend="vllm", task="generate")
    advisory = build_advisory(cells, None, si, None)
    assert advisory.fitness is not None
    assert "verdict" in advisory.fitness
    assert "profiles" in advisory.fitness
    assert len(advisory.fitness["profiles"]) == 8
    # Without structured/embeddings runs, those profiles are N/A.
    assert advisory.fitness["profiles"]["structured_output_fc"]["grade"] == "N/A"
    assert advisory.fitness["profiles"]["embeddings_rerank"]["grade"] == "N/A"


def test_build_advisory_with_workload_metrics():
    cells = [_cell()]
    si = ServerInfo(backend="vllm", task="generate")
    pm = {
        "structured": {"schema_adherence_rate": 0.97, "best_ttft": 0.1},
        "embeddings": {"peak_docs_per_second": 400.0, "best_avg_latency": 0.05},
    }
    advisory = build_advisory(cells, None, si, None, profile_metrics=pm)
    profiles = advisory.fitness["profiles"]
    assert profiles["structured_output_fc"]["grade"] == "Good"
    assert profiles["embeddings_rerank"]["grade"] == "Good"
