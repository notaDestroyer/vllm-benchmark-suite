"""Application-fitness profiles.

Grades a benchmark run against eight deployment profiles — interactive
chat, RAG long-context, batch/offline, agentic tool-use, code completion,
structured-output / function-calling, embeddings/rerank and speculative
decoding.  Each profile is graded ``"Good" | "Marginal" | "Poor" | "N/A"``
with the *limiting factor* named.

Design principles:

* All thresholds live in :data:`THRESHOLDS` — a single documented table,
  not magic numbers scattered through the logic.
* Missing the *required* signal for a profile yields grade ``"N/A"`` with a
  reason; a grade is never fabricated from absent data.

Author: amit
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.analysis.model_intel import ModelProfile

Grade = Literal["Good", "Marginal", "Poor", "N/A"]


# ---------------------------------------------------------------------------
# Threshold table (single source of truth)
# ---------------------------------------------------------------------------

#: All numeric bands used by the fitness profiles.  Latencies are in
#: *seconds* (matching the result-cell ``ttft_estimate`` / ``avg_latency``
#: units); throughputs are tokens/sec or docs/sec.
THRESHOLDS: dict[str, dict[str, float]] = {
    "interactive_chat": {
        # TTFT bands (seconds) and a secondary decode-throughput floor.
        "ttft_good": 0.200,
        "ttft_marginal": 1.000,
        "decode_tps_floor": 10.0,
    },
    "rag_long_context": {
        # Prefill throughput bands (tokens/sec) at long context.
        "prefill_good": 2000.0,
        "prefill_marginal": 500.0,
    },
    "batch_offline": {
        # Peak aggregate throughput bands (tokens/sec).
        "throughput_good": 2000.0,
        "throughput_marginal": 500.0,
    },
    "agentic_tooluse": {
        # TTFT bands (seconds) plus a tail-consistency (p99/p50) ceiling.
        "ttft_good": 0.500,
        "ttft_marginal": 1.500,
        "tail_ratio_good": 2.0,
        "tail_ratio_marginal": 4.0,
    },
    "code_completion": {
        # Very tight TTFT bands (seconds) for inline completion.
        "ttft_good": 0.100,
        "ttft_marginal": 0.300,
    },
    "structured_output_fc": {
        # Schema-adherence bands (fraction) plus a TTFT marginal ceiling.
        "adherence_good": 0.95,
        "adherence_marginal": 0.80,
        "ttft_marginal": 1.000,
    },
    "embeddings_rerank": {
        # Docs/sec bands plus a latency marginal ceiling (seconds).
        "docs_good": 200.0,
        "docs_marginal": 50.0,
        "latency_marginal": 1.000,
    },
    "speculative_decoding": {
        # Token acceptance-rate bands (fraction).
        "acceptance_good": 0.60,
        "acceptance_marginal": 0.30,
    },
}

#: One-line human descriptions of each profile, for verdicts.
PROFILE_LABELS: dict[str, str] = {
    "interactive_chat": "interactive chat",
    "rag_long_context": "RAG / long-context",
    "batch_offline": "batch / offline throughput",
    "agentic_tooluse": "agentic tool-use",
    "code_completion": "code completion",
    "structured_output_fc": "structured output / function calling",
    "embeddings_rerank": "embeddings / rerank",
    "speculative_decoding": "speculative decoding",
}


@dataclass
class FitnessGrade:
    """A single application-fitness verdict.

    Attributes:
        profile: Profile identifier (e.g. ``"interactive_chat"``).
        grade: ``"Good" | "Marginal" | "Poor" | "N/A"``.
        limiting_factor: The metric/condition that capped the grade, or a
            reason string when ``grade == "N/A"``.
        detail: Human-readable one-line explanation.
        confidence: ``"high" | "medium" | "low"``.
    """

    profile: str
    grade: Grade
    limiting_factor: Optional[str]
    detail: str
    confidence: Literal["high", "medium", "low"] = "medium"

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation of this grade."""
        return {
            "profile": self.profile,
            "grade": self.grade,
            "limiting_factor": self.limiting_factor,
            "detail": self.detail,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

def _na(profile: str, reason: str) -> FitnessGrade:
    """Build an ``N/A`` grade with a stated reason."""
    return FitnessGrade(
        profile=profile,
        grade="N/A",
        limiting_factor=reason,
        detail=f"No signal: {reason}.",
        confidence="low",
    )


def _min_metric(results: list[dict], key: str) -> Optional[float]:
    """Return the minimum non-``None`` value of ``key`` across cells."""
    vals = [r[key] for r in results if r.get(key) is not None]
    return min(vals) if vals else None


def _max_metric(results: list[dict], key: str) -> Optional[float]:
    """Return the maximum non-``None`` value of ``key`` across cells."""
    vals = [r[key] for r in results if r.get(key) is not None]
    return max(vals) if vals else None


def _best_single_user_ttft(results: list[dict]) -> Optional[float]:
    """Return the best (lowest) TTFT, preferring single-user cells.

    Interactive / code / agentic latency is best characterized at low
    concurrency; fall back to the global best when no single-user cell is
    present.
    """
    single = [r for r in results if r.get("concurrent_users") == 1]
    pool = single or results
    return _min_metric(pool, "ttft_estimate")


def _tail_ratio(results: list[dict]) -> Optional[float]:
    """Return the best (lowest) p99/p50 latency ratio across cells."""
    ratios: list[float] = []
    for r in results:
        p50 = r.get("latency_p50")
        p99 = r.get("latency_p99")
        if p50 and p99 and p50 > 0:
            ratios.append(p99 / p50)
    return min(ratios) if ratios else None


def _best_decode_tps(results: list[dict]) -> Optional[float]:
    """Return the best decode throughput across cells, if measured."""
    for key in ("decode_tps_mean", "decode_tps_p50", "decode_tps"):
        val = _max_metric(results, key)
        if val is not None:
            return val
    return None


def _best_prefill_tps(results: list[dict]) -> Optional[float]:
    """Return the best prefill throughput across cells, if measured."""
    for key in ("prefill_tps_mean", "prefill_tps_p50", "prefill_tps"):
        val = _max_metric(results, key)
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# Per-profile assessors
# ---------------------------------------------------------------------------

def _grade_interactive_chat(results: list[dict]) -> FitnessGrade:
    """Grade interactive chat: primary TTFT, secondary decode floor."""
    p = "interactive_chat"
    t = THRESHOLDS[p]
    ttft = _best_single_user_ttft(results)
    if ttft is None:
        return _na(p, "no TTFT measured (non-streaming run?)")

    decode = _best_decode_tps(results)
    if ttft < t["ttft_good"]:
        grade: Grade = "Good"
        limiting = None
    elif ttft < t["ttft_marginal"]:
        grade = "Marginal"
        limiting = "TTFT"
    else:
        grade = "Poor"
        limiting = "TTFT"

    # Secondary: slow decode demotes an otherwise-Good verdict.
    if grade == "Good" and decode is not None and decode < t["decode_tps_floor"]:
        grade = "Marginal"
        limiting = "decode throughput"

    detail = f"best TTFT {ttft * 1000:.0f}ms"
    if decode is not None:
        detail += f", decode {decode:.0f} tok/s"
    return FitnessGrade(p, grade, limiting, detail)


def _grade_rag_long_context(
    results: list[dict],
    model_profile: Optional["ModelProfile"],
) -> FitnessGrade:
    """Grade RAG/long-context: prefill throughput + KV-fits-at-max-context."""
    p = "rag_long_context"
    t = THRESHOLDS[p]
    prefill = _best_prefill_tps(results)
    if prefill is None:
        return _na(p, "no prefill throughput measured")

    if prefill >= t["prefill_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif prefill >= t["prefill_marginal"]:
        grade = "Marginal"
        limiting = "prefill throughput"
    else:
        grade = "Poor"
        limiting = "prefill throughput"

    detail = f"prefill {prefill:.0f} tok/s"

    # KV-fits-at-max-context demotion when the profile says context is
    # capped well below what RAG needs.
    max_ctx = _max_metric(results, "context_length")
    if model_profile is not None:
        cap = getattr(model_profile, "max_position_embeddings", None)
        if cap and max_ctx and max_ctx > cap:
            grade = "Poor" if grade != "N/A" else grade
            limiting = "context exceeds model max"
            detail += f"; tested {int(max_ctx)} > model max {int(cap)}"
    return FitnessGrade(p, grade, limiting, detail)


def _grade_batch_offline(results: list[dict]) -> FitnessGrade:
    """Grade batch/offline: peak aggregate throughput, cost annotation."""
    p = "batch_offline"
    t = THRESHOLDS[p]
    peak = _max_metric(results, "tokens_per_second")
    if peak is None:
        return _na(p, "no throughput measured")

    if peak >= t["throughput_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif peak >= t["throughput_marginal"]:
        grade = "Marginal"
        limiting = "peak throughput"
    else:
        grade = "Poor"
        limiting = "peak throughput"

    detail = f"peak {peak:.0f} tok/s"
    cost = _min_metric(results, "cost_per_1m_tokens")
    if cost is not None:
        detail += f", ${cost:.2f}/1M tok"
    return FitnessGrade(p, grade, limiting, detail)


def _grade_agentic_tooluse(results: list[dict]) -> FitnessGrade:
    """Grade agentic tool-use: TTFT plus tail consistency (p99/p50)."""
    p = "agentic_tooluse"
    t = THRESHOLDS[p]
    ttft = _best_single_user_ttft(results)
    if ttft is None:
        return _na(p, "no TTFT measured (non-streaming run?)")

    tail = _tail_ratio(results)
    if ttft < t["ttft_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif ttft < t["ttft_marginal"]:
        grade = "Marginal"
        limiting = "TTFT"
    else:
        grade = "Poor"
        limiting = "TTFT"

    # Tail consistency can only demote (a wobbly tail hurts multi-step
    # agent loops).
    if tail is not None:
        if tail > t["tail_ratio_marginal"] and grade != "Poor":
            grade = "Poor"
            limiting = "latency tail (p99/p50)"
        elif tail > t["tail_ratio_good"] and grade == "Good":
            grade = "Marginal"
            limiting = "latency tail (p99/p50)"

    detail = f"best TTFT {ttft * 1000:.0f}ms"
    if tail is not None:
        detail += f", p99/p50 {tail:.1f}x"
    return FitnessGrade(p, grade, limiting, detail)


def _grade_code_completion(results: list[dict]) -> FitnessGrade:
    """Grade code completion: very tight TTFT."""
    p = "code_completion"
    t = THRESHOLDS[p]
    ttft = _best_single_user_ttft(results)
    if ttft is None:
        return _na(p, "no TTFT measured (non-streaming run?)")

    if ttft < t["ttft_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif ttft < t["ttft_marginal"]:
        grade = "Marginal"
        limiting = "TTFT"
    else:
        grade = "Poor"
        limiting = "TTFT"
    return FitnessGrade(p, grade, limiting, f"best TTFT {ttft * 1000:.0f}ms")


def _grade_structured_output(
    results: list[dict],
    profile_metrics: dict[str, Any],
) -> FitnessGrade:
    """Grade structured output / function calling: adherence + TTFT.

    ``N/A`` when no structured workload was run (no ``schema_adherence_rate``
    in ``profile_metrics``).
    """
    p = "structured_output_fc"
    t = THRESHOLDS[p]
    struct = profile_metrics.get("structured") or {}
    adherence = struct.get("schema_adherence_rate")
    if adherence is None:
        return _na(p, "no structured-output run")

    if adherence >= t["adherence_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif adherence >= t["adherence_marginal"]:
        grade = "Marginal"
        limiting = "schema adherence"
    else:
        grade = "Poor"
        limiting = "schema adherence"

    detail = f"schema adherence {adherence * 100:.0f}%"
    tool = struct.get("tool_call_correctness")
    if tool is not None:
        detail += f", tool correctness {tool * 100:.0f}%"

    # TTFT can demote a Good adherence verdict.
    ttft = struct.get("best_ttft")
    if ttft is None:
        ttft = _best_single_user_ttft(results)
    if grade == "Good" and ttft is not None and ttft >= t["ttft_marginal"]:
        grade = "Marginal"
        limiting = "TTFT"
    return FitnessGrade(p, grade, limiting, detail)


def _grade_embeddings_rerank(profile_metrics: dict[str, Any]) -> FitnessGrade:
    """Grade embeddings/rerank: docs/sec + latency.

    ``N/A`` when no embeddings workload was run.
    """
    p = "embeddings_rerank"
    t = THRESHOLDS[p]
    emb = profile_metrics.get("embeddings") or {}
    docs = emb.get("peak_docs_per_second")
    if docs is None:
        return _na(p, "no embeddings run")

    if docs >= t["docs_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif docs >= t["docs_marginal"]:
        grade = "Marginal"
        limiting = "docs/sec"
    else:
        grade = "Poor"
        limiting = "docs/sec"

    detail = f"peak {docs:.0f} docs/s"
    latency = emb.get("best_avg_latency")
    if latency is not None:
        detail += f", best latency {latency * 1000:.0f}ms"
        if grade == "Good" and latency >= t["latency_marginal"]:
            grade = "Marginal"
            limiting = "latency"
    return FitnessGrade(p, grade, limiting, detail)


def _grade_speculative(
    results: list[dict],
    server_info: Any,
) -> FitnessGrade:
    """Grade speculative decoding: acceptance rate + effective tok/s.

    ``N/A`` when the server is not running speculative decoding.
    """
    p = "speculative_decoding"
    t = THRESHOLDS[p]
    spec = getattr(server_info, "speculative", None) if server_info is not None else None
    if not spec:
        return _na(p, "no speculative decoding configured")

    acceptance = (
        spec.get("acceptance_rate")
        if isinstance(spec, dict)
        else None
    )
    if acceptance is None:
        return _na(p, "speculative decoding on but no acceptance rate reported")

    if acceptance >= t["acceptance_good"]:
        grade: Grade = "Good"
        limiting: Optional[str] = None
    elif acceptance >= t["acceptance_marginal"]:
        grade = "Marginal"
        limiting = "acceptance rate"
    else:
        grade = "Poor"
        limiting = "acceptance rate"

    detail = f"acceptance {acceptance * 100:.0f}%"
    eff = _best_decode_tps(results)
    if eff is not None:
        detail += f", effective {eff:.0f} tok/s"
    return FitnessGrade(p, grade, limiting, detail)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assess_fitness(
    results: list[dict],
    profile_metrics: dict[str, Any],
    server_info: Any,
    model_profile: Optional["ModelProfile"],
) -> dict[str, FitnessGrade]:
    """Grade a run against all eight application-fitness profiles.

    Args:
        results: Generative result cells from the run (may be empty).
        profile_metrics: Workload summaries keyed by name, e.g.
            ``{"structured": {...}, "embeddings": {...}}``.  Missing keys
            cause the relevant profile to be ``N/A``.
        server_info: The :class:`ServerInfo` (for speculative info).
        model_profile: The :class:`ModelProfile` (for KV/context fit).

    Returns:
        A mapping ``profile -> FitnessGrade`` covering all eight profiles.
    """
    profile_metrics = profile_metrics or {}
    return {
        "interactive_chat": _grade_interactive_chat(results),
        "rag_long_context": _grade_rag_long_context(results, model_profile),
        "batch_offline": _grade_batch_offline(results),
        "agentic_tooluse": _grade_agentic_tooluse(results),
        "code_completion": _grade_code_completion(results),
        "structured_output_fc": _grade_structured_output(results, profile_metrics),
        "embeddings_rerank": _grade_embeddings_rerank(profile_metrics),
        "speculative_decoding": _grade_speculative(results, server_info),
    }


def fitness_verdict(grades: dict[str, FitnessGrade]) -> str:
    """Produce a one-line "best for / avoid for" verdict.

    Picks the strongest graded profile as the recommended use and the
    weakest *non-N/A* profile as the one to avoid (naming its limiting
    factor).  ``N/A`` profiles never appear in the verdict.

    Args:
        grades: The mapping returned by :func:`assess_fitness`.

    Returns:
        A single-line verdict string.
    """
    rank = {"Good": 3, "Marginal": 2, "Poor": 1}
    graded = [g for g in grades.values() if g.grade in rank]
    if not graded:
        return "Best for: insufficient data; avoid for: insufficient data."

    best = max(graded, key=lambda g: rank[g.grade])
    worst = min(graded, key=lambda g: rank[g.grade])
    best_label = PROFILE_LABELS.get(best.profile, best.profile)
    worst_label = PROFILE_LABELS.get(worst.profile, worst.profile)

    avoid = f"avoid for: {worst_label}"
    if worst.limiting_factor:
        avoid += f" (limited by {worst.limiting_factor})"
    return f"Best for: {best_label}; {avoid}."
