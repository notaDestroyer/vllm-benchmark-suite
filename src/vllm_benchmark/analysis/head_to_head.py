"""vLLM-vs-SGLang (or any A/B) head-to-head comparison of two endpoints.

Given two benchmark runs that swept the *same* matrix against two
different serving endpoints (e.g. vLLM vs SGLang, or two vLLM versions),
this module compares them **per matched cell and per metric** and decides
a winner *only when the difference is real*.

For every matched ``(context_length, concurrent_users, prompt_type)`` cell
and every metric in :data:`_METRICS` the two per-cell samples are compared
with :func:`vllm_benchmark.analysis.statistics.compare_groups`.  A side is
declared the winner only when the difference is a *real difference*
(statistically significant **and** practically meaningful) **and** the
bootstrap confidence intervals of the two means do not overlap.  Otherwise
the cell/metric is reported as a **tie** — the module never fabricates an
unsupported "X is faster" claim.

Because a single benchmark cell usually yields one aggregate value per
metric, samples are drawn from the per-cell distribution where available
(``*_samples`` / percentile fields) and fall back to the point estimate;
ties are the honest default when there is not enough spread to separate
the endpoints.

All functions here are pure (no I/O, no printing) and fully unit-testable
on synthetic result dicts.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import Any, Optional

from vllm_benchmark.analysis.statistics import (
    bootstrap_ci,
    compare_groups,
    holm_adjusted_p,
)

#: Metrics compared per cell.  ``higher_is_better`` orients the winner.
_METRICS: dict[str, dict[str, Any]] = {
    "tokens_per_second": {"higher_is_better": True},
    "ttft_estimate": {"higher_is_better": False},
    "prefill_tps": {"higher_is_better": True},
    "decode_tps": {"higher_is_better": True},
    "latency_p99": {"higher_is_better": False},
}

#: Per-metric candidate sample sources, tried in order.  The first source
#: that yields a non-empty list of values wins; otherwise the point
#: estimate (single value) is used.
_SAMPLE_SOURCES: dict[str, tuple[str, ...]] = {
    "tokens_per_second": ("tokens_per_second_samples",),
    "ttft_estimate": ("ttft_samples", "ttft_estimate_samples"),
    "prefill_tps": ("prefill_tps_samples",),
    "decode_tps": ("decode_tps_samples",),
    "latency_p99": ("latency_samples",),
}


# ---------------------------------------------------------------------------
# Cell / sample helpers (pure)
# ---------------------------------------------------------------------------

def _cell_key(cell: dict) -> tuple:
    """Matching key for a result cell."""
    return (
        cell.get("context_length"),
        cell.get("concurrent_users"),
        cell.get("prompt_type"),
    )


def _endpoint_label(meta: dict) -> str:
    """Derive a short, stable label for an endpoint from its metadata."""
    meta = meta or {}
    server = meta.get("server_info", {}) or {}
    backend = server.get("backend") or server.get("engine")
    model = server.get("model_name") or meta.get("model")
    if backend and model:
        return f"{backend}:{model}"
    if backend:
        return str(backend)
    if model:
        return str(model)
    return "endpoint"


def _metric_sample(cell: dict, metric: str) -> list[float]:
    """Extract a per-cell sample for a metric.

    Prefers a raw per-request sample list (``*_samples``) when present so
    the comparison has real spread; otherwise falls back to the point
    estimate as a single-element list.
    """
    for source in _SAMPLE_SOURCES.get(metric, ()):  # explicit sample lists
        raw = cell.get(source)
        if isinstance(raw, (list, tuple)) and raw:
            return [float(v) for v in raw if v is not None]
    val = cell.get(metric)
    if val is None:
        return []
    return [float(val)]


def _index_by_cell(results: list[dict]) -> dict[tuple, dict]:
    """Index result cells by their matching key."""
    return {_cell_key(c): c for c in (results or [])}


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def head_to_head(
    results_a: list[dict],
    meta_a: dict,
    results_b: list[dict],
    meta_b: dict,
) -> dict:
    """Compare two endpoints cell-by-cell, metric-by-metric.

    For every matched ``(context, concurrency, prompt_type)`` cell present
    in *both* runs and every metric in :data:`_METRICS`, the two per-cell
    samples are compared with :func:`statistics.compare_groups`.  All raw
    Welch *p*-values across the whole comparison family are Holm-corrected
    together.  A winner (``"a"`` / ``"b"``) is assigned only when the
    difference is a real difference, the adjusted *p* < 0.05 **and** the
    bootstrap CIs do not overlap; everything else is a **tie**.

    Args:
        results_a: Per-cell result dicts for endpoint A.
        meta_a: Metadata for endpoint A (used for the label only).
        results_b: Per-cell result dicts for endpoint B.
        meta_b: Metadata for endpoint B.

    Returns:
        A dict with keys:

        * ``label_a`` / ``label_b`` — endpoint labels.
        * ``cells`` — list of per-cell dicts, each ``{"cell", "metrics"}``
          where ``metrics`` maps a metric name to a verdict dict
          (``winner`` in ``{"a", "b", None}``, ``tie``, ``cohens_d``,
          ``cliffs_delta``, ``welch_p``, ``adjusted_p``, ``ci_a``,
          ``ci_b``, ``n_a``, ``n_b``, ``mean_a``, ``mean_b``).
        * ``summary`` — per-metric win counts ``{"a", "b", "tie"}``.
        * ``n_matched_cells`` — number of matched cells.
        * ``n_comparisons`` — number of Holm-corrected comparisons.
    """
    label_a = _endpoint_label(meta_a)
    label_b = _endpoint_label(meta_b)

    a_by_cell = _index_by_cell(results_a)
    b_by_cell = _index_by_cell(results_b)
    matched_keys = [k for k in a_by_cell if k in b_by_cell]

    # Build the comparison family first so p-values can be Holm-corrected
    # across the whole set of (cell, metric) comparisons.
    raw: list[dict] = []
    for key in matched_keys:
        cell_a = a_by_cell[key]
        cell_b = b_by_cell[key]
        for metric, spec in _METRICS.items():
            a_vals = _metric_sample(cell_a, metric)
            b_vals = _metric_sample(cell_b, metric)
            cmp = compare_groups(a_vals, b_vals)
            ci_a = bootstrap_ci(a_vals, seed=0) if len(a_vals) > 1 else _point_ci(a_vals)
            ci_b = bootstrap_ci(b_vals, seed=0) if len(b_vals) > 1 else _point_ci(b_vals)
            raw.append({
                "cell": key,
                "metric": metric,
                "higher_is_better": spec["higher_is_better"],
                "compare": cmp,
                "ci_a": ci_a,
                "ci_b": ci_b,
                "n_a": len(a_vals),
                "n_b": len(b_vals),
                "mean_a": (sum(a_vals) / len(a_vals)) if a_vals else None,
                "mean_b": (sum(b_vals) / len(b_vals)) if b_vals else None,
            })

    adjusted = holm_adjusted_p([r["compare"]["welch_p"] for r in raw])

    # Assemble per-cell verdicts and running win/tie summary.
    cell_map: dict[tuple, dict[str, dict]] = {k: {} for k in matched_keys}
    summary: dict[str, dict[str, int]] = {
        m: {"a": 0, "b": 0, "tie": 0} for m in _METRICS
    }

    for entry, adj_p in zip(raw, adjusted):
        cmp = entry["compare"]
        ci_a = entry["ci_a"]
        ci_b = entry["ci_b"]
        cis_overlap = _cis_overlap(ci_a, ci_b)
        real = bool(cmp["real_difference"]) and adj_p < 0.05 and not cis_overlap
        tie = not real

        winner: Optional[str] = None
        if real:
            mean_a = entry["mean_a"] if entry["mean_a"] is not None else 0.0
            mean_b = entry["mean_b"] if entry["mean_b"] is not None else 0.0
            a_better = (mean_a > mean_b) if entry["higher_is_better"] else (mean_a < mean_b)
            winner = "a" if a_better else "b"

        metric = entry["metric"]
        summary[metric]["tie" if tie else winner] += 1

        cell_map[entry["cell"]][metric] = {
            "winner": winner,
            "tie": tie,
            "cohens_d": cmp["cohens_d"],
            "cliffs_delta": cmp["cliffs_delta"],
            "welch_p": cmp["welch_p"],
            "adjusted_p": adj_p,
            "ci_a": ci_a,
            "ci_b": ci_b,
            "n_a": entry["n_a"],
            "n_b": entry["n_b"],
            "mean_a": entry["mean_a"],
            "mean_b": entry["mean_b"],
        }

    cells = [
        {"cell": list(key), "metrics": cell_map[key]}
        for key in matched_keys
    ]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "cells": cells,
        "summary": summary,
        "n_matched_cells": len(matched_keys),
        "n_comparisons": len(raw),
    }


# ---------------------------------------------------------------------------
# Small numeric helpers (pure)
# ---------------------------------------------------------------------------

def _point_ci(values: list[float]) -> tuple[float, float, float]:
    """Return a degenerate ``(low, point, high)`` for 0/1-length samples."""
    if not values:
        return (0.0, 0.0, 0.0)
    v = float(values[0])
    return (v, v, v)


def _cis_overlap(
    ci_a: tuple[float, float, float],
    ci_b: tuple[float, float, float],
) -> bool:
    """Whether two ``(low, point, high)`` intervals overlap.

    Empty/degenerate intervals (all zeros) are treated as non-informative
    and therefore *do* overlap (forcing a tie), so a winner is never
    claimed without real interval evidence.
    """
    low_a, _, high_a = ci_a
    low_b, _, high_b = ci_b
    if (low_a, high_a) == (0.0, 0.0) or (low_b, high_b) == (0.0, 0.0):
        return True
    return not (high_a < low_b or high_b < low_a)
