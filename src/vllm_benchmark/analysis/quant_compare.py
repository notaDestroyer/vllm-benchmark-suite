"""Cross-quantization (and cross-model) comparison of benchmark runs.

Given several loaded results-JSON packages — typically the *same* model
served at different quantizations, or different models — this module
compares matched ``(context_length, concurrent_users, prompt_type)``
cells across runs on the metrics that matter (throughput, peak VRAM,
single-user MBU, and quality) and produces:

* a **ranking** of the runs by headline throughput, and
* **per-metric significance** between every pair of runs, using
  :func:`vllm_benchmark.analysis.statistics.compare_groups` with the
  family of *p*-values **Holm-corrected** across the whole comparison
  set.

Honesty first: a pair is only called a *winner/loser* when the
difference is a "real difference" (significant **and** practically
meaningful) *and* survives Holm correction.  When the confidence
intervals overlap, or the effect is not real, the pair is reported as a
**tie** — never a fabricated winner.

All functions here are pure (no I/O, no printing) and fully unit-testable
on synthetic run dicts.

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

#: Metrics compared across quant runs.  ``higher_is_better`` controls how
#: rankings and per-pair winners are oriented.
_METRICS: dict[str, dict[str, Any]] = {
    "tokens_per_second": {"higher_is_better": True},
    "peak_mem": {"higher_is_better": False},
    "mbu": {"higher_is_better": True},
    "quality": {"higher_is_better": True},
}


# ---------------------------------------------------------------------------
# Run-level extraction helpers (pure)
# ---------------------------------------------------------------------------

def _run_label(run: dict) -> str:
    """Derive a short, stable label for a run (model + quant)."""
    meta = run.get("metadata", {}) or {}
    server = meta.get("server_info", {}) or {}
    model = server.get("model_name") or meta.get("model") or "unknown"
    quant = server.get("quantization") or "none"
    return f"{model} [{quant}]"


def _cell_key(cell: dict) -> tuple:
    """Matching key for a result cell."""
    return (
        cell.get("context_length"),
        cell.get("concurrent_users"),
        cell.get("prompt_type"),
    )


def _peak_mem(cell: dict) -> Optional[float]:
    """Peak VRAM used for a cell (MB/GB as recorded), if present."""
    return cell.get("max_mem_used") or cell.get("avg_mem_used")


def _cell_mbu(cell: dict, run: dict) -> Optional[float]:
    """Single-user MBU for a cell, taken from per-run bottleneck verdicts.

    The bottleneck verdicts carry ``mbu`` per cell; we match by the same
    ``(context, concurrency, prompt_type)`` key.
    """
    meta = run.get("metadata", {}) or {}
    for v in meta.get("bottlenecks", []) or []:
        cell_id = v.get("cell")
        if cell_id and tuple(cell_id) == _cell_key(cell) and v.get("mbu") is not None:
            return float(v["mbu"])
    return None


def _run_quality(run: dict) -> Optional[float]:
    """Extract an overall quality score for the run, if measured."""
    meta = run.get("metadata", {}) or {}
    quality = meta.get("quality")
    if isinstance(quality, dict):
        score = quality.get("score")
        if score is not None:
            return float(score)
    return None


def _metric_values(run: dict, metric: str) -> list[float]:
    """Collect a metric's per-cell sample for a run.

    For ``tokens_per_second`` / ``peak_mem`` / ``mbu`` the sample is the
    set of matched-cell values; for ``quality`` (a single run-level score)
    the sample is a single value repeated is avoided — the run-level score
    is returned as a one-element list (compared as a point estimate).
    """
    results = run.get("results", []) or []
    if metric == "tokens_per_second":
        return [float(c["tokens_per_second"]) for c in results if c.get("tokens_per_second")]
    if metric == "peak_mem":
        return [float(v) for c in results if (v := _peak_mem(c)) is not None]
    if metric == "mbu":
        return [float(v) for c in results if (v := _cell_mbu(c, run)) is not None]
    if metric == "quality":
        q = _run_quality(run)
        return [q] if q is not None else []
    return []


def _matched_metric_values(
    run_a: dict,
    run_b: dict,
    metric: str,
) -> tuple[list[float], list[float]]:
    """Return per-cell samples for two runs aligned on common cells.

    Only cells present (with a value for the metric) in *both* runs are
    used, so the comparison is paired across identical configurations.
    Quality is run-level and falls back to the unmatched single values.
    """
    if metric == "quality":
        return _metric_values(run_a, metric), _metric_values(run_b, metric)

    def _by_cell(run: dict) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        for c in run.get("results", []) or []:
            if metric == "tokens_per_second":
                val = c.get("tokens_per_second")
            elif metric == "peak_mem":
                val = _peak_mem(c)
            elif metric == "mbu":
                val = _cell_mbu(c, run)
            else:
                val = None
            if val is not None:
                out[_cell_key(c)] = float(val)
        return out

    a_map = _by_cell(run_a)
    b_map = _by_cell(run_b)
    common = [k for k in a_map if k in b_map]
    return [a_map[k] for k in common], [b_map[k] for k in common]


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_quant_runs(runs: list[dict]) -> dict:
    """Compare multiple benchmark runs across quantizations / models.

    For each metric in :data:`_METRICS` and each unordered pair of runs,
    the matched-cell samples are compared with
    :func:`statistics.compare_groups`.  The raw Welch *p*-values are
    collected across **all** (metric, pair) comparisons and Holm-corrected
    together; a pair is only assigned a winner when the difference is a
    real difference *and* its adjusted *p* < 0.05.  Otherwise the pair is
    a **tie** (also when bootstrap CIs of the two means overlap).

    Args:
        runs: A list of loaded results-JSON packages, each shaped like
            ``{"metadata": {...}, "results": [cell, ...]}``.

    Returns:
        A dict with keys:

        * ``runs`` — list of ``{"label", "tokens_per_second_mean",
          "peak_mem_mean", "mbu_mean", "quality"}`` per input run.
        * ``ranking`` — run labels ordered best-to-worst by mean
          throughput.
        * ``metrics`` — per-metric dict mapping ``"A vs B"`` to a verdict
          dict (``winner``, ``tie``, ``cohens_d``, ``welch_p``,
          ``adjusted_p``, ``ci_a``, ``ci_b``, ``n_a``, ``n_b``).
        * ``n_comparisons`` — number of Holm-corrected comparisons.
    """
    labels = [_run_label(r) for r in runs]

    # Per-run summary means (for ranking / charting).
    run_summaries: list[dict] = []
    for label, run in zip(labels, runs):
        summary: dict[str, Any] = {"label": label}
        for metric in _METRICS:
            vals = _metric_values(run, metric)
            mean_val = sum(vals) / len(vals) if vals else None
            summary[f"{metric}_mean" if metric != "quality" else "quality"] = mean_val
        run_summaries.append(summary)

    # Ranking by headline throughput (None sinks to the bottom).
    ranking = sorted(
        labels,
        key=lambda lab: (
            run_summaries[labels.index(lab)].get("tokens_per_second_mean") or float("-inf")
        ),
        reverse=True,
    )

    # Build the full comparison family, then Holm-correct the p-values.
    raw: list[dict] = []
    for metric, spec in _METRICS.items():
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a_vals, b_vals = _matched_metric_values(runs[i], runs[j], metric)
                cmp = compare_groups(a_vals, b_vals)
                ci_a = bootstrap_ci(a_vals, seed=0) if len(a_vals) > 1 else _point_ci(a_vals)
                ci_b = bootstrap_ci(b_vals, seed=0) if len(b_vals) > 1 else _point_ci(b_vals)
                raw.append({
                    "metric": metric,
                    "higher_is_better": spec["higher_is_better"],
                    "i": i,
                    "j": j,
                    "compare": cmp,
                    "ci_a": ci_a,
                    "ci_b": ci_b,
                    "n_a": len(a_vals),
                    "n_b": len(b_vals),
                    "mean_a": (sum(a_vals) / len(a_vals)) if a_vals else None,
                    "mean_b": (sum(b_vals) / len(b_vals)) if b_vals else None,
                })

    adjusted = holm_adjusted_p([r["compare"]["welch_p"] for r in raw])

    metrics_out: dict[str, dict[str, dict]] = {m: {} for m in _METRICS}
    for entry, adj_p in zip(raw, adjusted):
        label_a = labels[entry["i"]]
        label_b = labels[entry["j"]]
        pair_key = f"{label_a} vs {label_b}"
        cmp = entry["compare"]
        ci_a = entry["ci_a"]
        ci_b = entry["ci_b"]

        cis_overlap = _cis_overlap(ci_a, ci_b)
        real = bool(cmp["real_difference"]) and adj_p < 0.05 and not cis_overlap
        tie = not real

        winner: Optional[str] = None
        if real:
            mean_a = entry["mean_a"] or 0.0
            mean_b = entry["mean_b"] or 0.0
            a_better = (mean_a > mean_b) if entry["higher_is_better"] else (mean_a < mean_b)
            winner = label_a if a_better else label_b

        metrics_out[entry["metric"]][pair_key] = {
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
        }

    return {
        "runs": run_summaries,
        "ranking": ranking,
        "metrics": metrics_out,
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
