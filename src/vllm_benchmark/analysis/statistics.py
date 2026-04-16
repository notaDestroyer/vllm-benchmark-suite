"""Scientific statistical analysis for vLLM benchmark results.

Provides confidence intervals, outlier detection, significance testing,
robust descriptive statistics, steady-state detection, and cross-iteration
aggregation.  All functions are pure (no side effects, no printing) and
rely on NumPy for core calculations.  SciPy is used when available but
every function degrades gracefully to a manual implementation when SciPy
is absent.

Author: amit
License: MIT
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 1. Confidence Intervals
# ---------------------------------------------------------------------------


def confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute the mean and a symmetric confidence interval.

    Uses the Student-*t* distribution when SciPy is available; otherwise
    falls back to a percentile bootstrap (10 000 resamples).

    Args:
        values: Observed numeric values.
        confidence: Confidence level in (0, 1).  Default is 0.95.

    Returns:
        ``(mean, lower_bound, upper_bound)``

        * Empty input  -> ``(0.0, 0.0, 0.0)``
        * Single value -> ``(val, val, val)``
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.asarray(values, dtype=np.float64)

    if len(arr) == 1:
        v = float(arr[0])
        return (v, v, v)

    sample_mean = float(np.mean(arr))
    n = len(arr)

    try:
        from scipy import stats as sp_stats

        se = float(np.std(arr, ddof=1)) / np.sqrt(n)
        t_crit = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_crit * se
        return (sample_mean, sample_mean - margin, sample_mean + margin)
    except ImportError:
        pass

    # Bootstrap fallback
    rng = np.random.default_rng(seed=42)
    n_boot = 10_000
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot_means[i] = np.mean(rng.choice(arr, size=n, replace=True))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (sample_mean, lower, upper)


# ---------------------------------------------------------------------------
# 2. Outlier Detection
# ---------------------------------------------------------------------------


def detect_outliers(
    values: list[float],
    method: str = "iqr",
) -> dict[str, Any]:
    """Detect outliers using the interquartile-range (IQR) method.

    Fences are placed at ``Q1 - 1.5 * IQR`` and ``Q3 + 1.5 * IQR``.

    Args:
        values: Observed numeric values.
        method: Detection method.  Currently only ``"iqr"`` is supported.

    Returns:
        Dictionary with keys:

        * **clean** -- values inside the fences
        * **outliers** -- values outside the fences
        * **lower_fence** -- lower bound
        * **upper_fence** -- upper bound
        * **n_outliers** -- count of outlier values
    """
    if method != "iqr":
        raise ValueError(f"Unsupported outlier detection method: {method!r}")

    if not values:
        return {
            "clean": [],
            "outliers": [],
            "lower_fence": 0.0,
            "upper_fence": 0.0,
            "n_outliers": 0,
        }

    arr = np.asarray(values, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    mask = (arr >= lower_fence) & (arr <= upper_fence)
    clean = arr[mask].tolist()
    outliers = arr[~mask].tolist()

    return {
        "clean": clean,
        "outliers": outliers,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_outliers": len(outliers),
    }


# ---------------------------------------------------------------------------
# 3. Statistical Significance Testing
# ---------------------------------------------------------------------------


def is_statistically_significant(
    group_a: list[float],
    group_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Welch's *t*-test for independent samples with unequal variance.

    Computes the *t* statistic, *p*-value, and Cohen's *d* effect size.

    Args:
        group_a: Measurements from condition A.
        group_b: Measurements from condition B.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with keys:

        * **significant** -- ``True`` if *p* < *alpha*
        * **p_value** -- two-tailed *p*-value
        * **effect_size** -- Cohen's *d*
        * **t_statistic** -- Welch *t* value
        * **interpretation** -- human-readable label
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    if len(a) < 2 or len(b) < 2:
        return {
            "significant": False,
            "p_value": 1.0,
            "effect_size": 0.0,
            "t_statistic": 0.0,
            "interpretation": "no significant difference",
        }

    # --- Cohen's d (pooled SD) ---
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    effect_size = float(abs(mean_a - mean_b) / pooled_sd) if pooled_sd > 0 else 0.0

    # --- Welch's t-test ---
    try:
        from scipy import stats as sp_stats

        t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
        t_stat = float(t_stat)
        p_value = float(p_value)
    except ImportError:
        t_stat, p_value = _welch_ttest(a, b)

    significant = p_value < alpha

    # Interpretation based on significance and effect size
    if not significant:
        interpretation = "no significant difference"
    elif effect_size < 0.2:
        interpretation = "no significant difference"
    elif effect_size < 0.5:
        interpretation = "small effect"
    elif effect_size < 0.8:
        interpretation = "medium effect"
    else:
        interpretation = "large effect"

    return {
        "significant": significant,
        "p_value": p_value,
        "effect_size": effect_size,
        "t_statistic": t_stat,
        "interpretation": interpretation,
    }


def _welch_ttest(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Manual Welch's *t*-test when SciPy is unavailable.

    Approximates the two-tailed *p*-value using the Welch--Satterthwaite
    degrees of freedom and the regularised incomplete beta function
    implemented via a simple numerical integration.

    Returns:
        ``(t_statistic, p_value)``
    """
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return (0.0, 1.0)

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    p_value = _two_tailed_p_from_t(abs(t_stat), df)
    return (float(t_stat), float(p_value))


def _two_tailed_p_from_t(t: float, df: float) -> float:
    """Approximate two-tailed *p*-value from |t| and degrees of freedom.

    Uses numerical integration of the Student-*t* PDF via the composite
    Simpson rule.  Accurate to roughly four decimal places for typical
    benchmark sample sizes.
    """
    import math

    def _t_pdf(x: float, nu: float) -> float:
        coeff = math.gamma((nu + 1) / 2) / (
            math.sqrt(nu * math.pi) * math.gamma(nu / 2)
        )
        return coeff * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)

    # Integrate from |t| to a large upper bound (effectively infinity)
    upper = max(abs(t) + 50, 100.0)
    n_steps = 2000
    a_val = abs(t)
    h = (upper - a_val) / n_steps

    # Composite Simpson's rule
    total = _t_pdf(a_val, df) + _t_pdf(upper, df)
    for i in range(1, n_steps):
        x = a_val + i * h
        total += (4 if i % 2 == 1 else 2) * _t_pdf(x, df)
    one_tail = (h / 3) * total

    p = 2 * one_tail
    return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# 4. Robust Descriptive Statistics
# ---------------------------------------------------------------------------


def compute_robust_stats(values: list[float]) -> dict[str, Any]:
    """Compute comprehensive descriptive statistics with diagnostics.

    Includes central tendency, dispersion, quantiles, confidence interval,
    outlier detection, and quality warnings.

    Args:
        values: Observed numeric values.

    Returns:
        Dictionary with keys: ``mean``, ``median``, ``std``, ``cv``,
        ``ci_lower``, ``ci_upper``, ``p50``, ``p90``, ``p95``, ``p99``,
        ``iqr``, ``n``, ``min``, ``max``, ``outlier_detection``,
        ``warnings``.
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "iqr": 0.0,
            "n": 0,
            "min": 0.0,
            "max": 0.0,
            "outlier_detection": detect_outliers([]),
            "warnings": [],
        }

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)

    sample_mean = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    cv = (sample_std / abs(sample_mean)) if sample_mean != 0 else 0.0

    _, ci_lower, ci_upper = confidence_interval(values)
    outlier_info = detect_outliers(values)

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    warnings: list[str] = []
    if n < 5:
        warnings.append("sample size too small for reliable inference")
    if cv > 0.5:
        warnings.append("high variance — results may not be stable")

    return {
        "mean": sample_mean,
        "median": float(np.median(arr)),
        "std": sample_std,
        "cv": cv,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "iqr": q3 - q1,
        "n": n,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "outlier_detection": outlier_info,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 5. Steady-State (Warmup) Detection
# ---------------------------------------------------------------------------


def steady_state_check(
    time_series_values: list[float],
    window: int = 5,
) -> dict[str, Any]:
    """Check whether the tail of a time series has stabilised.

    A window of the last *window* values is considered stable when its
    coefficient of variation (CV) is below 0.1.  This is useful for
    determining whether a warmup phase has completed.

    Args:
        time_series_values: Ordered measurements (e.g. per-iteration
            throughput).
        window: Number of trailing values to evaluate.

    Returns:
        Dictionary with keys:

        * **stable** -- ``True`` if CV < 0.1
        * **cv** -- coefficient of variation of the window
        * **n_values** -- number of values actually used (may be less
          than *window* if the series is shorter)
    """
    if not time_series_values:
        return {"stable": False, "cv": float("inf"), "n_values": 0}

    tail = time_series_values[-window:]
    arr = np.asarray(tail, dtype=np.float64)
    n = len(arr)

    if n < 2:
        return {"stable": False, "cv": float("inf"), "n_values": n}

    mean_val = float(np.mean(arr))
    if mean_val == 0:
        return {"stable": True, "cv": 0.0, "n_values": n}

    std_val = float(np.std(arr, ddof=1))
    cv = std_val / abs(mean_val)

    return {"stable": cv < 0.1, "cv": cv, "n_values": n}


# ---------------------------------------------------------------------------
# 6. Cross-Iteration Aggregation
# ---------------------------------------------------------------------------

# Metrics for which robust statistics are computed during aggregation.
_AGGREGATION_METRICS = (
    "tokens_per_second",
    "avg_latency",
    "ttft_estimate",
    "throughput_per_user",
)


def aggregate_iterations(
    iteration_results: list[list[dict]],
) -> list[dict]:
    """Aggregate multiple iterations of the same test configuration.

    Groups individual result dicts by
    ``(context_length, concurrent_users, prompt_type)`` and, for each
    group, computes robust statistics across iterations for the key
    performance metrics.

    Args:
        iteration_results: A list of iterations, where each iteration is
            itself a list of result dicts.  Every result dict is expected
            to contain at least ``context_length``, ``concurrent_users``,
            and ``prompt_type`` keys.

    Returns:
        A list of aggregated result dicts, one per unique configuration
        group.  Each dict contains, for every metric in
        ``_AGGREGATION_METRICS``, the median value under the metric name
        and additional ``{metric}_ci_lower`` / ``{metric}_ci_upper``
        fields.  Robust-stats summaries are stored under
        ``{metric}_stats``.
    """
    if not iteration_results:
        return []

    # Collect per-group, per-metric values across iterations.
    # Key: (context_length, concurrent_users, prompt_type)
    # Value: {metric_name: [values across iterations]}
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for iteration in iteration_results:
        for result in iteration:
            key = (
                result.get("context_length", 0),
                result.get("concurrent_users", 1),
                result.get("prompt_type", ""),
            )
            for metric in _AGGREGATION_METRICS:
                val = result.get(metric)
                if val is not None:
                    grouped[key][metric].append(float(val))

    # Build aggregated output.
    aggregated: list[dict] = []
    for (ctx, users, ptype), metrics_map in grouped.items():
        entry: dict[str, Any] = {
            "context_length": ctx,
            "concurrent_users": users,
            "prompt_type": ptype,
        }

        for metric in _AGGREGATION_METRICS:
            metric_values = metrics_map.get(metric, [])
            if not metric_values:
                continue

            stats = compute_robust_stats(metric_values)

            # Use the median as the representative value (robust to outliers).
            entry[metric] = stats["median"]
            entry[f"{metric}_ci_lower"] = stats["ci_lower"]
            entry[f"{metric}_ci_upper"] = stats["ci_upper"]
            entry[f"{metric}_stats"] = stats

        aggregated.append(entry)

    return aggregated
