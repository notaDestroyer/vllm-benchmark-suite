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

    Method & assumptions: Welch's two-sample *t*-test does not assume equal
    variances but does assume approximately normal sampling distributions of
    the mean.  When either input has zero variance (e.g. all-identical
    measurements) the *t* statistic is undefined; rather than propagate a
    ``nan`` *p*-value we treat the comparison as not significant
    (``p_value == 1.0``), which is the conservative outcome.

    Args:
        group_a: Measurements from condition A.
        group_b: Measurements from condition B.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with keys:

        * **significant** -- ``True`` if *p* < *alpha*
        * **p_value** -- two-tailed *p*-value (never ``nan``)
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

    # --- Degenerate / zero-variance guard ---
    # When both groups have zero variance the Welch statistic is undefined
    # (SciPy returns a nan p-value with a precision-loss warning).  Decide the
    # outcome analytically: identical means -> not different; differing means
    # with no spread -> a perfectly separated, significant difference.
    if var_a == 0.0 and var_b == 0.0:
        if mean_a == mean_b:
            return {
                "significant": False,
                "p_value": 1.0,
                "effect_size": 0.0,
                "t_statistic": 0.0,
                "interpretation": "no significant difference",
            }
        return {
            "significant": True,
            "p_value": 0.0,
            "effect_size": effect_size,
            "t_statistic": float("inf"),
            "interpretation": "large effect",
        }

    # --- Welch's t-test ---
    try:
        from scipy import stats as sp_stats

        t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
        t_stat = float(t_stat)
        p_value = float(p_value)
    except ImportError:
        t_stat, p_value = _welch_ttest(a, b)

    # Final safety net: never surface a nan p-value.
    if not np.isfinite(p_value):
        p_value = 1.0
        t_stat = 0.0 if not np.isfinite(t_stat) else t_stat

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


# ---------------------------------------------------------------------------
# 7. Bootstrap & parametric confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: list[float],
    statistic: Any = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    method: str = "bca",
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an arbitrary statistic.

    Method & assumptions: resampling bootstrap.  When ``method == "bca"``
    the bias-corrected and accelerated (BCa) interval of Efron is used,
    which adjusts the percentile interval for both median bias and
    skewness (acceleration) via a jackknife estimate.  BCa makes no
    distributional assumption beyond i.i.d. sampling but degenerates when
    the data carry no variance: if every value is identical (or the
    jackknife acceleration is non-finite) the interval collapses to the
    point estimate ``(x, x, x)`` and a plain percentile bootstrap is used
    as the fallback.  SciPy's :func:`scipy.stats.bootstrap` is preferred
    for the BCa case when applicable.

    Args:
        data: Observed numeric values.
        statistic: Callable mapping a 1-D array to a scalar.  Defaults to
            :func:`numpy.mean`.
        confidence: Confidence level in (0, 1).  Default 0.95.
        n_resamples: Number of bootstrap resamples.  Default 10 000.
        method: ``"bca"`` (default) or ``"percentile"``.
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(low, point, high)`` with ``low <= point <= high``.

        * Empty input  -> ``(0.0, 0.0, 0.0)``
        * Single value -> ``(x, x, x)``
        * Zero-variance data -> ``(x, x, x)``
    """
    if not data:
        return (0.0, 0.0, 0.0)

    arr = np.asarray(data, dtype=np.float64)
    point = float(statistic(arr))

    if len(arr) == 1:
        v = float(arr[0])
        return (v, v, v)

    # Degenerate / zero-variance data -> collapse to the point estimate.
    if float(np.ptp(arr)) == 0.0:
        return (point, point, point)

    alpha = 1.0 - confidence

    # Prefer SciPy's vetted BCa implementation when requested and valid.
    if method == "bca":
        try:
            from scipy import stats as sp_stats

            rng = np.random.default_rng(seed)
            res = sp_stats.bootstrap(
                (arr,),
                statistic,
                n_resamples=n_resamples,
                confidence_level=confidence,
                method="BCa",
                vectorized=False,
                random_state=rng,
            )
            low = float(res.confidence_interval.low)
            high = float(res.confidence_interval.high)
            if np.isfinite(low) and np.isfinite(high):
                low, high = min(low, high), max(low, high)
                low = min(low, point)
                high = max(high, point)
                return (low, point, high)
        except Exception:
            # Fall through to the manual percentile bootstrap below.
            pass

    # Manual percentile bootstrap fallback.
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        boot[i] = statistic(rng.choice(arr, size=n, replace=True))

    low = float(np.percentile(boot, 100 * alpha / 2))
    high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    low = min(low, point)
    high = max(high, point)
    return (low, point, high)


def t_ci(
    data: list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Student-*t* confidence interval for the mean (small-*n* fallback).

    Method & assumptions: classical parametric interval using the
    Student-*t* distribution, appropriate when *n* is small and the
    underlying distribution is approximately normal.  Uses SciPy's
    ``t.ppf`` when available and a manual normal approximation otherwise.

    Args:
        data: Observed numeric values.
        confidence: Confidence level in (0, 1).  Default 0.95.

    Returns:
        ``(low, point, high)`` with ``low <= point <= high``.

        * Empty input  -> ``(0.0, 0.0, 0.0)``
        * Single value -> ``(x, x, x)``
        * Zero-variance data -> ``(x, x, x)``
    """
    if not data:
        return (0.0, 0.0, 0.0)

    arr = np.asarray(data, dtype=np.float64)
    point = float(np.mean(arr))
    n = len(arr)

    if n == 1 or float(np.ptp(arr)) == 0.0:
        return (point, point, point)

    se = float(np.std(arr, ddof=1)) / np.sqrt(n)

    try:
        from scipy import stats as sp_stats

        t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    except ImportError:
        # Normal approximation when SciPy is unavailable.
        z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(confidence, 2), 1.96)
        t_crit = z

    margin = t_crit * se
    return (point - margin, point, point + margin)


# ---------------------------------------------------------------------------
# 8. Non-parametric tests & effect sizes
# ---------------------------------------------------------------------------


def mann_whitney_u(a: list[float], b: list[float]) -> dict[str, float]:
    """Mann-Whitney *U* rank-sum test for two independent samples.

    Method & assumptions: a non-parametric test of whether one
    distribution is stochastically greater than the other; no normality
    assumption.  Delegates to :func:`scipy.stats.mannwhitneyu` (two-sided).
    Identical or degenerate inputs (empty, or all-tied values) cannot
    reject the null, so the *p*-value is reported as ``1.0`` rather than
    ``nan``.

    Args:
        a: Measurements from condition A.
        b: Measurements from condition B.

    Returns:
        Dictionary with ``u`` (the *U* statistic) and ``p_value``.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if arr_a.size == 0 or arr_b.size == 0:
        return {"u": 0.0, "p_value": 1.0}

    # All values tied across both groups -> no evidence of a difference.
    combined = np.concatenate([arr_a, arr_b])
    if float(np.ptp(combined)) == 0.0:
        return {"u": float(arr_a.size * arr_b.size / 2.0), "p_value": 1.0}

    try:
        from scipy import stats as sp_stats

        u_stat, p_value = sp_stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        u_stat = float(u_stat)
        p_value = float(p_value)
    except ImportError:  # pragma: no cover - SciPy is a hard dependency here
        u_stat, p_value = 0.0, 1.0

    if not np.isfinite(p_value):
        p_value = 1.0
    return {"u": u_stat, "p_value": p_value}


def cliffs_delta(a: list[float], b: list[float]) -> dict[str, Any]:
    """Cliff's delta non-parametric effect size.

    Method & assumptions: ``delta`` is the difference between the
    probability that a randomly chosen value from *a* exceeds one from *b*
    and the reverse probability; it ranges in [-1, 1] and is robust to
    non-normality.  Magnitude thresholds follow Romano et al.:
    ``|d| < 0.147`` negligible, ``< 0.33`` small, ``< 0.474`` medium,
    otherwise large.

    Args:
        a: Measurements from condition A.
        b: Measurements from condition B.

    Returns:
        Dictionary with ``delta`` and ``magnitude``.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if arr_a.size == 0 or arr_b.size == 0:
        return {"delta": 0.0, "magnitude": "negligible"}

    # delta = (#(a>b) - #(a<b)) / (na * nb), computed via broadcasting.
    diff = arr_a[:, None] - arr_b[None, :]
    greater = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    delta = (greater - less) / (arr_a.size * arr_b.size)

    magnitude = _cliffs_magnitude(abs(delta))
    return {"delta": float(delta), "magnitude": magnitude}


def _cliffs_magnitude(abs_delta: float) -> str:
    """Map an absolute Cliff's delta to a magnitude label."""
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.33:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's *d* standardized mean difference (pooled SD).

    Method & assumptions: the difference in means divided by the pooled
    standard deviation; assumes comparable variances for interpretability.
    Returns ``0.0`` when the pooled SD is zero (e.g. both groups identical)
    so the result is never ``nan``.

    Args:
        a: Measurements from condition A.
        b: Measurements from condition B.

    Returns:
        The signed effect size (sign follows ``mean(a) - mean(b)``).
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    n_a, n_b = arr_a.size, arr_b.size
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))

    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd <= 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_sd)


# ---------------------------------------------------------------------------
# 9. Multiple-comparison correction
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down rejection flags.

    Method & assumptions: controls the family-wise error rate.  The
    *p*-values are sorted ascending; the *k*-th smallest (0-indexed) is
    compared against ``alpha / (m - k)``.  Once a hypothesis fails to
    reject, all subsequent (larger) ones are retained as well.

    Args:
        p_values: Raw *p*-values.
        alpha: Family-wise significance level.

    Returns:
        A list of booleans (rejected/not) in the **original** input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            # Step-down: stop rejecting once a hypothesis is retained.
            break
    return rejected


def holm_adjusted_p(p_values: list[float]) -> list[float]:
    """Holm-adjusted *p*-values (monotone, clipped to 1.0).

    Method & assumptions: each sorted *p*-value is scaled by its remaining
    family size ``(m - k)`` and made monotonically non-decreasing via a
    running maximum, then clipped to ``[0, 1]``.

    Args:
        p_values: Raw *p*-values.

    Returns:
        Adjusted *p*-values in the **original** input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_values[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# ---------------------------------------------------------------------------
# 10. Binomial proportion interval
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Method & assumptions: the Wilson score interval is well-behaved for
    small samples and proportions near 0 or 1, unlike the naive normal
    (Wald) interval.  The point estimate returned is the observed
    proportion ``successes / n``.

    Args:
        successes: Number of successes.
        n: Number of trials.
        confidence: Confidence level in (0, 1).  Default 0.95.

    Returns:
        ``(low, point, high)``.  ``n == 0`` -> ``(0.0, 0.0, 0.0)``.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)

    try:
        from scipy import stats as sp_stats

        z = float(sp_stats.norm.ppf((1 + confidence) / 2))
    except ImportError:
        z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(confidence, 2), 1.96)

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (float(low), float(p_hat), float(high))


# ---------------------------------------------------------------------------
# 11. Group comparison convenience (used by PR5 comparisons)
# ---------------------------------------------------------------------------


def compare_groups(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
    effect_threshold: float = 0.33,
) -> dict[str, Any]:
    """Compare two groups with parametric + non-parametric tests.

    Combines Welch's *t*-test, the Mann-Whitney *U* test, Cohen's *d* and
    Cliff's delta into a single verdict.  A difference is flagged as a
    "real difference" only when it is statistically significant **and** the
    effect size is practically meaningful (Cohen's |d| above
    ``effect_threshold``), guarding against statistically-significant but
    trivial differences.

    Args:
        a: Measurements from condition A.
        b: Measurements from condition B.
        alpha: Significance level.  Default 0.05.
        effect_threshold: Minimum |Cohen's d| for practical significance.

    Returns:
        Dictionary with ``welch_p``, ``mann_whitney_p``, ``cohens_d``,
        ``cliffs_delta``, ``cliffs_magnitude``, ``significant`` and
        ``real_difference``.
    """
    welch = is_statistically_significant(a, b, alpha=alpha)
    mw = mann_whitney_u(a, b)
    d = cohens_d(a, b)
    cliff = cliffs_delta(a, b)

    significant = bool(welch["significant"])
    real_difference = significant and abs(d) >= effect_threshold

    return {
        "welch_p": float(welch["p_value"]),
        "mann_whitney_p": float(mw["p_value"]),
        "cohens_d": float(d),
        "cliffs_delta": float(cliff["delta"]),
        "cliffs_magnitude": cliff["magnitude"],
        "significant": significant,
        "real_difference": bool(real_difference),
    }
