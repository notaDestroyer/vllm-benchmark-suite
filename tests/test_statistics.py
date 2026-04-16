"""Tests for the statistical analysis module."""

import math

from vllm_benchmark.analysis.statistics import (
    aggregate_iterations,
    compute_robust_stats,
    confidence_interval,
    detect_outliers,
    is_statistically_significant,
    steady_state_check,
)

# ---------------------------------------------------------------------------
# 1. confidence_interval
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    """Tests for confidence_interval()."""

    def test_known_constant_distribution(self):
        """Identical values should yield a very tight CI around the value."""
        mean, lower, upper = confidence_interval([10, 10, 10, 10, 10])
        assert mean == 10.0
        # With zero variance the CI collapses to a single point.
        assert abs(lower - 10.0) < 1e-9
        assert abs(upper - 10.0) < 1e-9

    def test_empty_list(self):
        """Empty input returns (0, 0, 0)."""
        assert confidence_interval([]) == (0.0, 0.0, 0.0)

    def test_single_value(self):
        """A single observation returns (val, val, val)."""
        assert confidence_interval([42.0]) == (42.0, 42.0, 42.0)

    def test_wide_vs_narrow_distribution(self):
        """A wider distribution should produce a wider CI."""
        narrow = [10.0, 10.1, 9.9, 10.0, 10.1]
        wide = [1.0, 5.0, 10.0, 15.0, 20.0]

        _, n_lo, n_hi = confidence_interval(narrow)
        _, w_lo, w_hi = confidence_interval(wide)

        narrow_width = n_hi - n_lo
        wide_width = w_hi - w_lo
        assert wide_width > narrow_width

    def test_mean_between_bounds(self):
        """The mean must lie between the lower and upper bounds."""
        values = [3.0, 7.0, 12.0, 5.0, 8.0, 6.0, 11.0, 4.0]
        mean, lower, upper = confidence_interval(values)
        assert lower <= mean <= upper

    def test_higher_confidence_wider_interval(self):
        """A 99% CI should be at least as wide as a 90% CI."""
        values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        _, lo90, hi90 = confidence_interval(values, confidence=0.90)
        _, lo99, hi99 = confidence_interval(values, confidence=0.99)
        assert (hi99 - lo99) >= (hi90 - lo90)

    def test_two_values(self):
        """Two observations should still return a valid interval."""
        mean, lower, upper = confidence_interval([1.0, 3.0])
        assert mean == 2.0
        assert lower <= mean <= upper


# ---------------------------------------------------------------------------
# 2. detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    """Tests for detect_outliers()."""

    def test_no_outliers(self):
        """A compact dataset [1..5] should have no outliers."""
        result = detect_outliers([1, 2, 3, 4, 5])
        assert result["n_outliers"] == 0
        assert len(result["outliers"]) == 0
        assert sorted(result["clean"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_clear_outlier(self):
        """100 in [1,2,3,4,5,100] should be detected as an outlier."""
        result = detect_outliers([1, 2, 3, 4, 5, 100])
        assert 100.0 in result["outliers"]
        assert result["n_outliers"] >= 1
        assert 100.0 not in result["clean"]

    def test_empty_list(self):
        """Empty input returns empty clean/outliers lists."""
        result = detect_outliers([])
        assert result["clean"] == []
        assert result["outliers"] == []
        assert result["n_outliers"] == 0

    def test_all_same_values(self):
        """When all values are identical there are no outliers."""
        result = detect_outliers([7, 7, 7, 7, 7])
        assert result["n_outliers"] == 0
        assert result["clean"] == [7.0, 7.0, 7.0, 7.0, 7.0]

    def test_fences_present(self):
        """The result should always contain fence values."""
        result = detect_outliers([1, 2, 3, 4, 5])
        assert "lower_fence" in result
        assert "upper_fence" in result
        assert result["lower_fence"] <= result["upper_fence"]

    def test_clean_plus_outliers_equals_original(self):
        """All original values should appear in either clean or outliers."""
        values = [1, 2, 3, 4, 5, 100]
        result = detect_outliers(values)
        assert len(result["clean"]) + len(result["outliers"]) == len(values)

    def test_unsupported_method_raises(self):
        """An unsupported method string should raise ValueError."""
        try:
            detect_outliers([1, 2, 3], method="zscore")
            assert False, "Expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 3. is_statistically_significant
# ---------------------------------------------------------------------------


class TestIsStatisticallySignificant:
    """Tests for is_statistically_significant()."""

    def test_same_distribution_not_significant(self):
        """Two samples from the same population should not be significant."""
        a = [10, 10, 10, 10, 10]
        b = [10, 10, 10, 10, 10]
        result = is_statistically_significant(a, b)
        assert result["significant"] is False
        assert result["p_value"] > 0.05

    def test_very_different_distributions_significant(self):
        """Completely separated groups should be highly significant."""
        a = [1, 2, 1, 2, 1]
        b = [100, 101, 100, 101, 100]
        result = is_statistically_significant(a, b)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_effect_size_identical_groups(self):
        """Effect size for identical groups should be approximately zero."""
        a = [5, 5, 5, 5, 5]
        b = [5, 5, 5, 5, 5]
        result = is_statistically_significant(a, b)
        assert result["effect_size"] < 0.01

    def test_effect_size_very_different_groups(self):
        """Effect size for very different groups should be large (>0.8)."""
        a = [1, 2, 1, 2, 1]
        b = [100, 101, 100, 101, 100]
        result = is_statistically_significant(a, b)
        assert result["effect_size"] > 0.8

    def test_returns_all_expected_keys(self):
        """The result should contain all documented keys."""
        result = is_statistically_significant([1, 2, 3], [4, 5, 6])
        expected_keys = {"significant", "p_value", "effect_size", "t_statistic", "interpretation"}
        assert expected_keys == set(result.keys())

    def test_interpretation_is_string(self):
        """The interpretation value should be a non-empty string."""
        result = is_statistically_significant([1, 2, 3], [4, 5, 6])
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_small_sample_fallback(self):
        """Groups with fewer than 2 values should return a safe default."""
        result = is_statistically_significant([1], [2])
        assert result["significant"] is False
        assert result["p_value"] == 1.0

    def test_large_effect_interpretation(self):
        """Very different groups should get a 'large effect' interpretation."""
        a = [1, 2, 1, 2, 1]
        b = [100, 101, 100, 101, 100]
        result = is_statistically_significant(a, b)
        assert result["interpretation"] == "large effect"

    def test_no_diff_interpretation(self):
        """Identical groups should get 'no significant difference'."""
        a = [5, 5, 5, 5, 5]
        b = [5, 5, 5, 5, 5]
        result = is_statistically_significant(a, b)
        assert result["interpretation"] == "no significant difference"


# ---------------------------------------------------------------------------
# 4. compute_robust_stats
# ---------------------------------------------------------------------------


class TestComputeRobustStats:
    """Tests for compute_robust_stats()."""

    def test_returns_all_expected_keys(self):
        """Result should contain every documented key."""
        result = compute_robust_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_keys = {
            "mean", "median", "std", "cv",
            "ci_lower", "ci_upper",
            "p50", "p90", "p95", "p99",
            "iqr", "n", "min", "max",
            "outlier_detection", "warnings",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_small_sample_warning(self):
        """Fewer than 5 observations should produce a warning."""
        result = compute_robust_stats([1, 2, 3])
        warnings = result["warnings"]
        assert any("sample size" in w.lower() for w in warnings)

    def test_high_variance_warning(self):
        """High CV (>0.5) should produce a variance warning."""
        # Values with very high spread relative to mean.
        result = compute_robust_stats([1, 1, 1, 1, 1, 100])
        warnings = result["warnings"]
        assert any("variance" in w.lower() or "stable" in w.lower() for w in warnings)

    def test_stable_data_no_warnings(self):
        """Stable data with n>=5 should produce no warnings."""
        result = compute_robust_stats([10.0, 10.1, 9.9, 10.0, 10.05])
        assert result["warnings"] == []

    def test_empty_input(self):
        """Empty input should return zeroed-out stats."""
        result = compute_robust_stats([])
        assert result["mean"] == 0.0
        assert result["n"] == 0
        assert result["warnings"] == []

    def test_basic_statistics_values(self):
        """Verify basic stat correctness on a known dataset."""
        result = compute_robust_stats([2, 4, 6, 8, 10])
        assert result["mean"] == 6.0
        assert result["median"] == 6.0
        assert result["min"] == 2.0
        assert result["max"] == 10.0
        assert result["n"] == 5

    def test_percentiles_ordered(self):
        """p50 <= p90 <= p95 <= p99."""
        result = compute_robust_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert result["p50"] <= result["p90"]
        assert result["p90"] <= result["p95"]
        assert result["p95"] <= result["p99"]

    def test_ci_contains_mean(self):
        """The CI should contain the sample mean."""
        result = compute_robust_stats([5, 6, 7, 8, 9, 10, 11])
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_cv_is_nonnegative(self):
        """Coefficient of variation should never be negative."""
        result = compute_robust_stats([3, 5, 7, 9])
        assert result["cv"] >= 0.0

    def test_single_value(self):
        """A single value should return that value for mean/median, std=0."""
        result = compute_robust_stats([42])
        assert result["mean"] == 42.0
        assert result["median"] == 42.0
        assert result["std"] == 0.0


# ---------------------------------------------------------------------------
# 5. steady_state_check
# ---------------------------------------------------------------------------


class TestSteadyStateCheck:
    """Tests for steady_state_check()."""

    def test_stable_values(self):
        """A constant series should be classified as stable."""
        result = steady_state_check([5, 5, 5, 5, 5])
        assert result["stable"] is True
        assert result["cv"] < 0.1

    def test_diverging_values(self):
        """An exponentially growing series should not be stable."""
        result = steady_state_check([1, 10, 100, 1000, 10000])
        assert result["stable"] is False

    def test_too_few_values(self):
        """A single observation cannot be stable."""
        result = steady_state_check([5])
        assert result["stable"] is False

    def test_empty_series(self):
        """Empty input should return stable=False."""
        result = steady_state_check([])
        assert result["stable"] is False
        assert result["n_values"] == 0

    def test_returns_expected_keys(self):
        """Result should always contain stable, cv, and n_values."""
        result = steady_state_check([1, 2, 3, 4, 5])
        assert "stable" in result
        assert "cv" in result
        assert "n_values" in result

    def test_custom_window(self):
        """Only the tail window should be evaluated, ignoring noisy prefix."""
        # Noisy prefix + stable tail.
        series = [100, 200, 300, 5, 5, 5, 5, 5]
        result = steady_state_check(series, window=5)
        assert result["stable"] is True
        assert result["n_values"] == 5

    def test_window_larger_than_series(self):
        """When the window is larger than the series, all values are used."""
        result = steady_state_check([5, 5, 5], window=10)
        assert result["n_values"] == 3
        assert result["stable"] is True

    def test_cv_is_inf_for_single_value(self):
        """A single value should yield cv=inf since std is undefined."""
        result = steady_state_check([5])
        assert math.isinf(result["cv"])


# ---------------------------------------------------------------------------
# 6. aggregate_iterations
# ---------------------------------------------------------------------------


class TestAggregateIterations:
    """Tests for aggregate_iterations()."""

    def _make_result(self, ctx=512, users=1, prompt_type="chat", **metrics):
        """Helper to build a mock result dict."""
        base = {
            "context_length": ctx,
            "concurrent_users": users,
            "prompt_type": prompt_type,
            "tokens_per_second": 100.0,
            "avg_latency": 2.0,
        }
        base.update(metrics)
        return base

    def test_two_iterations_produces_ci_fields(self):
        """Two iterations for the same config should produce CI fields."""
        iter1 = [self._make_result(tokens_per_second=100)]
        iter2 = [self._make_result(tokens_per_second=110)]
        aggregated = aggregate_iterations([iter1, iter2])

        assert len(aggregated) == 1
        entry = aggregated[0]

        # Should have CI fields for tokens_per_second.
        assert "tokens_per_second_ci_lower" in entry
        assert "tokens_per_second_ci_upper" in entry
        # The representative value (median) should be present.
        assert "tokens_per_second" in entry

    def test_single_iteration(self):
        """A single iteration should still produce valid output."""
        iter1 = [self._make_result(tokens_per_second=50)]
        aggregated = aggregate_iterations([iter1])

        assert len(aggregated) == 1
        entry = aggregated[0]
        assert entry["tokens_per_second"] == 50.0

    def test_empty_input(self):
        """No iterations should return an empty list."""
        assert aggregate_iterations([]) == []

    def test_groups_by_config(self):
        """Results with different configs should be in separate groups."""
        iter1 = [
            self._make_result(ctx=512, users=1),
            self._make_result(ctx=1024, users=4),
        ]
        iter2 = [
            self._make_result(ctx=512, users=1),
            self._make_result(ctx=1024, users=4),
        ]
        aggregated = aggregate_iterations([iter1, iter2])
        assert len(aggregated) == 2

    def test_preserves_config_keys(self):
        """Aggregated entries should carry forward the config keys."""
        iter1 = [self._make_result(ctx=256, users=8, prompt_type="code")]
        aggregated = aggregate_iterations([iter1])

        entry = aggregated[0]
        assert entry["context_length"] == 256
        assert entry["concurrent_users"] == 8
        assert entry["prompt_type"] == "code"

    def test_median_used_as_representative(self):
        """The representative metric value should be the median."""
        iter1 = [self._make_result(tokens_per_second=10)]
        iter2 = [self._make_result(tokens_per_second=20)]
        iter3 = [self._make_result(tokens_per_second=30)]
        aggregated = aggregate_iterations([iter1, iter2, iter3])

        entry = aggregated[0]
        # Median of [10, 20, 30] is 20.
        assert entry["tokens_per_second"] == 20.0

    def test_stats_dict_attached(self):
        """Each metric should have a full _stats dict with robust stats."""
        iter1 = [self._make_result(tokens_per_second=100)]
        iter2 = [self._make_result(tokens_per_second=120)]
        aggregated = aggregate_iterations([iter1, iter2])

        entry = aggregated[0]
        stats = entry.get("tokens_per_second_stats")
        assert stats is not None
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
