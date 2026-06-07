"""Tests for the PR4 statistics additions and the nan-significance fix."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm_benchmark.analysis.statistics import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    compare_groups,
    holm_adjusted_p,
    holm_bonferroni,
    is_statistically_significant,
    mann_whitney_u,
    t_ci,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ordering(self):
        rng = np.random.default_rng(0)
        data = rng.normal(10.0, 2.0, size=50).tolist()
        low, point, high = bootstrap_ci(data, seed=123)
        assert low <= point <= high
        assert all(math.isfinite(v) for v in (low, point, high))

    def test_degenerate_all_identical_no_nan(self):
        low, point, high = bootstrap_ci([5.0, 5.0, 5.0, 5.0], seed=1)
        assert (low, point, high) == (5.0, 5.0, 5.0)
        assert all(math.isfinite(v) for v in (low, point, high))

    def test_empty(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)

    def test_single_value(self):
        assert bootstrap_ci([42.0]) == (42.0, 42.0, 42.0)

    def test_percentile_method_ordering(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0.0, 1.0, size=40).tolist()
        low, point, high = bootstrap_ci(data, method="percentile", seed=7)
        assert low <= point <= high

    def test_seed_reproducible(self):
        rng = np.random.default_rng(3)
        data = rng.normal(1.0, 0.5, size=30).tolist()
        a = bootstrap_ci(data, seed=99)
        b = bootstrap_ci(data, seed=99)
        assert a == b


# ---------------------------------------------------------------------------
# t_ci
# ---------------------------------------------------------------------------


class TestTCI:
    def test_ordering(self):
        low, point, high = t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert low <= point <= high
        assert point == 3.0

    def test_degenerate(self):
        assert t_ci([7.0, 7.0, 7.0]) == (7.0, 7.0, 7.0)

    def test_empty(self):
        assert t_ci([]) == (0.0, 0.0, 0.0)

    def test_wider_for_higher_confidence(self):
        data = [2.0, 4.0, 6.0, 8.0, 10.0]
        _, _, hi95 = t_ci(data, 0.95)
        lo95 = t_ci(data, 0.95)[0]
        _, _, hi99 = t_ci(data, 0.99)
        lo99 = t_ci(data, 0.99)[0]
        assert (hi99 - lo99) >= (hi95 - lo95)


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------


class TestMannWhitney:
    def test_identical_groups_p1(self):
        res = mann_whitney_u([5, 5, 5], [5, 5, 5])
        assert res["p_value"] == 1.0

    def test_empty(self):
        res = mann_whitney_u([], [1, 2, 3])
        assert res["p_value"] == 1.0

    def test_separated_groups_significant(self):
        res = mann_whitney_u([1, 2, 3, 4], [100, 200, 300, 400])
        assert res["p_value"] < 0.05

    def test_no_nan(self):
        res = mann_whitney_u([1, 1, 1], [1, 1, 1])
        assert math.isfinite(res["p_value"])


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


class TestCliffsDelta:
    def test_identical_negligible(self):
        res = cliffs_delta([1, 2, 3], [1, 2, 3])
        assert res["delta"] == 0.0
        assert res["magnitude"] == "negligible"

    def test_full_separation_large(self):
        res = cliffs_delta([1, 2, 3], [10, 11, 12])
        assert abs(res["delta"]) == 1.0
        assert res["magnitude"] == "large"

    def test_magnitude_thresholds(self):
        # Construct overlapping groups giving a small/medium delta and check
        # the threshold buckets are coherent.
        small = cliffs_delta([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        assert small["magnitude"] in {"negligible", "small", "medium", "large"}

    def test_known_magnitude_buckets(self):
        from vllm_benchmark.analysis.statistics import _cliffs_magnitude

        assert _cliffs_magnitude(0.10) == "negligible"
        assert _cliffs_magnitude(0.20) == "small"
        assert _cliffs_magnitude(0.40) == "medium"
        assert _cliffs_magnitude(0.90) == "large"


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_identical_zero(self):
        assert cohens_d([5, 5, 5], [5, 5, 5]) == 0.0

    def test_no_nan_zero_variance(self):
        assert math.isfinite(cohens_d([3, 3, 3], [3, 3, 3]))

    def test_large_for_separated(self):
        d = cohens_d([1, 2, 1, 2], [100, 101, 100, 101])
        assert abs(d) > 0.8

    def test_sign_follows_mean_difference(self):
        assert cohens_d([1, 2, 3], [10, 11, 12]) < 0
        assert cohens_d([10, 11, 12], [1, 2, 3]) > 0


# ---------------------------------------------------------------------------
# holm_bonferroni / holm_adjusted_p
# ---------------------------------------------------------------------------


class TestHolm:
    def test_known_vector(self):
        # m=4. Sorted: 0.005, 0.01, 0.03, 0.5 vs alpha/(m-k)=0.0125,0.0167,0.025,0.05
        # 0.005<=0.0125 reject; 0.01<=0.0167 reject; 0.03>0.025 -> stop.
        pvals = [0.01, 0.03, 0.5, 0.005]
        rejected = holm_bonferroni(pvals, alpha=0.05)
        # original order: 0.01->True, 0.03->False, 0.5->False, 0.005->True
        assert rejected == [True, False, False, True]

    def test_empty(self):
        assert holm_bonferroni([]) == []
        assert holm_adjusted_p([]) == []

    def test_adjusted_monotonic_in_sorted_order(self):
        pvals = [0.04, 0.005, 0.5, 0.01]
        adj = holm_adjusted_p(pvals)
        order = sorted(range(len(pvals)), key=lambda i: pvals[i])
        sorted_adj = [adj[i] for i in order]
        assert sorted_adj == sorted(sorted_adj)
        assert all(0.0 <= p <= 1.0 for p in adj)

    def test_adjusted_clipped_to_one(self):
        adj = holm_adjusted_p([0.9, 0.95, 0.99])
        assert all(p <= 1.0 for p in adj)


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------


class TestWilson:
    def test_n_zero(self):
        assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)

    def test_textbook_value(self):
        # 8 successes out of 10, 95% Wilson interval ~ (0.490, 0.943).
        low, point, high = wilson_interval(8, 10, confidence=0.95)
        assert abs(point - 0.8) < 1e-9
        assert abs(low - 0.490) < 0.01
        assert abs(high - 0.943) < 0.01

    def test_bounds_in_unit_interval(self):
        low, point, high = wilson_interval(3, 7)
        assert 0.0 <= low <= point <= high <= 1.0

    def test_perfect_success(self):
        low, point, high = wilson_interval(10, 10)
        assert point == 1.0
        assert high <= 1.0
        assert low < 1.0


# ---------------------------------------------------------------------------
# compare_groups
# ---------------------------------------------------------------------------


class TestCompareGroups:
    def test_real_difference(self):
        a = [1, 2, 1, 2, 1, 2, 1, 2]
        b = [100, 101, 100, 101, 100, 101, 100, 101]
        res = compare_groups(a, b)
        assert res["significant"] is True
        assert res["real_difference"] is True
        assert abs(res["cohens_d"]) >= 0.33

    def test_noise_no_real_difference(self):
        rng = np.random.default_rng(11)
        a = rng.normal(100.0, 5.0, size=30).tolist()
        b = rng.normal(100.2, 5.0, size=30).tolist()
        res = compare_groups(a, b)
        assert res["real_difference"] is False

    def test_keys_present(self):
        res = compare_groups([1, 2, 3], [4, 5, 6])
        for key in (
            "welch_p", "mann_whitney_p", "cohens_d", "cliffs_delta",
            "cliffs_magnitude", "significant", "real_difference",
        ):
            assert key in res


# ---------------------------------------------------------------------------
# is_statistically_significant nan fix (the previously-failing test)
# ---------------------------------------------------------------------------


class TestNanFix:
    def test_identical_inputs_not_significant_no_nan(self):
        res = is_statistically_significant([10, 10, 10, 10, 10], [10, 10, 10, 10, 10])
        assert res["significant"] is False
        assert res["p_value"] > 0.05
        assert math.isfinite(res["p_value"])


# ---------------------------------------------------------------------------
# Property test: CI bound ordering
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=40,
    )
)
def test_property_ci_bounds_ordered(data):
    """For any finite sample, low <= point <= high for both CI helpers."""
    b_low, b_point, b_high = bootstrap_ci(data, seed=0)
    assert b_low <= b_point <= b_high
    assert math.isfinite(b_low) and math.isfinite(b_high)

    t_low, t_point, t_high = t_ci(data)
    assert t_low <= t_point <= t_high
    assert math.isfinite(t_low) and math.isfinite(t_high)
