"""Performance regression detection for vLLM benchmark results.

Compares current benchmark results against a saved baseline to detect
throughput drops, latency increases, and other performance regressions
across matching test configurations.

Author: amit
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RegressionResult:
    """A single metric comparison between current and baseline runs."""

    metric: str
    context_length: int
    concurrent_users: int
    previous_value: float
    current_value: float
    change_pct: float
    is_regression: bool  # True if performance got worse
    severity: str  # "major" (>15%), "minor" (5-15%), "none" (<5%)


# Metrics where higher values are better (regression = current < previous)
_HIGHER_IS_BETTER = {"tokens_per_second", "throughput_per_user"}

# Metrics where lower values are better (regression = current > previous)
_LOWER_IS_BETTER = {"avg_latency", "ttft_estimate"}

_ALL_METRICS = _HIGHER_IS_BETTER | _LOWER_IS_BETTER


class RegressionDetector:
    """Compare benchmark results against a baseline.

    Loads a previously saved JSON results file and compares each matching
    test configuration (context_length + concurrent_users + prompt_type)
    to flag performance regressions above a configurable threshold.
    """

    def __init__(self, threshold_pct: float = 5.0):
        self.threshold_pct = threshold_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self, current_results: list[dict], baseline_path: str
    ) -> list[RegressionResult]:
        """Load baseline from JSON file, compare with current results.

        Args:
            current_results: List of result dicts from the current run.
            baseline_path: Path to a JSON file with structure
                ``{"metadata": {...}, "results": [...]}``.

        Returns:
            List of RegressionResult for every metric in every matching
            test configuration.

        Raises:
            FileNotFoundError: If baseline_path does not exist.
            ValueError: If baseline JSON is malformed or contains no results.
        """
        baseline_results = self._load_baseline(baseline_path)
        baseline_index = self._index_results(baseline_results)
        current_index = self._index_results(current_results)

        regressions: list[RegressionResult] = []

        for key, current_row in current_index.items():
            baseline_row = baseline_index.get(key)
            if baseline_row is None:
                continue  # no matching baseline entry

            ctx = key[0]
            users = key[1]

            for metric in _ALL_METRICS:
                prev_val = baseline_row.get(metric)
                curr_val = current_row.get(metric)
                if prev_val is None or curr_val is None:
                    continue
                if prev_val == 0:
                    # Avoid division by zero; skip metric if baseline is 0
                    continue

                change_pct = ((curr_val - prev_val) / abs(prev_val)) * 100

                is_regression = self._is_regression(metric, change_pct)
                severity = self._classify_severity(metric, change_pct)

                regressions.append(
                    RegressionResult(
                        metric=metric,
                        context_length=ctx,
                        concurrent_users=users,
                        previous_value=prev_val,
                        current_value=curr_val,
                        change_pct=change_pct,
                        is_regression=is_regression,
                        severity=severity,
                    )
                )

        return regressions

    def format_report(self, regressions: list[RegressionResult]) -> str:
        """Format regression report as a Rich-compatible string.

        Produces a table with columns:
            Metric | Config | Previous | Current | Change | Status
        """
        if not regressions:
            return "[green]No regressions detected. All metrics are stable.[/]"

        lines: list[str] = []
        lines.append("")
        lines.append("[bold]Performance Regression Report[/]")
        lines.append("")

        # Column headers
        header = (
            f"  {'Metric':<22} {'Config':<18} {'Previous':>12} "
            f"{'Current':>12} {'Change':>10} {'Status':<10}"
        )
        lines.append(f"[bold]{header}[/]")
        lines.append(f"  {'─' * 88}")

        for r in sorted(regressions, key=lambda x: (x.severity != "major", x.severity != "minor", x.metric)):
            config_label = f"{_ctx_label(r.context_length)} / {r.concurrent_users}u"
            change_str = f"{r.change_pct:+.1f}%"
            status, color = self._status_display(r)

            prev_str = _format_value(r.metric, r.previous_value)
            curr_str = _format_value(r.metric, r.current_value)

            line = (
                f"  {r.metric:<22} {config_label:<18} {prev_str:>12} "
                f"{curr_str:>12} [{color}]{change_str:>10}[/] {status:<10}"
            )
            lines.append(line)

        lines.append("")

        # Summary
        major_count = sum(1 for r in regressions if r.severity == "major")
        minor_count = sum(1 for r in regressions if r.severity == "minor")
        improved = sum(
            1 for r in regressions
            if not r.is_regression and r.severity != "none"
        )

        summary_parts: list[str] = []
        if major_count:
            summary_parts.append(f"[bold red]{major_count} major regression(s)[/]")
        if minor_count:
            summary_parts.append(f"[yellow]{minor_count} minor regression(s)[/]")
        if improved:
            summary_parts.append(f"[green]{improved} improvement(s)[/]")
        if not summary_parts:
            summary_parts.append("[green]All changes within tolerance[/]")

        lines.append("  Summary: " + ", ".join(summary_parts))
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_baseline(baseline_path: str) -> list[dict]:
        """Read and validate baseline JSON file."""
        path = Path(baseline_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Baseline file not found: {baseline_path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Accept a bare list of results
            return data

        if isinstance(data, dict):
            results = data.get("results")
            if results is None:
                raise ValueError(
                    f"Baseline JSON has no 'results' key: {baseline_path}"
                )
            if not isinstance(results, list):
                raise ValueError(
                    f"'results' in baseline JSON is not a list: {baseline_path}"
                )
            return results

        raise ValueError(
            f"Unexpected baseline JSON format in: {baseline_path}"
        )

    @staticmethod
    def _index_results(results: list[dict]) -> dict[tuple, dict]:
        """Index results by (context_length, concurrent_users, prompt_type).

        If prompt_type is absent, it defaults to an empty string so that
        results without prompt_type still match each other.
        """
        index: dict[tuple, dict] = {}
        for r in results:
            ctx = r.get("context_length", 0)
            users = r.get("concurrent_users", 1)
            pt = r.get("prompt_type", "")
            key = (ctx, users, pt)
            index[key] = r
        return index

    def _is_regression(self, metric: str, change_pct: float) -> bool:
        """Determine if the change direction constitutes a regression."""
        abs_change = abs(change_pct)
        if abs_change < self.threshold_pct:
            return False

        if metric in _HIGHER_IS_BETTER:
            # Regression when current is lower (negative change)
            return change_pct < -self.threshold_pct
        if metric in _LOWER_IS_BETTER:
            # Regression when current is higher (positive change)
            return change_pct > self.threshold_pct
        return False

    def _classify_severity(self, metric: str, change_pct: float) -> str:
        """Classify the severity of a change based on its magnitude and direction."""
        # Determine if the direction is a regression
        is_bad_direction = False
        if metric in _HIGHER_IS_BETTER and change_pct < 0:
            is_bad_direction = True
        elif metric in _LOWER_IS_BETTER and change_pct > 0:
            is_bad_direction = True

        abs_change = abs(change_pct)

        if not is_bad_direction:
            # Improvements also get severity labels (useful for the report)
            if abs_change > 15:
                return "major"
            if abs_change > self.threshold_pct:
                return "minor"
            return "none"

        if abs_change > 15:
            return "major"
        if abs_change > self.threshold_pct:
            return "minor"
        return "none"

    @staticmethod
    def _status_display(r: RegressionResult) -> tuple[str, str]:
        """Return (status_text, rich_color) for a regression result."""
        if r.severity == "none":
            return "✓ Stable", "dim"

        if r.is_regression:
            if r.severity == "major":
                return "✗ REGRESSION", "bold red"
            return "✗ Regressed", "yellow"

        # Improvement
        if r.severity == "major":
            return "★ Improved!", "bold green"
        return "↑ Improved", "green"


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _ctx_label(ctx: int) -> str:
    """Format context length as a readable label like '32K'."""
    if ctx >= 1_000_000:
        return f"{ctx / 1_000_000:.0f}M"
    return f"{ctx / 1000:.0f}K"


def _format_value(metric: str, value: float) -> str:
    """Format a metric value with appropriate units."""
    if metric in ("avg_latency", "ttft_estimate"):
        if value < 1.0:
            return f"{value * 1000:.0f}ms"
        return f"{value:.2f}s"
    if metric == "tokens_per_second":
        return f"{value:.0f} tok/s"
    if metric == "throughput_per_user":
        return f"{value:.1f} tok/s/u"
    return f"{value:.2f}"
