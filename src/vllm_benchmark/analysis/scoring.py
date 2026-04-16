"""Composite benchmark scoring for vLLM deployments.

Produces a single 0-10000 score (like Geekbench for vLLM) with breakdowns
across throughput, latency, efficiency, energy, and consistency dimensions.

Author: amit
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreBreakdown:
    """Complete score breakdown with per-dimension scores and overall grade."""

    overall: int  # 0-10000
    throughput: int  # 0-10000
    latency: int  # 0-10000
    efficiency: int  # 0-10000
    energy: int  # 0-10000
    consistency: int  # 0-10000
    grade: str  # "S", "A", "B", "C", "D", "F"


class VLLMScore:
    """Compute composite benchmark score.

    Weights each dimension and produces a single overall score plus a
    letter grade, giving users an at-a-glance assessment of their
    vLLM deployment performance.
    """

    WEIGHTS = {
        "throughput": 0.30,
        "latency": 0.25,
        "efficiency": 0.20,
        "energy": 0.15,
        "consistency": 0.10,
    }

    # Reference values for score calculation
    _THROUGHPUT_REF = 2000  # tok/s for 10000 points
    _LATENCY_CEIL = 10.0  # seconds; at this value, score = 0
    _EFFICIENCY_REF = 200  # tok/s/user for 10000 points
    _ENERGY_REF = 10.0  # tok/watt for 10000 points
    _CV_CEIL = 1.0  # coefficient of variation; at this value, score = 0

    def calculate(self, results: list[dict], gpu_name: str = None) -> ScoreBreakdown:
        """Calculate composite score from benchmark results.

        Args:
            results: List of result dicts from the benchmark run. Expected
                keys per result: tokens_per_second, avg_latency, std_latency,
                throughput_per_user, tokens_per_watt (optional).
            gpu_name: Optional GPU name (unused for scoring, reserved for
                future GPU-relative scoring).

        Returns:
            ScoreBreakdown with all dimension scores and final grade.
        """
        throughput_score = self._score_throughput(results)
        latency_score = self._score_latency(results)
        efficiency_score = self._score_efficiency(results)
        energy_score = self._score_energy(results)
        consistency_score = self._score_consistency(results)

        overall = round(
            throughput_score * self.WEIGHTS["throughput"]
            + latency_score * self.WEIGHTS["latency"]
            + efficiency_score * self.WEIGHTS["efficiency"]
            + energy_score * self.WEIGHTS["energy"]
            + consistency_score * self.WEIGHTS["consistency"]
        )
        overall = max(0, min(10000, overall))

        grade = self._compute_grade(overall)

        return ScoreBreakdown(
            overall=overall,
            throughput=throughput_score,
            latency=latency_score,
            efficiency=efficiency_score,
            energy=energy_score,
            consistency=consistency_score,
            grade=grade,
        )

    def format_score_display(self, score: ScoreBreakdown) -> str:
        """Return a Rich-formatted string showing the score with a visual bar.

        The output is designed to be printed directly with rich.console.Console
        or used inside a rich.panel.Panel.
        """
        lines: list[str] = []

        # Header with overall score and grade
        grade_color = self._grade_color(score.grade)
        lines.append("")
        lines.append(
            f"  [bold {grade_color}]vLLM Benchmark Score: "
            f"{score.overall:,} / 10,000  (Grade: {score.grade})[/]"
        )
        lines.append("")

        # Dimension bars
        dimensions = [
            ("Throughput ", score.throughput, self.WEIGHTS["throughput"]),
            ("Latency    ", score.latency, self.WEIGHTS["latency"]),
            ("Efficiency ", score.efficiency, self.WEIGHTS["efficiency"]),
            ("Energy     ", score.energy, self.WEIGHTS["energy"]),
            ("Consistency", score.consistency, self.WEIGHTS["consistency"]),
        ]

        bar_width = 30
        for label, value, weight in dimensions:
            filled = round((value / 10000) * bar_width)
            empty = bar_width - filled
            pct_label = f"{value:>5,}"
            weight_label = f"{weight * 100:.0f}%"
            color = self._bar_color(value)
            bar = f"[{color}]{'█' * filled}[/]{'░' * empty}"
            lines.append(
                f"  {label} ({weight_label:>3}) {bar} {pct_label}"
            )

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dimension scoring
    # ------------------------------------------------------------------

    def _score_throughput(self, results: list[dict]) -> int:
        """Score based on peak tokens_per_second across all results."""
        tps_values = [
            r["tokens_per_second"]
            for r in results
            if r.get("tokens_per_second") is not None
        ]
        if not tps_values:
            return 0
        peak_tps = max(tps_values)
        return self._clamp((peak_tps / self._THROUGHPUT_REF) * 10000)

    def _score_latency(self, results: list[dict]) -> int:
        """Score based on best (lowest) avg_latency. Lower is better."""
        lat_values = [
            r["avg_latency"]
            for r in results
            if r.get("avg_latency") is not None and r["avg_latency"] > 0
        ]
        if not lat_values:
            return 0
        best_latency = min(lat_values)
        return self._clamp((1 - best_latency / self._LATENCY_CEIL) * 10000)

    def _score_efficiency(self, results: list[dict]) -> int:
        """Score based on best throughput_per_user."""
        tpu_values = [
            r["throughput_per_user"]
            for r in results
            if r.get("throughput_per_user") is not None
        ]
        if not tpu_values:
            return 0
        best_tpu = max(tpu_values)
        return self._clamp((best_tpu / self._EFFICIENCY_REF) * 10000)

    def _score_energy(self, results: list[dict]) -> int:
        """Score based on best tokens_per_watt. Defaults to 5000 if unavailable."""
        tpw_values = [
            r["tokens_per_watt"]
            for r in results
            if r.get("tokens_per_watt") is not None
        ]
        if not tpw_values:
            return 5000
        best_tpw = max(tpw_values)
        return self._clamp((best_tpw / self._ENERGY_REF) * 10000)

    def _score_consistency(self, results: list[dict]) -> int:
        """Score based on latency consistency (coefficient of variation).

        Uses the best (lowest) CV across all results. A CV of 0 means
        perfectly consistent latency (10000 points), CV >= 1.0 scores 0.
        """
        cv_values: list[float] = []
        for r in results:
            avg = r.get("avg_latency")
            std = r.get("std_latency")
            if avg is not None and std is not None and avg > 0:
                cv_values.append(std / avg)

        if not cv_values:
            return 5000  # neutral when no data
        best_cv = min(cv_values)
        return self._clamp((1 - best_cv / self._CV_CEIL) * 10000)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> int:
        """Clamp a raw score to [0, 10000] and round to int."""
        return max(0, min(10000, round(value)))

    @staticmethod
    def _compute_grade(overall: int) -> str:
        """Map overall score to a letter grade."""
        if overall >= 9000:
            return "S"
        if overall >= 7500:
            return "A"
        if overall >= 6000:
            return "B"
        if overall >= 4500:
            return "C"
        if overall >= 3000:
            return "D"
        return "F"

    @staticmethod
    def _grade_color(grade: str) -> str:
        """Rich color for a letter grade."""
        return {
            "S": "bright_magenta",
            "A": "green",
            "B": "bright_green",
            "C": "yellow",
            "D": "bright_red",
            "F": "red",
        }.get(grade, "white")

    @staticmethod
    def _bar_color(value: int) -> str:
        """Rich color for a score bar segment."""
        if value >= 8000:
            return "green"
        if value >= 6000:
            return "bright_green"
        if value >= 4000:
            return "yellow"
        if value >= 2000:
            return "bright_red"
        return "red"
