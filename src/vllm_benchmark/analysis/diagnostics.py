"""Auto-diagnostics engine for vLLM benchmark results.

Analyzes benchmark results and produces plain-English recommendations,
as if an expert vLLM operator were reviewing your deployment.

Author: amit
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class Diagnostic:
    """A single diagnostic finding from benchmark analysis."""

    severity: str  # "critical", "warning", "info", "success"
    title: str
    message: str
    metric: str  # which metric triggered this


class DiagnosticEngine:
    """Analyzes benchmark results and produces actionable diagnostics.

    Each result dict is expected to contain keys such as:
        - context_length (int)
        - concurrent_users (int)
        - tokens_per_second (float)
        - avg_latency (float, seconds)
        - max_latency (float, seconds)
        - std_latency (float, seconds)
        - ttft_estimate (float, seconds)
        - throughput_per_user (float)
        - tokens_per_watt (float, optional)
        - failed (int)
        - prompt_type (str, optional)
        - gpu_stats (dict, optional) with keys:
            - avg_gpu_util (float, 0-100)
            - max_temperature (float, Celsius)
            - max_mem_used (float, bytes or GB)
            - mem_total (float, bytes or GB)
            - cache_hit_rate (float, 0-100, optional)
    """

    def analyze(self, results: list[dict], server_info: dict = None) -> list[Diagnostic]:
        """Run all diagnostic rules against results. Returns list of Diagnostic objects."""
        if not results:
            return [
                Diagnostic(
                    severity="info",
                    title="No results to analyze",
                    message="No benchmark results were provided for analysis.",
                    metric="results",
                )
            ]

        diagnostics: list[Diagnostic] = []

        diagnostics.extend(self._check_request_failures(results))
        diagnostics.extend(self._check_high_latency_variance(results))
        diagnostics.extend(self._check_low_gpu_utilization(results))
        diagnostics.extend(self._check_gpu_saturated_low_throughput(results))
        diagnostics.extend(self._check_cache_inefficiency(results))
        diagnostics.extend(self._check_slow_ttft(results))
        diagnostics.extend(self._check_poor_batch_scaling(results))
        diagnostics.extend(self._check_high_temperature(results))
        diagnostics.extend(self._check_memory_near_limit(results))
        diagnostics.extend(self._check_excellent_throughput(results))
        diagnostics.extend(self._check_energy_efficiency(results))

        has_warning_or_critical = any(
            d.severity in ("critical", "warning") for d in diagnostics
        )
        if not has_warning_or_critical:
            diagnostics.append(
                Diagnostic(
                    severity="success",
                    title="All healthy",
                    message=(
                        "All metrics look healthy. Your vLLM deployment is "
                        "well-configured for the tested workloads."
                    ),
                    metric="overall",
                )
            )

        return diagnostics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ctx_label(ctx: int) -> str:
        """Format context length as a readable label like '32K'."""
        if ctx >= 1_000_000:
            return f"{ctx / 1_000_000:.0f}M"
        return f"{ctx / 1000:.0f}K"

    @staticmethod
    def _gpu(result: dict) -> dict:
        """Safely extract GPU stats — supports both flat and nested formats."""
        # v3 flattens GPU stats into the result dict
        if "avg_gpu_util" in result:
            return result
        return result.get("gpu_stats") or {}

    # ------------------------------------------------------------------
    # Diagnostic rules
    # ------------------------------------------------------------------

    def _check_request_failures(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            failed = r.get("failed", 0)
            if failed and failed > 0:
                diagnostics.append(
                    Diagnostic(
                        severity="critical",
                        title="Request failures detected",
                        message=(
                            f"{failed} requests failed during testing. Common causes: "
                            "context length exceeding max_model_len, GPU OOM, or "
                            "request timeout. Check vLLM server logs."
                        ),
                        metric="failed",
                    )
                )
        return diagnostics

    def _check_high_latency_variance(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            avg_lat = r.get("avg_latency")
            max_lat = r.get("max_latency")
            if not avg_lat or not max_lat or avg_lat <= 0:
                continue
            ratio = max_lat / avg_lat
            if ratio > 3:
                ctx = self._ctx_label(r.get("context_length", 0))
                users = r.get("concurrent_users", "?")
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="Extreme tail latency detected",
                        message=(
                            f"Your worst-case latency is {ratio:.1f}x your average "
                            f"at {ctx} context with {users} users. This usually means "
                            "request queuing under load. Try reducing max_num_seqs or "
                            "enabling chunked prefill."
                        ),
                        metric="max_latency",
                    )
                )
        return diagnostics

    def _check_low_gpu_utilization(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            gpu = self._gpu(r)
            util = gpu.get("avg_gpu_util")
            if util is None:
                continue
            if util < 60:
                ctx = self._ctx_label(r.get("context_length", 0))
                users = r.get("concurrent_users", "?")
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="GPU underutilized",
                        message=(
                            f"GPU underutilized at {util:.0f}% with {users} concurrent "
                            f"users at {ctx} context. Your server may be CPU/network "
                            "bottlenecked. Consider increasing concurrent users or "
                            "batch size."
                        ),
                        metric="avg_gpu_util",
                    )
                )
        return diagnostics

    def _check_gpu_saturated_low_throughput(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        tps_values = [
            r["tokens_per_second"]
            for r in results
            if r.get("tokens_per_second") is not None
        ]
        if not tps_values:
            return diagnostics
        median_tps = statistics.median(tps_values)

        for r in results:
            gpu = self._gpu(r)
            util = gpu.get("avg_gpu_util")
            tps = r.get("tokens_per_second")
            if util is None or tps is None:
                continue
            if util > 90 and tps < median_tps:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="GPU saturated with low throughput",
                        message=(
                            f"GPU is saturated ({util:.0f}%) but throughput is below "
                            "average. This may indicate memory bandwidth saturation. "
                            "Consider a smaller model or different quantization."
                        ),
                        metric="tokens_per_second",
                    )
                )
        return diagnostics

    def _check_cache_inefficiency(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            gpu = self._gpu(r)
            rate = gpu.get("cache_hit_rate")
            if rate is None:
                continue
            prompt_type = r.get("prompt_type", "")
            is_deterministic = prompt_type in ("deterministic", "fixed", "cached")
            if rate < 20 and is_deterministic:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="Low prefix cache hit rate",
                        message=(
                            f"Prefix cache hit rate is only {rate:.0f}% despite using "
                            "deterministic prompts. Verify --enable-prefix-caching is "
                            "set on your vLLM server."
                        ),
                        metric="cache_hit_rate",
                    )
                )
        return diagnostics

    def _check_slow_ttft(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            ttft = r.get("ttft_estimate")
            if ttft is None:
                continue
            ttft_ms = ttft * 1000
            if ttft_ms > 2000:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="Slow Time to First Token",
                        message=(
                            f"Time to First Token is {ttft_ms:.0f}ms, which will feel "
                            "sluggish to users. This may indicate KV cache pressure or "
                            "insufficient GPU memory bandwidth."
                        ),
                        metric="ttft_estimate",
                    )
                )
        return diagnostics

    def _check_poor_batch_scaling(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        # Group results by context_length (and prompt_type if present)
        groups: dict[tuple, list[dict]] = {}
        for r in results:
            ctx = r.get("context_length", 0)
            pt = r.get("prompt_type", "")
            key = (ctx, pt)
            groups.setdefault(key, []).append(r)

        for key, group in groups.items():
            # Find single-user throughput
            single_user = [
                r for r in group if r.get("concurrent_users") == 1
            ]
            if not single_user:
                continue
            base_tps = single_user[0].get("tokens_per_second")
            if not base_tps or base_tps <= 0:
                continue

            for r in group:
                users = r.get("concurrent_users", 1)
                if users <= 1:
                    continue
                tps = r.get("tokens_per_second")
                if tps is None:
                    continue
                ideal = base_tps * users
                efficiency = (tps / ideal) * 100
                if efficiency < 50:
                    ctx_label = self._ctx_label(key[0])
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            title="Poor batch scaling efficiency",
                            message=(
                                f"Batch scaling efficiency is poor at {efficiency:.0f}% "
                                f"({users} users, {ctx_label} context). Adding more "
                                "concurrent users isn't proportionally increasing "
                                "throughput. You've likely hit memory bandwidth limits."
                            ),
                            metric="throughput_scaling",
                        )
                    )
        return diagnostics

    def _check_high_temperature(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            gpu = self._gpu(r)
            temp = gpu.get("max_temperature")
            if temp is None:
                continue
            if temp > 85:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="High GPU temperature",
                        message=(
                            f"GPU temperature peaked at {temp:.0f}\u00b0C. This is close "
                            "to thermal throttling thresholds and may cause inconsistent "
                            "performance."
                        ),
                        metric="max_temperature",
                    )
                )
        return diagnostics

    def _check_memory_near_limit(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            gpu = self._gpu(r)
            max_mem = gpu.get("max_mem_used")
            mem_total = gpu.get("mem_total")
            if max_mem is None or mem_total is None or mem_total <= 0:
                continue
            pct = max_mem / mem_total
            if pct > 0.95:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        title="GPU memory near limit",
                        message=(
                            f"GPU memory usage peaked at {pct * 100:.0f}%, very close "
                            "to the limit. Risk of OOM under production load. Consider "
                            "reducing max_num_seqs or using more aggressive quantization."
                        ),
                        metric="max_mem_used",
                    )
                )
        return diagnostics

    def _check_excellent_throughput(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            tps = r.get("tokens_per_second")
            if tps is not None and tps > 500:
                ctx = self._ctx_label(r.get("context_length", 0))
                users = r.get("concurrent_users", "?")
                diagnostics.append(
                    Diagnostic(
                        severity="success",
                        title="Excellent throughput",
                        message=(
                            f"Excellent throughput of {tps:.0f} tokens/sec achieved "
                            f"at {ctx} context with {users} users. Your setup is "
                            "well-optimized for this workload."
                        ),
                        metric="tokens_per_second",
                    )
                )
        return diagnostics

    def _check_energy_efficiency(self, results: list[dict]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for r in results:
            tpw = r.get("tokens_per_watt")
            if tpw is not None and tpw > 5:
                diagnostics.append(
                    Diagnostic(
                        severity="success",
                        title="Good energy efficiency",
                        message=(
                            f"Strong energy efficiency at {tpw:.1f} tokens/watt. "
                            "Your setup is power-efficient."
                        ),
                        metric="tokens_per_watt",
                    )
                )
        return diagnostics
