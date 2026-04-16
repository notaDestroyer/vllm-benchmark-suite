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

        # Append vLLM config-specific recommendations when server_info is available
        if server_info:
            diagnostics.extend(self.advise_config(results, server_info))

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

    def advise_config(self, results: list[dict], server_info: dict) -> list[Diagnostic]:
        """Produce vLLM config-specific recommendations.

        Examines server configuration alongside benchmark results to
        surface actionable tuning advice.

        Args:
            results: List of result dicts from the benchmark run.
            server_info: Dictionary of server configuration values such as
                prefix_caching, max_num_seqs, gpu_memory_utilization,
                tensor_parallel, quantization, etc.

        Returns:
            List of Diagnostic objects with config recommendations.
        """
        diagnostics: list[Diagnostic] = []

        prompt_types = [r.get("prompt_type", "") for r in results if r.get("prompt_type")]
        all_random = all(pt == "random" for pt in prompt_types) if prompt_types else False
        has_deterministic = any(
            pt in ("deterministic", "fixed", "cached") for pt in prompt_types
        )

        # Rule 1: prefix caching disabled but deterministic prompts would benefit
        prefix_caching = server_info.get("prefix_caching")
        if not prefix_caching and has_deterministic:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    title="Enable prefix caching",
                    message=(
                        "Enable --enable-prefix-caching on your vLLM server. "
                        "Your deterministic prompts would benefit significantly."
                    ),
                    metric="prefix_caching",
                )
            )

        # Rule 2: prefix caching enabled but all prompts are random
        if prefix_caching and all_random:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    title="Prefix caching overhead with random prompts",
                    message=(
                        "You have prefix caching enabled but your prompts are "
                        "fully random — you're paying cache overhead with no "
                        "benefit. Consider disabling it for ~5-10% throughput gain."
                    ),
                    metric="prefix_caching",
                )
            )

        # Rule 3: max_num_seqs too low for tested concurrency
        max_num_seqs = server_info.get("max_num_seqs")
        if max_num_seqs is not None:
            max_users_tested = max(
                (r.get("concurrent_users", 0) for r in results), default=0
            )
            if max_num_seqs < max_users_tested:
                # Check if throughput at high concurrency is significantly lower
                high_conc = [
                    r for r in results
                    if r.get("concurrent_users", 0) >= max_users_tested
                    and r.get("tokens_per_second") is not None
                ]
                low_conc = [
                    r for r in results
                    if r.get("concurrent_users", 0) <= 1
                    and r.get("tokens_per_second") is not None
                ]
                high_tps = (
                    statistics.mean([r["tokens_per_second"] for r in high_conc])
                    if high_conc else None
                )
                low_tps = (
                    statistics.mean([r["tokens_per_second"] for r in low_conc])
                    if low_conc else None
                )
                # Emit if high-concurrency throughput is not scaling well
                if high_tps is not None and low_tps is not None and high_tps < low_tps * 1.5:
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            title="max_num_seqs too low for concurrency",
                            message=(
                                f"Your max_num_seqs ({max_num_seqs}) is lower than "
                                f"the concurrency you tested ({max_users_tested}). "
                                "Increase it to allow more concurrent requests."
                            ),
                            metric="max_num_seqs",
                        )
                    )

        # Rule 4: GPU memory utilization has headroom
        gpu_mem_util = server_info.get("gpu_memory_utilization")
        has_failures = any(r.get("failed", 0) > 0 for r in results)

        if gpu_mem_util is not None and gpu_mem_util < 0.85:
            diagnostics.append(
                Diagnostic(
                    severity="info",
                    title="GPU memory headroom available",
                    message=(
                        f"GPU memory utilization is set to {gpu_mem_util:.0%}. "
                        "You have headroom — consider increasing to 0.90-0.95 "
                        "for better KV cache capacity."
                    ),
                    metric="gpu_memory_utilization",
                )
            )

        # Rule 5: GPU memory utilization very high with failures
        if gpu_mem_util is not None and gpu_mem_util >= 0.95 and has_failures:
            diagnostics.append(
                Diagnostic(
                    severity="critical",
                    title="High memory utilization with failures",
                    message=(
                        f"GPU memory utilization is at {gpu_mem_util:.0%} and "
                        "you're seeing failures. Try reducing to 0.90."
                    ),
                    metric="gpu_memory_utilization",
                )
            )

        # Rule 6: tensor parallel with low GPU utilization
        tp = server_info.get("tensor_parallel")
        if tp is not None and tp > 1:
            gpu_utils = []
            for r in results:
                gpu = self._gpu(r)
                util = gpu.get("avg_gpu_util")
                if util is not None:
                    gpu_utils.append(util)
            if gpu_utils:
                avg_util = statistics.mean(gpu_utils)
                if avg_util < 70:
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            title="Low GPU utilization with tensor parallelism",
                            message=(
                                f"With tensor_parallel={tp}, GPU utilization is "
                                f"low ({avg_util:.0f}%). If running a single GPU "
                                "is feasible, TP=1 may give better single-stream "
                                "latency."
                            ),
                            metric="tensor_parallel",
                        )
                    )

        # Rule 7: FP16/BF16 with large model (low throughput + high memory)
        quantization = server_info.get("quantization")
        if quantization in ("FP16", "BF16", "FP16/BF16", None):
            tps_values = [
                r["tokens_per_second"]
                for r in results
                if r.get("tokens_per_second") is not None
            ]
            mem_usages = []
            for r in results:
                gpu = self._gpu(r)
                max_mem = gpu.get("max_mem_used")
                mem_total = gpu.get("mem_total")
                if max_mem is not None and mem_total is not None and mem_total > 0:
                    mem_usages.append(max_mem / mem_total)

            low_throughput = (
                tps_values and statistics.mean(tps_values) < 300
            )
            high_memory = mem_usages and statistics.mean(mem_usages) > 0.80

            if low_throughput and high_memory:
                diagnostics.append(
                    Diagnostic(
                        severity="info",
                        title="Consider quantization",
                        message=(
                            "Consider FP8 or AWQ quantization for better "
                            "throughput with minimal quality loss."
                        ),
                        metric="quantization",
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
