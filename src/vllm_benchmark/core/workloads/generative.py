"""Generative (chat/completions) workload.

A thin wrapper over the existing :func:`run_benchmark_async` path so the
default generative behavior is unchanged — this exists only to give the
runner a uniform :class:`Workload` interface.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm_benchmark.core.async_engine import run_benchmark_async
from vllm_benchmark.core.workloads.base import Workload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.config import BenchmarkConfig
    from vllm_benchmark.core.backends.base import ServerInfo


class GenerativeWorkload(Workload):
    """Standard generative chat/completions benchmark workload.

    Delegates execution to :func:`run_benchmark_async`, sweeping the
    configured context lengths x concurrency x prompt types.  Behavior is
    intentionally identical to the pre-existing benchmark path.
    """

    name = "generative"

    def applicable(self, server_info: "ServerInfo") -> bool:
        """Applicable for ``generate`` (or unknown) server tasks."""
        task = getattr(server_info, "task", "unknown")
        return task in ("generate", "unknown")

    async def run(
        self,
        config: "BenchmarkConfig",
        model_name: str,
        *,
        prompt_type: str = "classic",
        cost_per_hour: Optional[float] = None,
        gpu_monitor: Any = None,
        live_display: Any = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Run the generative sweep over context lengths x concurrency.

        Returns one result cell per ``(context_length, concurrency)`` pair.
        Cells that produce no successful requests are skipped.
        """
        results: list[dict] = []
        for context_length in config.context_lengths:
            for users in config.concurrency_levels:
                cell = await run_benchmark_async(
                    context_length,
                    users,
                    config,
                    model_name=model_name,
                    live_display=live_display,
                    gpu_monitor=gpu_monitor,
                    prompt_type=prompt_type,
                    cost_per_hour=cost_per_hour,
                )
                if cell:
                    cell.setdefault("workload", self.name)
                    results.append(cell)
        return results

    def summarize(self, results: list[dict]) -> dict:
        """Summarize peak throughput and best latency across the sweep."""
        if not results:
            return {"workload": self.name, "cells": 0}

        def _best(key: str, *, maximize: bool) -> Optional[float]:
            vals = [r[key] for r in results if r.get(key) is not None]
            if not vals:
                return None
            return max(vals) if maximize else min(vals)

        return {
            "workload": self.name,
            "cells": len(results),
            "peak_tokens_per_second": _best("tokens_per_second", maximize=True),
            "best_avg_latency": _best("avg_latency", maximize=False),
            "best_ttft": _best("ttft_estimate", maximize=False),
        }
