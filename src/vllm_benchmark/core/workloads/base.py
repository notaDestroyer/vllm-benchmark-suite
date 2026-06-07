"""Workload abstraction base class.

A :class:`Workload` is a uniform interface over the different *kinds* of
inference traffic a server can serve — generative chat/completions,
embeddings, structured/function-calling output, and so on.  The runner
selects one or more applicable workloads for a server and drives them
through the same ``applicable`` / ``run`` / ``summarize`` contract.

Author: amit
License: MIT
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.config import BenchmarkConfig
    from vllm_benchmark.core.backends.base import ServerInfo


class Workload(abc.ABC):
    """Abstract base class for a benchmark workload.

    Concrete workloads know which servers they apply to, how to execute a
    sweep of measured request cells, and how to summarize those cells into
    a single metrics dict consumed by the advisor / reports.
    """

    #: Short workload identifier ("generative", "embeddings", ...).
    name: str = "workload"

    @abc.abstractmethod
    def applicable(self, server_info: "ServerInfo") -> bool:
        """Return whether this workload can run against ``server_info``.

        Implementations must never raise; an indeterminate server should
        return ``False`` (or ``True`` only when there is positive
        evidence the workload is supported).

        Args:
            server_info: Normalized server description from a backend.

        Returns:
            ``True`` if this workload is applicable.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def run(
        self,
        config: "BenchmarkConfig",
        model_name: str,
        **kwargs: Any,
    ) -> list[dict]:
        """Execute the workload and return a list of per-cell result dicts.

        Args:
            config: Benchmark configuration (endpoint, sweep, timeout).
            model_name: Model id to send to the server.
            **kwargs: Workload-specific extras (e.g. monitors).

        Returns:
            A list of result-cell dicts in the same shape used elsewhere
            in the suite (one cell per sweep point).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def summarize(self, results: list[dict]) -> dict:
        """Summarize per-cell results into a single metrics dict.

        Args:
            results: The cells returned by :meth:`run`.

        Returns:
            A dict of workload-level metrics (peak throughput, adherence
            rate, docs/sec, ...).  Empty input yields an empty-ish dict
            rather than raising.
        """
        raise NotImplementedError


def _coalesce(*values: Optional[float]) -> Optional[float]:
    """Return the first non-``None`` value, else ``None``."""
    for v in values:
        if v is not None:
            return v
    return None
