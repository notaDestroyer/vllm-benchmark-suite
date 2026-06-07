"""Embeddings (and optional rerank) workload.

Drives an OpenAI-compatible ``/v1/embeddings`` endpoint with a batch-size
sweep and reports document throughput, token throughput and latency
percentiles.  Embeddings are a single forward pass, so there is **no**
TTFT or decode-throughput concept — those fields are deliberately omitted
from the result cells rather than fabricated.

An optional rerank path is attempted only when a rerank endpoint pattern
is detected; otherwise it is skipped cleanly.

Author: amit
License: MIT
"""

from __future__ import annotations

import re
import time
from statistics import mean
from typing import TYPE_CHECKING, Any, Optional

import aiohttp

from vllm_benchmark.core.async_engine import calculate_percentiles, count_tokens
from vllm_benchmark.core.workloads.base import Workload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.config import BenchmarkConfig
    from vllm_benchmark.core.backends.base import ServerInfo


# ---------------------------------------------------------------------------
# Embedding-model detection
# ---------------------------------------------------------------------------

#: Substrings that strongly indicate an embedding / reranker / encoder model.
_EMBEDDING_PATTERNS = (
    "e5",
    "bge",
    "gte",
    "sentence-transformers",
    "sentence-transformer",
    "all-minilm",
    "all-mpnet",
    "bert",
    "roberta",
    "xlm",
    "nomic-embed",
    "instructor",
    "jina-embed",
    "jina-embeddings",
    "gtr",
    "contriever",
    "rerank",
    "reranker",
)


def is_embedding_model(name: Optional[str]) -> bool:
    """Heuristically decide whether a model name denotes an embedding model.

    Matches well-known embedding / reranker / encoder families (BERT,
    RoBERTa, XLM, e5, bge, gte, sentence-transformers, ...).  Matching is
    word-boundary aware for short tokens (``e5``, ``gte``, ``bge``) to
    avoid false positives inside unrelated names.

    Args:
        name: Model name or HuggingFace repo id.

    Returns:
        ``True`` if the name looks like an embedding/reranker model.
    """
    if not name:
        return False
    lower = name.lower()
    # Short ambiguous tokens require a word boundary; longer ones may match
    # as a substring.
    short = {"e5", "bge", "gte", "gtr", "xlm"}
    for pat in _EMBEDDING_PATTERNS:
        if pat in short:
            if re.search(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", lower):
                return True
        elif pat in lower:
            return True
    return False


def _looks_like_rerank(name: Optional[str], server_info: "ServerInfo") -> bool:
    """Return whether a rerank path should be attempted."""
    task = getattr(server_info, "task", "unknown")
    if task == "rerank":
        return True
    if name and ("rerank" in name.lower()):
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP helper (monkeypatchable for tests)
# ---------------------------------------------------------------------------

async def _post_embeddings(
    session: aiohttp.ClientSession,
    endpoint: str,
    model_name: str,
    inputs: list[str],
    request_timeout: int,
) -> dict:
    """POST a single embeddings request and measure its wall-clock latency.

    This is the sole network boundary of the workload and is intended to
    be monkeypatched in unit tests.

    Returns:
        A per-request dict with ``success``, ``duration``, ``num_docs`` and
        (on success) the parsed response ``body``.
    """
    body = {"model": model_name, "input": inputs}
    start = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with session.post(endpoint, json=body, timeout=timeout) as response:
            duration = time.perf_counter() - start
            if response.status != 200:
                return {
                    "success": False,
                    "duration": duration,
                    "num_docs": len(inputs),
                    "error": f"HTTP {response.status}",
                }
            payload = await response.json()
            return {
                "success": True,
                "duration": duration,
                "num_docs": len(inputs),
                "body": payload,
            }
    except Exception as exc:  # network errors are recorded, never raised
        return {
            "success": False,
            "duration": time.perf_counter() - start,
            "num_docs": len(inputs),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class EmbeddingsWorkload(Workload):
    """Embeddings throughput / latency workload with a batch-size sweep.

    Uses ``config.concurrency_levels`` as the batch-size sweep (each level
    is the number of documents sent in one request).  Reports docs/sec,
    tokens/sec and latency p50/p90/p99 per batch size.
    """

    name = "embeddings"

    #: A small, fixed probe corpus of documents to embed.
    _PROBE_DOCS = (
        "The mitochondria is the powerhouse of the cell.",
        "Quarterly revenue grew 12% year over year on strong demand.",
        "To reset your password, click the link in the verification email.",
        "Photosynthesis converts sunlight, water and carbon dioxide into glucose.",
        "The capital of France is Paris, a major European cultural center.",
        "Kubernetes orchestrates containerized applications across clusters.",
        "A balanced diet includes proteins, carbohydrates, fats and fiber.",
        "The novel explores themes of memory, identity and belonging.",
    )

    def applicable(self, server_info: "ServerInfo") -> bool:
        """Applicable for ``embed``/``rerank`` tasks or embedding model names."""
        task = getattr(server_info, "task", "unknown")
        if task in ("embed", "rerank"):
            return True
        name = getattr(server_info, "model_name", None) or getattr(
            server_info, "served_model_path", None
        )
        return is_embedding_model(name)

    def _make_batch(self, batch_size: int) -> list[str]:
        """Build a document batch of ``batch_size`` by cycling the probe set."""
        docs = list(self._PROBE_DOCS)
        if batch_size <= len(docs):
            return docs[:batch_size]
        out: list[str] = []
        while len(out) < batch_size:
            out.extend(docs)
        return out[:batch_size]

    async def run(
        self,
        config: "BenchmarkConfig",
        model_name: str,
        *,
        server_info: "ServerInfo" = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Run the batch-size sweep against the embeddings endpoint.

        Returns one cell per batch size.  Cells contain ``docs_per_second``,
        ``tokens_per_second`` and latency percentiles, and deliberately
        omit ``ttft``/``decode_tps`` fields (single forward pass).
        """
        endpoint = f"{config.api_url}/v1/embeddings"
        batch_sizes = config.concurrency_levels or [1, 4, 8]
        results: list[dict] = []

        connector = aiohttp.TCPConnector(limit=max(batch_sizes) + 10)
        async with aiohttp.ClientSession(connector=connector) as session:
            for batch in batch_sizes:
                inputs = self._make_batch(batch)
                token_count = sum(count_tokens(d, model_name or "") for d in inputs)
                start = time.perf_counter()
                req = await _post_embeddings(
                    session, endpoint, model_name, inputs, config.request_timeout,
                )
                total_time = time.perf_counter() - start
                cell = self._cell_from_request(req, batch, token_count, total_time)
                if cell is not None:
                    results.append(cell)

        # Optional rerank probe (best-effort, skipped when not detected).
        if server_info is not None and _looks_like_rerank(model_name, server_info):
            results.append({
                "workload": self.name,
                "batch_size": 0,
                "rerank": True,
                "rerank_status": "detected_not_implemented",
            })

        return results

    def _cell_from_request(
        self,
        req: dict,
        batch: int,
        token_count: int,
        total_time: float,
    ) -> Optional[dict]:
        """Convert a per-request result into a sweep cell, or ``None``."""
        if not req.get("success"):
            return None
        duration = req.get("duration", total_time) or total_time
        docs = req.get("num_docs", batch)
        docs_per_second = docs / duration if duration > 0 else 0.0
        tokens_per_second = token_count / duration if duration > 0 else 0.0
        # A single request gives one latency sample; percentiles collapse to
        # the same value, which is the honest representation for one probe.
        pct = calculate_percentiles([duration])
        return {
            "workload": self.name,
            "batch_size": batch,
            "concurrent_users": batch,
            "num_docs": docs,
            "docs_per_second": docs_per_second,
            "tokens_per_second": tokens_per_second,
            "avg_latency": duration,
            "latency_p50": pct["p50"],
            "latency_p90": pct["p90"],
            "latency_p99": pct["p99"],
        }

    def summarize(self, results: list[dict]) -> dict:
        """Summarize peak docs/sec, tokens/sec and best latency."""
        cells = [r for r in results if not r.get("rerank")]
        if not cells:
            return {"workload": self.name, "cells": 0}
        docs_ps = [c["docs_per_second"] for c in cells if c.get("docs_per_second") is not None]
        toks_ps = [c["tokens_per_second"] for c in cells if c.get("tokens_per_second") is not None]
        lats = [c["avg_latency"] for c in cells if c.get("avg_latency") is not None]
        return {
            "workload": self.name,
            "cells": len(cells),
            "peak_docs_per_second": max(docs_ps) if docs_ps else None,
            "peak_tokens_per_second": max(toks_ps) if toks_ps else None,
            "best_avg_latency": min(lats) if lats else None,
            "mean_latency": mean(lats) if lats else None,
            "rerank_probed": any(r.get("rerank") for r in results),
        }
