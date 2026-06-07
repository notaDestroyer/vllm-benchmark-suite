"""Workload package — uniform abstractions over inference traffic kinds.

Exposes the :class:`Workload` ABC and its concrete implementations
(generative chat/completions, embeddings, structured/function-calling),
plus :func:`select_workloads` which picks the applicable workload(s) for a
server given an optional explicit request.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from vllm_benchmark.core.workloads.base import Workload
from vllm_benchmark.core.workloads.embeddings import (
    EmbeddingsWorkload,
    is_embedding_model,
)
from vllm_benchmark.core.workloads.generative import GenerativeWorkload
from vllm_benchmark.core.workloads.structured import (
    StructuredWorkload,
    score_structured_response,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.core.backends.base import ServerInfo

__all__ = [
    "Workload",
    "GenerativeWorkload",
    "EmbeddingsWorkload",
    "StructuredWorkload",
    "is_embedding_model",
    "score_structured_response",
    "select_workloads",
]

#: Valid values for the ``--workload`` selector.
_VALID = ("auto", "generative", "embeddings", "structured")


def select_workloads(
    server_info: "ServerInfo",
    requested: Optional[str] = None,
) -> list[Workload]:
    """Select the workload(s) to run for a server.

    Args:
        server_info: Normalized server description from a backend.
        requested: One of ``{"auto", "generative", "embeddings",
            "structured"}`` (or ``None`` == ``"auto"``).  An explicit
            value forces that single workload; ``"auto"`` picks by the
            server's task / model name.

    Returns:
        A list of workload instances.  For ``"auto"`` this is the embeddings
        workload when the server is an embedding/rerank model, otherwise the
        generative workload (the default path, unchanged from prior
        behavior).  Never empty — falls back to generative.
    """
    choice = (requested or "auto").lower()
    if choice not in _VALID:
        choice = "auto"

    if choice == "generative":
        return [GenerativeWorkload()]
    if choice == "embeddings":
        return [EmbeddingsWorkload()]
    if choice == "structured":
        return [StructuredWorkload()]

    # auto: prefer embeddings only when the server clearly is one; otherwise
    # keep the default generative path identical to today.
    embeddings = EmbeddingsWorkload()
    if embeddings.applicable(server_info):
        task = getattr(server_info, "task", "unknown")
        name = getattr(server_info, "model_name", None) or getattr(
            server_info, "served_model_path", None
        )
        if task in ("embed", "rerank") or is_embedding_model(name):
            return [embeddings]
    return [GenerativeWorkload()]
