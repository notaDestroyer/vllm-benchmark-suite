"""Backend abstraction base classes.

Defines :class:`ServerInfo` (a normalized description of an inference
server's configuration) and the :class:`Backend` abstract base class that
concrete backends (vLLM, SGLang, generic OpenAI-compatible) implement.

Author: amit
License: MIT
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ServerInfo:
    """Normalized inference server information.

    A backend-agnostic description of a server's model, parallelism,
    quantization, and task configuration.  Fields that a particular
    backend cannot report are left as ``None``.
    """

    backend: Literal["vllm", "sglang", "unknown"]
    backend_version: Optional[str] = None
    model_name: Optional[str] = None
    served_model_path: Optional[str] = None
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    kv_cache_dtype: Optional[str] = None
    dtype: Optional[str] = None
    tensor_parallel: Optional[int] = None
    pipeline_parallel: Optional[int] = None
    expert_parallel: Optional[int] = None
    max_num_seqs: Optional[int] = None
    prefix_caching: Optional[bool] = None
    speculative: Optional[dict] = None
    task: Literal["generate", "embed", "rerank", "unknown"] = "unknown"
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation of this server info."""
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "model_name": self.model_name,
            "served_model_path": self.served_model_path,
            "max_model_len": self.max_model_len,
            "quantization": self.quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "dtype": self.dtype,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "expert_parallel": self.expert_parallel,
            "max_num_seqs": self.max_num_seqs,
            "prefix_caching": self.prefix_caching,
            "speculative": self.speculative,
            "task": self.task,
            "raw": self.raw,
        }


class Backend(abc.ABC):
    """Abstract base class for an inference-server backend.

    Concrete backends know how to detect a particular server type and how
    to query its configuration into a normalized :class:`ServerInfo`.
    """

    #: Short backend identifier ("vllm", "sglang", "unknown").
    name: str = "unknown"

    def __init__(self, base_url: str) -> None:
        """Initialize the backend.

        Args:
            base_url: Server base URL, e.g. ``"http://localhost:8000"``.
        """
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def detect(cls, base_url: str, timeout: float = 5.0) -> bool:
        """Probe ``base_url`` to determine whether this backend matches.

        Implementations must never raise on network/timeout errors;
        they should return ``False`` instead.

        Args:
            base_url: Server base URL.
            timeout: Per-request timeout in seconds.

        Returns:
            ``True`` if the server appears to be this backend.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def server_info(self, cfg) -> ServerInfo:
        """Query the server and return a normalized :class:`ServerInfo`.

        Args:
            cfg: Benchmark configuration providing endpoint URLs.

        Returns:
            A populated :class:`ServerInfo`.
        """
        raise NotImplementedError

    def metrics_monitor(self, cfg):
        """Return a metrics monitor for this backend, or ``None``.

        Args:
            cfg: Benchmark configuration providing endpoint URLs.

        Returns:
            A ``MetricsMonitor`` instance or ``None`` if unsupported.
        """
        return None

    # ------------------------------------------------------------------
    # Endpoint helpers
    # ------------------------------------------------------------------

    def chat_endpoint(self, cfg) -> str:
        """Return the chat-completions endpoint URL."""
        return f"{cfg.api_url}/v1/chat/completions"

    def completions_endpoint(self, cfg) -> str:
        """Return the text-completions endpoint URL."""
        return f"{cfg.api_url}/v1/completions"

    def embeddings_endpoint(self, cfg) -> str:
        """Return the embeddings endpoint URL."""
        return f"{cfg.api_url}/v1/embeddings"
