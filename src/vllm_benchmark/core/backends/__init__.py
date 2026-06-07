"""Inference-server backend abstraction.

Exposes the :class:`Backend` ABC, the normalized :class:`ServerInfo`
dataclass, concrete backends, and the :func:`detect_backend` helper.

Author: amit
License: MIT
"""

from __future__ import annotations

from vllm_benchmark.core.backends.base import Backend, ServerInfo
from vllm_benchmark.core.backends.detect import detect_backend
from vllm_benchmark.core.backends.openai_compat import (
    OpenAICompatBackend,
    infer_quantization,
)
from vllm_benchmark.core.backends.sglang import SGLangBackend
from vllm_benchmark.core.backends.vllm import VLLMBackend

__all__ = [
    "Backend",
    "ServerInfo",
    "OpenAICompatBackend",
    "VLLMBackend",
    "SGLangBackend",
    "detect_backend",
    "infer_quantization",
]
