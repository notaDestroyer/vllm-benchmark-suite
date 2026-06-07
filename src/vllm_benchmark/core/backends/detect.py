"""Backend detection.

Probes a server URL to determine which :class:`Backend` implementation
should be used, with an optional forced override.

Author: amit
License: MIT
"""

from __future__ import annotations

from vllm_benchmark.core.backends.base import Backend
from vllm_benchmark.core.backends.openai_compat import OpenAICompatBackend
from vllm_benchmark.core.backends.sglang import SGLangBackend
from vllm_benchmark.core.backends.vllm import VLLMBackend

#: Probe order for auto-detection.
_PROBE_ORDER = [VLLMBackend, SGLangBackend]


def detect_backend(base_url: str, forced: str | None = None, timeout: float = 5.0) -> Backend:
    """Detect and instantiate the appropriate backend for ``base_url``.

    Args:
        base_url: Server base URL, e.g. ``"http://localhost:8000"``.
        forced: When ``"vllm"`` or ``"sglang"``, force that backend and
            skip probing.  ``None`` or ``"auto"`` performs auto-detection.
        timeout: Per-probe timeout in seconds.

    Returns:
        An instantiated :class:`Backend`.  Falls back to a generic
        :class:`OpenAICompatBackend` with ``backend="unknown"`` when no
        known backend is detected.  Never raises on network errors.
    """
    if forced == "vllm":
        return VLLMBackend(base_url)
    if forced == "sglang":
        return SGLangBackend(base_url)

    for backend_cls in _PROBE_ORDER:
        try:
            if backend_cls.detect(base_url, timeout=timeout):
                return backend_cls(base_url)
        except Exception:
            continue

    return OpenAICompatBackend(base_url)
