"""Shared OpenAI-compatible backend behavior.

Implements the HTTP behavior common to vLLM, SGLang, and any other
OpenAI-compatible server: querying ``/v1/models`` and inferring the
quantization scheme from the model name.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import Optional

import requests

from vllm_benchmark.core.backends.base import Backend, ServerInfo


def infer_quantization(model_name: Optional[str]) -> str:
    """Infer the quantization scheme from a model name.

    Mirrors the inference logic historically used by
    ``VLLMServerInfo.get_server_info``.

    Args:
        model_name: The served model name, may be ``None``.

    Returns:
        A quantization label such as ``"FP8"`` or ``"FP16/BF16"``.
    """
    model_name_upper = (model_name or "").upper()
    if "FP8" in model_name_upper:
        return "FP8"
    elif "AWQ" in model_name_upper:
        return "AWQ"
    elif "GPTQ" in model_name_upper:
        return "GPTQ"
    elif "INT8" in model_name_upper:
        return "INT8"
    elif "INT4" in model_name_upper:
        return "INT4"
    return "FP16/BF16"


class OpenAICompatBackend(Backend):
    """Base backend for OpenAI-compatible inference servers.

    Provides shared querying of the ``/v1/models`` endpoint and
    quantization inference.  Used directly for unknown servers and as a
    base class for the vLLM and SGLang backends.
    """

    name = "unknown"

    @classmethod
    def detect(cls, base_url: str, timeout: float = 5.0) -> bool:
        """A generic OpenAI-compatible server cannot be uniquely detected.

        Returns ``False`` so that detection falls back explicitly rather
        than matching any server.
        """
        return False

    # ------------------------------------------------------------------
    # Shared HTTP helpers
    # ------------------------------------------------------------------

    def _query_models(self, cfg, info: ServerInfo) -> None:
        """Populate ``info`` from the ``/v1/models`` endpoint.

        Sets ``model_name`` and ``max_model_len`` when available.  Never
        raises; network failures are swallowed.
        """
        try:
            response = requests.get(cfg.models_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_data = data["data"][0]
                    info.model_name = model_data.get("id")
                    if "max_model_len" in model_data:
                        info.max_model_len = model_data["max_model_len"]
                    if "root" in model_data:
                        info.raw["root"] = model_data["root"]
        except Exception:
            pass

    def server_info(self, cfg) -> ServerInfo:
        """Return a minimal :class:`ServerInfo` for an unknown server.

        Queries ``/v1/models`` for the model name/context length and
        infers quantization from the name.
        """
        info = ServerInfo(backend=self.name if self.name in ("vllm", "sglang") else "unknown")
        self._query_models(cfg, info)
        info.quantization = infer_quantization(info.model_name or getattr(cfg, "model_name", None))
        return info
