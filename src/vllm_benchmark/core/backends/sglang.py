"""SGLang backend.

Detects an SGLang server and maps its ``/get_server_info`` and
``/get_model_info`` endpoints into a normalized :class:`ServerInfo`.

Author: amit
License: MIT
"""

from __future__ import annotations

import requests

from vllm_benchmark.core.backends.base import ServerInfo
from vllm_benchmark.core.backends.openai_compat import (
    OpenAICompatBackend,
    infer_quantization,
)


class SGLangBackend(OpenAICompatBackend):
    """Backend for SGLang OpenAI-compatible servers."""

    name = "sglang"

    @classmethod
    def detect(cls, base_url: str, timeout: float = 5.0) -> bool:
        """Detect an SGLang server.

        Matches when both ``/get_model_info`` and ``/get_server_info``
        return HTTP 200, or when ``/metrics`` contains an ``sglang:`` line.
        """
        base_url = base_url.rstrip("/")
        try:
            model_resp = requests.get(f"{base_url}/get_model_info", timeout=timeout)
            server_resp = requests.get(f"{base_url}/get_server_info", timeout=timeout)
            if model_resp.status_code == 200 and server_resp.status_code == 200:
                return True
        except Exception:
            pass

        try:
            response = requests.get(f"{base_url}/metrics", timeout=timeout)
            if response.status_code == 200:
                for line in response.text.split("\n"):
                    if line.startswith("sglang:"):
                        return True
        except Exception:
            pass

        return False

    # ------------------------------------------------------------------

    def server_info(self, cfg) -> ServerInfo:
        """Query the SGLang server and return a normalized :class:`ServerInfo`."""
        info = ServerInfo(backend="sglang")

        # OpenAI-compatible model listing (model_name, max_model_len)
        self._query_models(cfg, info)

        # /get_server_info — parallelism and scheduling config
        try:
            response = requests.get(f"{self.base_url}/get_server_info", timeout=5)
            if response.status_code == 200:
                server_data = response.json()
                info.raw["server_info"] = server_data

                if server_data.get("tp_size") is not None:
                    info.tensor_parallel = server_data["tp_size"]
                if server_data.get("dp_size") is not None:
                    info.raw["dp_size"] = server_data["dp_size"]
                if server_data.get("max_running_requests") is not None:
                    info.max_num_seqs = server_data["max_running_requests"]
                if server_data.get("attention_backend") is not None:
                    info.raw["attention_backend"] = server_data["attention_backend"]
                if server_data.get("version"):
                    info.backend_version = server_data["version"]

                spec = {
                    k: v for k, v in server_data.items() if k.startswith("speculative_")
                }
                if spec:
                    info.speculative = spec
        except Exception:
            pass

        # /get_model_info — model path and task
        try:
            response = requests.get(f"{self.base_url}/get_model_info", timeout=5)
            if response.status_code == 200:
                model_data = response.json()
                info.raw["model_info"] = model_data

                if model_data.get("model_path"):
                    info.served_model_path = model_data["model_path"]
                    if not info.model_name:
                        info.model_name = model_data["model_path"]
                if model_data.get("tokenizer_path"):
                    info.raw["tokenizer_path"] = model_data["tokenizer_path"]
                if "is_generation" in model_data:
                    info.task = "generate" if model_data["is_generation"] else "embed"
        except Exception:
            pass

        # Infer quantization from name/path
        info.quantization = infer_quantization(
            info.model_name or info.served_model_path or getattr(cfg, "model_name", None)
        )

        return info
