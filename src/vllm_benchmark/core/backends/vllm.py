"""vLLM backend.

Detects a vLLM server and queries its model, version, and Prometheus
metrics into a normalized :class:`ServerInfo`.

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


def _parse_metric_value(line: str) -> float | None:
    """Parse the trailing numeric value of a Prometheus metric line."""
    try:
        parts = line.split()
        if len(parts) >= 2:
            return float(parts[-1])
    except Exception:
        pass
    return None


class VLLMBackend(OpenAICompatBackend):
    """Backend for vLLM OpenAI-compatible servers."""

    name = "vllm"

    @classmethod
    def detect(cls, base_url: str, timeout: float = 5.0) -> bool:
        """Detect a vLLM server.

        Matches when ``/version`` returns a vLLM version, or when
        ``/metrics`` contains a line beginning with ``vllm:``.
        """
        base_url = base_url.rstrip("/")
        try:
            response = requests.get(f"{base_url}/version", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get("version"):
                    return True
        except Exception:
            pass

        try:
            response = requests.get(f"{base_url}/metrics", timeout=timeout)
            if response.status_code == 200:
                for line in response.text.split("\n"):
                    if line.startswith("vllm:"):
                        return True
        except Exception:
            pass

        return False

    # ------------------------------------------------------------------

    def server_info(self, cfg) -> ServerInfo:
        """Query the vLLM server and return a normalized :class:`ServerInfo`."""
        info = ServerInfo(backend="vllm")

        # /v1/models — model name and max context length
        self._query_models(cfg, info)

        # /version — backend version
        try:
            response = requests.get(cfg.version_endpoint, timeout=5)
            if response.status_code == 200:
                info.backend_version = response.json().get("version")
        except Exception:
            pass

        # /metrics — cache usage, running requests, speculative acceptance
        try:
            response = requests.get(cfg.metrics_endpoint, timeout=5)
            if response.status_code == 200:
                spec_accepted: float | None = None
                spec_draft: float | None = None
                for line in response.text.split("\n"):
                    if line.startswith("#") or not line.strip():
                        continue

                    if "vllm:gpu_cache_usage_perc" in line:
                        value = _parse_metric_value(line)
                        if value is not None:
                            info.raw["kv_cache_usage"] = value

                    if "vllm:num_requests_running" in line:
                        value = _parse_metric_value(line)
                        if value is not None:
                            info.raw["running_requests"] = int(value)

                    if "vllm:spec_decode_num_accepted_tokens_total" in line:
                        value = _parse_metric_value(line)
                        if value is not None:
                            spec_accepted = value

                    if "vllm:spec_decode_num_draft_tokens_total" in line:
                        value = _parse_metric_value(line)
                        if value is not None:
                            spec_draft = value

                if spec_draft is not None and spec_accepted is not None:
                    acceptance_rate = (spec_accepted / spec_draft) if spec_draft > 0 else 0.0
                    info.speculative = {
                        "accepted_tokens": spec_accepted,
                        "draft_tokens": spec_draft,
                        "acceptance_rate": acceptance_rate,
                    }
        except Exception:
            pass

        # Infer quantization from name
        info.quantization = infer_quantization(info.model_name or getattr(cfg, "model_name", None))

        return info

    def metrics_monitor(self, cfg):
        """Return a vLLM Prometheus metrics monitor."""
        from vllm_benchmark.core.metrics import MetricsMonitor

        return MetricsMonitor(cfg.metrics_endpoint)
