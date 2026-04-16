"""Server information and system detection utilities.

Provides SystemInfo for local hardware detection and VLLMServerInfo
for querying the vLLM server's configuration, model, and capabilities.

Author: amit
License: MIT
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional

import requests

from vllm_benchmark.config import BenchmarkConfig


class SystemInfo:
    """Collect and store system configuration information."""

    @staticmethod
    def get_cuda_version() -> Optional[str]:
        """Get CUDA version from nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                match = re.search(r"CUDA Version: ([\d.]+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return None

    @staticmethod
    def get_driver_version() -> Optional[str]:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def get_gpu_name() -> Optional[str]:
        """Get GPU model name."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def get_total_vram() -> Optional[float]:
        """Get total GPU VRAM in GB."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 1024  # Convert MB to GB
        except Exception:
            pass
        return None

    @staticmethod
    def get_system_info() -> Dict:
        """Collect comprehensive system information."""
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_version": SystemInfo.get_cuda_version(),
            "driver_version": SystemInfo.get_driver_version(),
            "gpu_name": SystemInfo.get_gpu_name(),
            "total_vram_gb": SystemInfo.get_total_vram(),
            "timestamp": datetime.now().isoformat(),
        }


class VLLMServerInfo:
    """Query and store vLLM server configuration and capabilities."""

    @staticmethod
    def get_server_info(config: BenchmarkConfig) -> Dict:
        """Retrieve comprehensive vLLM server information.

        Args:
            config: Benchmark configuration providing endpoint URLs.

        Returns:
            Dictionary containing server model, version, quantization,
            parallelism settings, and other detected capabilities.
        """
        info: Dict = {
            "model_name": None,
            "max_model_len": None,
            "backend": None,
            "version": None,
            "quantization": None,
            "tensor_parallel": None,
            "pipeline_parallel": None,
            "max_num_seqs": None,
            "gpu_memory_utilization": None,
            "kv_cache_usage": None,
            "prefix_caching": None,
            "additional_info": {},
        }

        # Try to get model information
        try:
            response = requests.get(config.models_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_data = data["data"][0]
                    info["model_name"] = model_data.get("id")
                    if "max_model_len" in model_data:
                        info["max_model_len"] = model_data["max_model_len"]
                    # Some vLLM versions expose root with more details
                    if "root" in model_data:
                        info["additional_info"]["root"] = model_data["root"]
                    print(
                        f"[INFO] Model endpoint: {json.dumps(model_data, indent=2)}"
                    )
        except Exception as e:
            print(
                f"[WARNING] Failed to query models endpoint: {e}", file=sys.stderr
            )

        # Try version endpoint
        try:
            response = requests.get(config.version_endpoint, timeout=5)
            if response.status_code == 200:
                version_data = response.json()
                info["version"] = version_data.get("version")
                print(f"[INFO] vLLM Version: {info['version']}")
        except Exception:
            pass

        # Try metrics endpoint (Prometheus format)
        try:
            response = requests.get(config.metrics_endpoint, timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                print("[INFO] Metrics endpoint available")

                # Parse key metrics
                for line in metrics_text.split("\n"):
                    if line.startswith("#") or not line.strip():
                        continue

                    # KV cache usage
                    if "vllm:gpu_cache_usage_perc" in line:
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                info["kv_cache_usage"] = float(parts[-1])
                        except Exception:
                            pass

                    # Number of running requests
                    if "vllm:num_requests_running" in line:
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                info["additional_info"]["running_requests"] = int(
                                    float(parts[-1])
                                )
                        except Exception:
                            pass

        except Exception as e:
            print(
                f"[INFO] Metrics endpoint not available: {e}", file=sys.stderr
            )

        # Try to get server args from health endpoint (some versions expose this)
        try:
            response = requests.get(config.health_endpoint, timeout=5)
            if response.status_code == 200:
                health_data = response.json()

                # Some vLLM versions include server config in health response
                if "model_config" in health_data:
                    model_cfg = health_data["model_config"]
                    info["additional_info"]["model_config"] = model_cfg

                print(f"[INFO] Health: {json.dumps(health_data, indent=2)}")
        except Exception:
            pass

        # Try completions endpoint with special system prompt to get config
        try:
            test_data = {
                "model": info["model_name"] or config.model_name or "unknown",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "logprobs": True,
            }
            response = requests.post(
                config.api_endpoint, json=test_data, timeout=10
            )
            if response.status_code == 200:
                # Check headers for server info
                if "x-request-id" in response.headers:
                    info["additional_info"]["supports_request_id"] = True
        except Exception:
            pass

        # Infer quantization from model name
        model_name = info["model_name"] or config.model_name or ""
        model_name_upper = model_name.upper()
        if "FP8" in model_name_upper:
            info["quantization"] = "FP8"
        elif "AWQ" in model_name_upper:
            info["quantization"] = "AWQ"
        elif "GPTQ" in model_name_upper:
            info["quantization"] = "GPTQ"
        elif "INT8" in model_name_upper:
            info["quantization"] = "INT8"
        elif "INT4" in model_name_upper:
            info["quantization"] = "INT4"
        else:
            info["quantization"] = "FP16/BF16"

        return info
