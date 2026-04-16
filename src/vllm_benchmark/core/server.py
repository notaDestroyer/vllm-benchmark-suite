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


def capture_environment(server_info: Optional[Dict] = None) -> Dict:
    """Create a comprehensive, reproducible environment fingerprint.

    Gathers kernel, CPU, memory, GPU, Python, and package information
    into a single dictionary.  Every section is wrapped in try/except
    so that failures in one area never prevent the rest from being
    collected.

    Args:
        server_info: Optional dictionary returned by
            ``VLLMServerInfo.get_server_info``.  When provided, the
            vLLM version is extracted from it.

    Returns:
        Dictionary with timestamp, hardware details, software versions,
        per-GPU configuration, and a deterministic SHA-256 fingerprint
        for easy comparison across runs.
    """
    import hashlib
    import importlib.metadata
    import os

    # -- timestamp --------------------------------------------------------
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        timestamp = "unknown"

    # -- kernel -----------------------------------------------------------
    try:
        kernel = f"{platform.system()} {platform.release()}"
    except Exception:
        kernel = "unknown"

    # -- cpu --------------------------------------------------------------
    cpu_info: Dict = {"model": "unknown", "cores": 0, "governor": "unknown"}
    try:
        cpu_info["model"] = platform.processor() or "unknown"
        # platform.processor() can return an empty string on some Linux
        # systems; fall back to /proc/cpuinfo in that case.
        if cpu_info["model"] == "unknown" or cpu_info["model"] == "":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_info["model"] = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
    except Exception:
        pass

    try:
        cpu_info["cores"] = os.cpu_count() or 0
    except Exception:
        pass

    try:
        with open(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r"
        ) as f:
            cpu_info["governor"] = f.read().strip()
    except Exception:
        cpu_info["governor"] = "unknown"

    # -- memory -----------------------------------------------------------
    memory_info: Dict = {"total_gb": 0, "available_gb": 0}
    try:
        import psutil

        vm = psutil.virtual_memory()
        memory_info["total_gb"] = round(vm.total / (1024 ** 3), 1)
        memory_info["available_gb"] = round(vm.available / (1024 ** 3), 1)
    except Exception:
        # Fallback: parse /proc/meminfo
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo: Dict = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1])  # value in kB
                total_kb = meminfo.get("MemTotal", 0)
                avail_kb = meminfo.get("MemAvailable", 0)
                memory_info["total_gb"] = round(total_kb / (1024 ** 2), 1)
                memory_info["available_gb"] = round(avail_kb / (1024 ** 2), 1)
        except Exception:
            pass

    # -- gpu --------------------------------------------------------------
    gpu_info: Dict = {
        "name": "unknown",
        "count": 0,
        "driver_version": "unknown",
        "cuda_version": "unknown",
        "per_gpu": [],
    }

    try:
        name = SystemInfo.get_gpu_name()
        if name:
            # get_gpu_name may return multiple lines for multi-GPU; take
            # the first.
            gpu_info["name"] = name.splitlines()[0].strip()
    except Exception:
        pass

    try:
        driver = SystemInfo.get_driver_version()
        if driver:
            gpu_info["driver_version"] = driver.splitlines()[0].strip()
    except Exception:
        pass

    try:
        cuda = SystemInfo.get_cuda_version()
        if cuda:
            gpu_info["cuda_version"] = cuda
    except Exception:
        pass

    # GPU count
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=count",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    gpu_info["count"] = int(line)
                    break
    except Exception:
        pass

    # Per-GPU details
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,persistence_mode,power.limit,"
                "clocks.max.graphics,memory.total,"
                "pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    try:
                        gpu_entry: Dict = {
                            "index": int(parts[0]),
                            "persistence_mode": parts[1],
                            "power_limit_w": _safe_numeric(parts[2]),
                            "max_clock_mhz": _safe_numeric(parts[3]),
                            "memory_total_mb": _safe_numeric(parts[4]),
                            "pcie_gen": _safe_numeric(parts[5]),
                            "pcie_width": _safe_numeric(parts[6]),
                        }
                        gpu_info["per_gpu"].append(gpu_entry)
                    except Exception:
                        pass
    except Exception:
        pass

    # -- python version ---------------------------------------------------
    try:
        python_version = platform.python_version()
    except Exception:
        python_version = "unknown"

    # -- vllm version -----------------------------------------------------
    vllm_version = "unknown"
    if server_info and isinstance(server_info, dict):
        vllm_version = server_info.get("version") or "unknown"
    if vllm_version == "unknown":
        try:
            vllm_version = importlib.metadata.version("vllm")
        except Exception:
            pass

    # -- key package versions ---------------------------------------------
    packages: Dict = {}
    for pkg in ("torch", "transformers", "numpy"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except Exception:
            packages[pkg] = "unknown"

    # -- assemble the environment dict (without fingerprint) --------------
    env: Dict = {
        "timestamp": timestamp,
        "kernel": kernel,
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "python_version": python_version,
        "vllm_version": vllm_version,
        "packages": packages,
    }

    # -- deterministic fingerprint ----------------------------------------
    try:
        canonical = json.dumps(env, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        fingerprint = "unknown"

    env["fingerprint"] = fingerprint
    return env


def _safe_numeric(value: str):
    """Convert a string to int or float, returning the string on failure."""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value
