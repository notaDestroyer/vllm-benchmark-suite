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
from datetime import datetime
from typing import Dict, Optional

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

        This now delegates to the vLLM backend
        (:class:`vllm_benchmark.core.backends.vllm.VLLMBackend`) and maps
        the normalized :class:`ServerInfo` back onto the historical dict
        shape so that existing callers continue to work unchanged.

        Args:
            config: Benchmark configuration providing endpoint URLs.

        Returns:
            Dictionary containing server model, version, quantization,
            parallelism settings, and other detected capabilities.  A
            superset of the historical keys is returned.
        """
        from vllm_benchmark.core.backends.vllm import VLLMBackend

        backend = VLLMBackend(config.api_url)
        server = backend.server_info(config)

        raw = server.raw or {}
        additional_info: Dict = {}
        if "root" in raw:
            additional_info["root"] = raw["root"]
        if "running_requests" in raw:
            additional_info["running_requests"] = raw["running_requests"]

        info: Dict = {
            # Historical keys (backward compatible)
            "model_name": server.model_name,
            "max_model_len": server.max_model_len,
            "backend": server.backend,
            "version": server.backend_version,
            "quantization": server.quantization,
            "tensor_parallel": server.tensor_parallel,
            "pipeline_parallel": server.pipeline_parallel,
            "max_num_seqs": server.max_num_seqs,
            "gpu_memory_utilization": None,
            "kv_cache_usage": raw.get("kv_cache_usage"),
            "prefix_caching": server.prefix_caching,
            "additional_info": additional_info,
            # Superset keys exposed by the normalized ServerInfo
            "backend_version": server.backend_version,
            "served_model_path": server.served_model_path,
            "kv_cache_dtype": server.kv_cache_dtype,
            "dtype": server.dtype,
            "expert_parallel": server.expert_parallel,
            "speculative": server.speculative,
            "task": server.task,
        }

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
