"""Hardware monitoring abstraction.

Exposes the :class:`HardwareMonitor` ABC, the NVIDIA implementation, and
the :func:`get_hardware_monitor` selector.

Author: amit
License: MIT
"""

from __future__ import annotations

from vllm_benchmark.core.hardware.base import HardwareMonitor
from vllm_benchmark.core.hardware.detect import get_hardware_monitor
from vllm_benchmark.core.hardware.gpu_derive import derive_local_gpu_spec, derive_spec, read_nvml_devices
from vllm_benchmark.core.hardware.gpu_names import match_gpu_key, normalize_gpu_name
from vllm_benchmark.core.hardware.nvidia import NvidiaMonitor

__all__ = [
    "HardwareMonitor",
    "NvidiaMonitor",
    "get_hardware_monitor",
    "derive_local_gpu_spec",
    "derive_spec",
    "read_nvml_devices",
    "match_gpu_key",
    "normalize_gpu_name",
]
