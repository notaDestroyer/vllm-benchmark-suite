"""Hardware monitoring abstraction.

Exposes the :class:`HardwareMonitor` ABC, the NVIDIA implementation, and
the :func:`get_hardware_monitor` selector.

Author: amit
License: MIT
"""

from __future__ import annotations

from vllm_benchmark.core.hardware.base import HardwareMonitor
from vllm_benchmark.core.hardware.detect import get_hardware_monitor
from vllm_benchmark.core.hardware.nvidia import NvidiaMonitor

__all__ = [
    "HardwareMonitor",
    "NvidiaMonitor",
    "get_hardware_monitor",
]
