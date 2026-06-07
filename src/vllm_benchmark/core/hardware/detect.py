"""Hardware monitor detection.

Selects the appropriate :class:`HardwareMonitor` for the current host.
Only NVIDIA GPUs are supported for now.

Author: amit
License: MIT
"""

from __future__ import annotations

from vllm_benchmark.core.hardware.base import HardwareMonitor
from vllm_benchmark.core.hardware.nvidia import NvidiaMonitor


def get_hardware_monitor(poll_interval: float = 0.1) -> HardwareMonitor:
    """Return a hardware monitor for the current host.

    Args:
        poll_interval: Polling interval in seconds.

    Returns:
        A :class:`HardwareMonitor`.  Currently always an
        :class:`NvidiaMonitor` (NVIDIA-only support).
    """
    return NvidiaMonitor(poll_interval)
