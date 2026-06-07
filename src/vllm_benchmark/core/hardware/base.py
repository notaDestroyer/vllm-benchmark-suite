"""Hardware monitor abstraction.

Defines the :class:`HardwareMonitor` ABC implemented by vendor-specific
monitors (currently NVIDIA via nvidia-smi).

Author: amit
License: MIT
"""

from __future__ import annotations

import abc
from typing import Optional


class HardwareMonitor(abc.ABC):
    """Abstract base class for a background hardware monitor."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin background sampling."""
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> Optional[dict]:
        """Stop sampling and return aggregated statistics.

        Returns:
            A dict of aggregated metrics, or ``None`` if no data was
            collected.
        """
        raise NotImplementedError
