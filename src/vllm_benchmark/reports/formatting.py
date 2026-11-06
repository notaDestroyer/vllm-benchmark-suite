"""Small formatting helpers shared by the report renderers.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import Any, Optional


def na(value: Optional[Any], suffix: str = "", fmt: str = "{}") -> str:
    """Format ``value`` with ``fmt`` and ``suffix``, or ``"N/A"`` when missing.

    ``None`` and values that do not fit the format spec (e.g. a string
    where a float format is requested) both render as ``"N/A"`` so a
    partial result never breaks a report.
    """
    if value is None:
        return "N/A"
    try:
        return f"{fmt.format(value)}{suffix}"
    except (ValueError, TypeError):
        return "N/A"
