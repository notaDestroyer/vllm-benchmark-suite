"""Tests for the shareable result card PNG (``reports/card.py``).

Asserts the saved file exists, is a valid PNG with exact 1200x630
dimensions, and that a card with all fields missing still renders the
``N/A`` path without raising.

Author: amit
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from vllm_benchmark.reports.card import render_result_card

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class _FakeScore:
    overall: int = 7321
    grade: str = "A"


def _png_dimensions(path: str) -> tuple[int, int]:
    """Read width/height from a PNG IHDR chunk."""
    data = Path(path).read_bytes()
    assert data[:8] == _PNG_MAGIC
    # IHDR width/height are big-endian uint32 at byte offsets 16 and 20.
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _full_metadata() -> dict:
    return {
        "system_info": {"gpu_name": "H100", "total_vram_gb": 80.0},
        "server_info": {"model_name": "deepseek-ai/DeepSeek-V3", "quantization": "fp8"},
        "model_profile": {
            "name": "deepseek-ai/DeepSeek-V3",
            "is_moe": True,
            "active_params": 37_000_000_000,
            "total_params": 671_000_000_000,
        },
        "bottlenecks": [
            {
                "cell": [32000, 1, "classic"],
                "primary": "decode_weight_bandwidth",
                "mbu": 0.42,
                "mfu": 0.31,
                "lever": "increase memory bandwidth",
                "confidence": "high",
            }
        ],
        "quality": {"grade": "A", "score": 92},
    }


def _full_results() -> list[dict]:
    return [
        {
            "context_length": 32000,
            "concurrent_users": 1,
            "prompt_type": "classic",
            "tokens_per_second": 1820.0,
            "prefill_tps": 9200.0,
            "ttft_estimate": 0.085,
            "max_mem_used": 71000.0,
            "avg_power": 540.0,
        },
        {
            "context_length": 32000,
            "concurrent_users": 8,
            "prompt_type": "classic",
            "tokens_per_second": 5400.0,
            "ttft_estimate": 0.21,
            "max_mem_used": 73000.0,
            "avg_power": 610.0,
        },
    ]


def test_card_renders_valid_png_with_dimensions(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    path = render_result_card(_full_results(), _full_metadata(), _FakeScore(), str(out))
    assert Path(path).exists()
    assert Path(path).read_bytes()[:8] == _PNG_MAGIC
    assert _png_dimensions(path) == (1200, 630)


def test_card_renders_with_missing_fields(tmp_path: Path) -> None:
    out = tmp_path / "empty_card.png"
    # Empty results + empty metadata must exercise the N/A path, not raise.
    path = render_result_card([], {}, None, str(out))
    assert Path(path).exists()
    assert _png_dimensions(path) == (1200, 630)


def test_card_partial_metadata(tmp_path: Path) -> None:
    out = tmp_path / "partial.png"
    metadata = {"system_info": {"gpu_name": "RTX 4090"}}
    path = render_result_card(
        [{"context_length": 8000, "concurrent_users": 1, "tokens_per_second": 120.0}],
        metadata,
        None,
        str(out),
    )
    assert Path(path).exists()
    assert _png_dimensions(path) == (1200, 630)
