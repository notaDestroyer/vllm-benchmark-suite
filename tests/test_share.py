"""Tests for the share Markdown builder (``reports/share.py``).

Asserts the headline table row, the collapsible matrix and the advisor /
bottleneck verdict lines are present, and that the asserted body is
deterministic (no wall-clock) for a fixed input.

Author: amit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vllm_benchmark.reports.share import build_share_markdown, save_share_markdown


@dataclass
class _FakeScore:
    overall: int = 6800
    grade: str = "B"


def _metadata() -> dict:
    return {
        "system_info": {"gpu_name": "H100", "total_vram_gb": 80.0},
        "server_info": {"model_name": "meta-llama/Llama-3-8B", "quantization": "fp8"},
        "model_profile": {"name": "meta-llama/Llama-3-8B"},
        "bottlenecks": [
            {
                "cell": [32000, 1, "classic"],
                "primary": "decode_weight_bandwidth",
                "mbu": 0.45,
                "lever": "increase memory bandwidth (or enable FP8 KV cache)",
                "confidence": "high",
            }
        ],
        "advisory": {"fitness": {"verdict": "Good for interactive chat; marginal for RAG."}},
    }


def _results() -> list[dict]:
    return [
        {"context_length": 32000, "concurrent_users": 1, "prompt_type": "classic",
         "tokens_per_second": 200.0, "prefill_tps": 9000.0, "ttft_estimate": 0.08},
        {"context_length": 32000, "concurrent_users": 8, "prompt_type": "classic",
         "tokens_per_second": 1400.0, "ttft_estimate": 0.18},
        {"context_length": 64000, "concurrent_users": 1, "prompt_type": "classic",
         "tokens_per_second": 180.0, "ttft_estimate": 0.15},
        {"context_length": 64000, "concurrent_users": 8, "prompt_type": "classic",
         "tokens_per_second": 1200.0, "ttft_estimate": 0.30},
    ]


def test_headline_row_present() -> None:
    md = build_share_markdown(_results(), _metadata(), _FakeScore())
    assert "| GPU | Model | Quant | Peak tok/s | pp/tg | Best TTFT | MBU | Bottleneck | Score |" in md
    # Headline data row.
    assert "H100 (80GB)" in md
    assert "meta-llama/Llama-3-8B" in md
    assert "fp8" in md
    assert "1400" in md  # peak tok/s
    assert "decode_weight_bandwidth" in md
    assert "6,800 (B)" in md


def test_details_matrix_present() -> None:
    md = build_share_markdown(_results(), _metadata(), _FakeScore())
    assert "<details>" in md
    assert "<summary>Full throughput matrix (tok/s)</summary>" in md
    assert "| Context \\ Users | 1 | 8 |" in md
    assert "| 32K |" in md
    assert "| 64K |" in md
    assert "</details>" in md


def test_verdict_and_bottleneck_lines_present() -> None:
    md = build_share_markdown(_results(), _metadata(), _FakeScore())
    assert "**Advisor:** Good for interactive chat; marginal for RAG." in md
    assert "**Bottleneck:** decode_weight_bandwidth" in md
    assert "confidence high" in md


def test_deterministic_no_wallclock() -> None:
    a = build_share_markdown(_results(), _metadata(), _FakeScore())
    b = build_share_markdown(_results(), _metadata(), _FakeScore())
    assert a == b
    # Default body must not include a generated-at footer.
    assert "_Generated" not in a


def test_missing_fields_render_na() -> None:
    md = build_share_markdown([], {}, None)
    assert "N/A" in md
    assert "**Advisor:** N/A" in md
    assert "**Bottleneck:** N/A" in md


def test_save_share_markdown(tmp_path: Path) -> None:
    md = build_share_markdown(_results(), _metadata(), _FakeScore())
    path = save_share_markdown(md, str(tmp_path))
    assert Path(path).exists()
    assert Path(path).name.startswith("share_")
    assert Path(path).read_text(encoding="utf-8") == md
