"""Shareable result card — a single 1200x630 PNG summary.

Renders a compact, social-media-friendly "result card" summarising one
benchmark run: the hardware, model, quantization, headline throughput,
best TTFT, peak VRAM, power, roofline utilization (MBU/MFU), the
governing bottleneck and the vLLM Score / quality grade.

The image is rendered **deterministically**: a fixed figure size and DPI
guarantee stable 1200x630 dimensions, and no timestamps or random values
are drawn into the image, so the same input always yields the same card.
Any missing field gracefully renders as ``"N/A"`` — the card never
raises on incomplete metadata.

Author: amit
License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed geometry so the saved PNG is exactly 1200x630 pixels.
_CARD_W_PX = 1200
_CARD_H_PX = 630
_CARD_DPI = 100

# Card palette (dark theme, matches the HTML report).
_BG = "#0f0f1a"
_PANEL = "#1a1a2e"
_ACCENT = "#64b5f6"
_TEXT = "#e0e0e0"
_MUTED = "#9aa0b0"

_GRADE_COLORS = {
    "S": "#e040fb", "A": "#4caf50", "B": "#8bc34a",
    "C": "#ffc107", "D": "#ff5722", "F": "#d32f2f",
}


# ---------------------------------------------------------------------------
# Field extraction helpers (pure, None-safe)
# ---------------------------------------------------------------------------

def _fmt(value: Optional[Any], suffix: str = "", fmt: str = "{}") -> str:
    """Format a value with an optional suffix, or ``"N/A"`` when missing."""
    if value is None:
        return "N/A"
    try:
        text = fmt.format(value)
    except (ValueError, TypeError):
        return "N/A"
    return f"{text}{suffix}"


def _gpu_label(system_info: dict) -> str:
    """Hardware label: ``GPU (VRAM GB)`` or ``"N/A"``."""
    name = system_info.get("gpu_name")
    vram = system_info.get("total_vram_gb")
    if not name:
        return "N/A"
    if vram:
        return f"{name} ({float(vram):.0f} GB)"
    return str(name)


def _model_label(model_profile: Optional[dict], server_info: dict) -> str:
    """Model name from the profile or server info."""
    if model_profile and model_profile.get("name"):
        return str(model_profile["name"])
    return str(server_info.get("model_name") or "N/A")


def _params_label(model_profile: Optional[dict]) -> str:
    """MoE/dense + active/total params, e.g. ``MoE 37B/671B`` or ``dense 8B``."""
    if not model_profile:
        return "N/A"
    is_moe = model_profile.get("is_moe")
    active = model_profile.get("active_params")
    total = model_profile.get("total_params")
    kind = "MoE" if is_moe else "dense" if is_moe is not None else ""

    def _b(n: Optional[int]) -> Optional[str]:
        if not n:
            return None
        return f"{n / 1e9:.0f}B"

    a, t = _b(active), _b(total)
    if a and t and a != t:
        params = f"{a}/{t}"
    elif t:
        params = t
    elif a:
        params = a
    else:
        params = ""
    parts = [p for p in (kind, params) if p]
    return " ".join(parts) if parts else "N/A"


def _top_verdict(metadata: dict) -> Optional[dict]:
    """Pick the most-trustworthy bottleneck verdict (prefer high confidence)."""
    verdicts = metadata.get("bottlenecks") or []
    if not verdicts:
        return None
    order = {"high": 0, "medium": 1, "low": 2}
    return min(verdicts, key=lambda v: order.get(v.get("confidence"), 3))


def _headline_throughput(results: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """Best prefill (pp) and decode/aggregate (tg) throughput across cells."""
    pp_vals = [
        r.get("prefill_tps_mean") or r.get("prefill_tps") or r.get("prefill_tps_p50")
        for r in results
    ]
    pp_vals = [v for v in pp_vals if v]
    tg_vals = [r.get("tokens_per_second") for r in results if r.get("tokens_per_second")]
    pp = max(pp_vals) if pp_vals else None
    tg = max(tg_vals) if tg_vals else None
    return pp, tg


def _best_ttft(results: list[dict]) -> Optional[float]:
    """Lowest TTFT estimate (seconds) across cells."""
    vals = [r.get("ttft_estimate") for r in results if r.get("ttft_estimate")]
    return min(vals) if vals else None


def _peak_vram(results: list[dict]) -> Optional[float]:
    """Peak VRAM used across cells (max of max/avg mem)."""
    vals = [
        (r.get("max_mem_used") or r.get("avg_mem_used"))
        for r in results
    ]
    vals = [v for v in vals if v]
    return max(vals) if vals else None


def _avg_power(results: list[dict]) -> Optional[float]:
    """Mean of per-cell average power draw."""
    vals = [r.get("avg_power") for r in results if r.get("avg_power")]
    return (sum(vals) / len(vals)) if vals else None


def _quality_grade(metadata: dict) -> Optional[str]:
    """Quality grade/score label from the quality section, if present."""
    quality = metadata.get("quality")
    if not isinstance(quality, dict):
        return None
    grade = quality.get("grade")
    if grade:
        return str(grade)
    score = quality.get("score")
    if score is not None:
        return f"{score}/100"
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_result_card(
    results: list[dict],
    metadata: dict,
    score: Any = None,
    out_path: str | None = None,
) -> str:
    """Render a 1200x630 PNG result card and return the saved path.

    The card is deterministic: fixed figure size/DPI yield stable pixel
    dimensions and no wall-clock or random content is drawn.  Missing
    fields render as ``"N/A"`` rather than raising.

    Args:
        results: Per-cell result dicts from a run.
        metadata: Run metadata (``system_info``, ``server_info``,
            ``model_profile``, ``bottlenecks``, ``quality`` ...).
        score: Optional score object exposing ``.overall`` and ``.grade``
            (e.g. a :class:`ScoreBreakdown`), or ``None``.
        out_path: Destination PNG path.  Defaults to
            ``./result_card.png`` in the current directory.

    Returns:
        The path the PNG was written to.
    """
    results = results or []
    metadata = metadata or {}
    system_info = metadata.get("system_info") or {}
    server_info = metadata.get("server_info") or {}
    model_profile = metadata.get("model_profile")

    verdict = _top_verdict(metadata)
    pp, tg = _headline_throughput(results)
    quant = server_info.get("quantization") or "none"

    # --- assemble display fields ---
    gpu = _gpu_label(system_info)
    model = _model_label(model_profile, server_info)
    params = _params_label(model_profile)
    ttft = _best_ttft(results)
    vram = _peak_vram(results)
    power = _avg_power(results)
    mbu = verdict.get("mbu") if verdict else None
    mfu = verdict.get("mfu") if verdict else None
    primary = verdict.get("primary") if verdict else None
    lever = verdict.get("lever") if verdict else None
    quality = _quality_grade(metadata)

    # --- figure ---
    fig = plt.figure(figsize=(_CARD_W_PX / _CARD_DPI, _CARD_H_PX / _CARD_DPI), dpi=_CARD_DPI)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(_BG)

    # Header band.
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color=_PANEL))
    ax.text(0.03, 0.93, "vLLM Benchmark", color=_ACCENT, fontsize=22,
            fontweight="bold", va="center", ha="left")
    ax.text(0.03, 0.885, gpu, color=_MUTED, fontsize=12, va="center", ha="left")

    # Score / grade badge (top-right).
    if score is not None and getattr(score, "overall", None) is not None:
        grade = getattr(score, "grade", "?")
        gc = _GRADE_COLORS.get(grade, _ACCENT)
        ax.text(0.97, 0.95, f"{int(score.overall):,}", color=gc, fontsize=30,
                fontweight="bold", va="center", ha="right")
        ax.text(0.97, 0.885, f"Grade {grade}", color=gc, fontsize=14,
                va="center", ha="right")

    # Model line.
    ax.text(0.03, 0.79, model, color=_TEXT, fontsize=18, fontweight="bold",
            va="center", ha="left")
    ax.text(0.03, 0.74, f"{params}  |  quant: {quant}", color=_MUTED, fontsize=12,
            va="center", ha="left")

    # Headline metric tiles (2 rows x 3 cols).
    tiles: list[tuple[str, str]] = [
        ("Prefill (pp)", _fmt(pp, " tok/s", "{:.0f}")),
        ("Decode/agg (tg)", _fmt(tg, " tok/s", "{:.0f}")),
        ("Best TTFT", _fmt(ttft * 1000 if ttft else None, " ms", "{:.0f}")),
        ("Peak VRAM", _fmt(vram, " MB", "{:.0f}")),
        ("Avg power", _fmt(power, " W", "{:.0f}")),
        ("MBU / MFU", f"{_fmt(mbu, '', '{:.0%}')} / {_fmt(mfu, '', '{:.0%}')}"),
    ]
    cols, rows = 3, 2
    x0, y0 = 0.03, 0.30
    tw, th = 0.30, 0.16
    gap_x, gap_y = 0.035, 0.04
    for idx, (label, value) in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        tx = x0 + c * (tw + gap_x)
        ty = y0 + (rows - 1 - r) * (th + gap_y)
        ax.add_patch(plt.Rectangle((tx, ty), tw, th, color=_PANEL, zorder=1))
        ax.text(tx + 0.015, ty + th - 0.045, label, color=_MUTED, fontsize=11,
                va="center", ha="left", zorder=2)
        ax.text(tx + 0.015, ty + 0.045, value, color=_TEXT, fontsize=18,
                fontweight="bold", va="center", ha="left", zorder=2)

    # Footer: governing bottleneck + quality grade.
    bottleneck_text = "N/A"
    if primary:
        bottleneck_text = primary
        if lever:
            bottleneck_text += f"  ->  {lever}"
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.10, color=_PANEL))
    ax.text(0.03, 0.06, "Bottleneck", color=_MUTED, fontsize=11, va="center", ha="left")
    ax.text(0.03, 0.025, bottleneck_text, color=_TEXT, fontsize=13, va="center", ha="left")
    ax.text(0.97, 0.05, f"Quality: {quality or 'N/A'}", color=_ACCENT, fontsize=13,
            va="center", ha="right")

    # Save deterministically.
    out = Path(out_path) if out_path else Path("./result_card.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_CARD_DPI, facecolor=_BG)
    plt.close(fig)
    return str(out)
