"""Copy-paste shareable Markdown summary for a benchmark run.

Builds a Reddit/forum-friendly Markdown block: a one-row headline table,
a collapsible ``<details>`` block holding the full context x concurrency
throughput matrix, the advisor's one-line fitness verdict and a one-line
governing-bottleneck summary.

The asserted body is **deterministic** for a fixed input — no wall-clock
is embedded except an optional footer (``include_footer``) which is off by
default so tests can assert the exact text.

Author: amit
License: MIT
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Extraction helpers (pure, None-safe)
# ---------------------------------------------------------------------------

def _na(value: Optional[Any], suffix: str = "", fmt: str = "{}") -> str:
    """Format a value with a suffix, or ``"N/A"`` when missing."""
    if value is None:
        return "N/A"
    try:
        return f"{fmt.format(value)}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def _gpu_label(metadata: dict) -> str:
    """Hardware label for the headline row."""
    system_info = metadata.get("system_info") or {}
    name = system_info.get("gpu_name")
    vram = system_info.get("total_vram_gb")
    if not name:
        return "N/A"
    if vram:
        return f"{name} ({float(vram):.0f}GB)"
    return str(name)


def _model_label(metadata: dict) -> str:
    """Model name for the headline row."""
    profile = metadata.get("model_profile") or {}
    if profile.get("name"):
        return str(profile["name"])
    server = metadata.get("server_info") or {}
    return str(server.get("model_name") or "N/A")


def _quant_label(metadata: dict) -> str:
    """Quantization label."""
    server = metadata.get("server_info") or {}
    return str(server.get("quantization") or "none")


def _top_verdict(metadata: dict) -> Optional[dict]:
    """Pick the most-trustworthy bottleneck verdict (prefer high confidence)."""
    verdicts = metadata.get("bottlenecks") or []
    if not verdicts:
        return None
    order = {"high": 0, "medium": 1, "low": 2}
    return min(verdicts, key=lambda v: order.get(v.get("confidence"), 3))


def _peak_tps(results: list[dict]) -> Optional[float]:
    """Peak aggregate throughput across cells."""
    vals = [r.get("tokens_per_second") for r in results if r.get("tokens_per_second")]
    return max(vals) if vals else None


def _best_pp_tg(results: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """Best prefill (pp) and decode/aggregate (tg) throughput."""
    pp_vals = [
        r.get("prefill_tps_mean") or r.get("prefill_tps") or r.get("prefill_tps_p50")
        for r in results
    ]
    pp_vals = [v for v in pp_vals if v]
    tg_vals = [r.get("tokens_per_second") for r in results if r.get("tokens_per_second")]
    pp = max(pp_vals) if pp_vals else None
    tg = max(tg_vals) if tg_vals else None
    return pp, tg


def _best_ttft_ms(results: list[dict]) -> Optional[float]:
    """Lowest TTFT estimate in milliseconds."""
    vals = [r.get("ttft_estimate") for r in results if r.get("ttft_estimate")]
    return (min(vals) * 1000.0) if vals else None


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------

def _headline_table(results: list[dict], metadata: dict, score: Any) -> str:
    """Build the single-row headline Markdown table."""
    pp, tg = _best_pp_tg(results)
    verdict = _top_verdict(metadata)
    mbu = verdict.get("mbu") if verdict else None
    bottleneck = verdict.get("primary") if verdict else None
    score_text = "N/A"
    if score is not None and getattr(score, "overall", None) is not None:
        score_text = f"{int(score.overall):,} ({getattr(score, 'grade', '?')})"

    header = (
        "| GPU | Model | Quant | Peak tok/s | pp/tg | Best TTFT | MBU | Bottleneck | Score |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    row = "| {gpu} | {model} | {quant} | {peak} | {pptg} | {ttft} | {mbu} | {bn} | {score} |\n".format(
        gpu=_gpu_label(metadata),
        model=_model_label(metadata),
        quant=_quant_label(metadata),
        peak=_na(_peak_tps(results), "", "{:.0f}"),
        pptg=f"{_na(pp, '', '{:.0f}')}/{_na(tg, '', '{:.0f}')}",
        ttft=_na(_best_ttft_ms(results), "ms", "{:.0f}"),
        mbu=_na(mbu, "", "{:.0%}"),
        bn=bottleneck or "N/A",
        score=score_text,
    )
    return header + row


def _matrix_block(results: list[dict]) -> str:
    """Build the collapsible context x concurrency throughput matrix."""
    contexts = sorted({r.get("context_length") for r in results if r.get("context_length") is not None})
    concurrencies = sorted({
        r.get("concurrent_users") for r in results if r.get("concurrent_users") is not None
    })
    # Average tok/s over any extra dims (e.g. prompt_type) per (ctx, conc).
    cell_vals: dict[tuple, list[float]] = {}
    for r in results:
        ctx = r.get("context_length")
        conc = r.get("concurrent_users")
        tps = r.get("tokens_per_second")
        if ctx is None or conc is None or tps is None:
            continue
        cell_vals.setdefault((ctx, conc), []).append(float(tps))

    lines = ["<details>", "<summary>Full throughput matrix (tok/s)</summary>", ""]
    if not contexts or not concurrencies:
        lines.append("_No matrix data available._")
        lines += ["", "</details>"]
        return "\n".join(lines)

    header = "| Context \\ Users | " + " | ".join(str(c) for c in concurrencies) + " |"
    sep = "|---|" + "|".join("---" for _ in concurrencies) + "|"
    lines += [header, sep]
    for ctx in contexts:
        ctx_label = f"{ctx // 1000}K" if ctx >= 1000 else str(ctx)
        cells = []
        for conc in concurrencies:
            vals = cell_vals.get((ctx, conc))
            cells.append(f"{sum(vals) / len(vals):.0f}" if vals else "N/A")
        lines.append(f"| {ctx_label} | " + " | ".join(cells) + " |")
    lines += ["", "</details>"]
    return "\n".join(lines)


def build_share_markdown(
    results: list[dict],
    metadata: dict,
    score: Any = None,
    *,
    include_footer: bool = False,
) -> str:
    """Build a copy-paste Markdown summary for a benchmark run.

    Args:
        results: Per-cell result dicts from the run.
        metadata: Run metadata (``system_info``, ``server_info``,
            ``model_profile``, ``bottlenecks``, ``advisory`` ...).
        score: Optional score object exposing ``.overall`` / ``.grade``.
        include_footer: When ``True`` append a wall-clock footer line.
            Off by default so the asserted body is deterministic.

    Returns:
        A Markdown string: headline table, collapsible matrix, advisor
        verdict and a one-line bottleneck summary.
    """
    results = results or []
    metadata = metadata or {}

    parts: list[str] = []
    parts.append(f"## vLLM Benchmark — {_model_label(metadata)}")
    parts.append("")
    parts.append(_headline_table(results, metadata, score))
    parts.append(_matrix_block(results))
    parts.append("")

    # Advisor one-line verdict.
    advisory = metadata.get("advisory") or {}
    fitness = advisory.get("fitness") or {}
    verdict_line = fitness.get("verdict")
    parts.append(f"**Advisor:** {verdict_line or 'N/A'}")

    # One-line bottleneck summary.
    top = _top_verdict(metadata)
    if top:
        lever = top.get("lever")
        conf = top.get("confidence")
        bn = f"**Bottleneck:** {top.get('primary', 'unknown')}"
        if conf:
            bn += f" (confidence {conf})"
        if lever:
            bn += f" — {lever}"
        parts.append(bn)
    else:
        parts.append("**Bottleneck:** N/A")

    body = "\n".join(parts)
    if include_footer:
        body += f"\n\n_Generated {datetime.now().isoformat(timespec='seconds')}_"
    return body


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_share_markdown(md: str, out_dir: str) -> str:
    """Write the share Markdown to ``share_<timestamp>.md`` in *out_dir*.

    Args:
        md: The Markdown body to write.
        out_dir: Destination directory (created if needed).

    Returns:
        The path the file was written to.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"share_{timestamp}.md"
    path.write_text(md, encoding="utf-8")
    return str(path)
