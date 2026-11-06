"""Capacity & ceilings figure: measured throughput against the roofline.

Three panels, one job each (form follows the data's job):

1. **Attainment vs concurrency** — per context length, the measured
   aggregate decode throughput (markers) against the roofline ceiling
   ``N * BW / (W + N * KV * L)`` (thin line, same hue) with the compute cap
   drawn once as a neutral dashed line.  One y-axis; contexts are the only
   categorical series and keep a fixed hue order.
2. **Context slope** — single-stream ceiling and measurement across
   context lengths, so the physics (KV share of bytes) and the server's
   behaviour can be told apart.
3. **What-if by weight format** — horizontal bars of the projected
   single-stream ceiling per format, current format marked, unsupported
   formats shown hollow.

Palette and mark rules follow the validated categorical set (fixed order,
thin marks, direct labels for <= 4 series, legend always present for >= 2,
recessive grid, text in ink colours not series colours).

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Validated categorical slots (light surface), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e2"
NEUTRAL = "#8a8985"


def _fmt_ctx(ctx: int) -> str:
    return f"{ctx // 1000}K" if ctx >= 1000 else str(ctx)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def plot_capacity(capacity: Optional[dict], out_path: str, *, title: str = "Capacity & ceilings") -> str:
    """Render the capacity figure; always writes a PNG (placeholder when empty)."""
    capacity = capacity or {}
    ceilings = [c for c in (capacity.get("ceilings") or []) if c.get("decode_ceiling_tps")]
    slope = capacity.get("context_slope") or []
    what_if = capacity.get("what_if") or []

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    # ---- Panel 1: attainment vs concurrency ----
    ax = axes[0]
    _style(ax)
    contexts = sorted({c["context_length"] for c in ceilings})
    compute_cap = None
    for i, ctx in enumerate(contexts[:8]):
        color = SERIES[i]
        rows = sorted((c for c in ceilings if c["context_length"] == ctx), key=lambda c: c["concurrent_users"])
        xs = [r["concurrent_users"] for r in rows]
        ax.plot(xs, [r["decode_ceiling_tps"] for r in rows], color=color, linewidth=2, alpha=0.9, label=f"{_fmt_ctx(ctx)} ceiling")
        meas = [(r["concurrent_users"], r["decode_measured_tps"]) for r in rows if r.get("decode_measured_tps")]
        if meas:
            ax.plot([m[0] for m in meas], [m[1] for m in meas], linestyle="none", marker="o", markersize=7,
                    markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2, label=f"{_fmt_ctx(ctx)} measured")
            u, t = meas[-1]
            att = rows[-1].get("decode_attainment")
            if att:
                ax.annotate(f"{att:.0%}", (u, t), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK_2)
        for r in rows:
            if r.get("bound_by") == "compute" and compute_cap is None:
                compute_cap = r["decode_ceiling_tps"]
    if compute_cap:
        ax.axhline(compute_cap, color=NEUTRAL, linestyle="--", linewidth=1.5)
        ax.annotate("compute cap", (ax.get_xlim()[0], compute_cap), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color=INK_2)
    if ceilings:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel("Aggregate decode tok/s")
        ax.set_title("Measured vs roofline ceiling", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=1)
    else:
        ax.text(0.5, 0.5, "no ceilings (model or GPU spec unknown)", ha="center", va="center", color=INK_2, transform=ax.transAxes)
        ax.set_axis_off()

    # ---- Panel 2: context slope ----
    ax = axes[1]
    _style(ax)
    rows = [r for r in slope if r.get("ceiling_tps")]
    if rows:
        xs = [r["context_length"] for r in rows]
        ax.plot(xs, [r["ceiling_tps"] for r in rows], color=SERIES[0], linewidth=2, label="single-stream ceiling")
        meas = [(r["context_length"], r["measured_tps"]) for r in rows if r.get("measured_tps")]
        if meas:
            ax.plot([m[0] for m in meas], [m[1] for m in meas], color=SERIES[1], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label="measured (1 user)")
        for r in rows:
            share = r.get("kv_share_of_bytes")
            if share is not None:
                ax.annotate(f"KV {share:.0%}", (r["context_length"], r["ceiling_tps"]), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=INK_2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([_fmt_ctx(x) for x in xs])
        ax.set_xlabel("Context length")
        ax.set_ylabel("Decode tok/s, one stream")
        ax.set_title("Context slope (KV share of bytes annotated)", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    else:
        ax.text(0.5, 0.5, "no single-user cells", ha="center", va="center", color=INK_2, transform=ax.transAxes)
        ax.set_axis_off()

    # ---- Panel 3: what-if by format ----
    ax = axes[2]
    _style(ax)
    rows = [r for r in what_if if r.get("single_stream_ceiling_tps") or not r.get("supported")]
    if rows:
        labels = [r["format"] for r in rows]
        vals = [r.get("single_stream_ceiling_tps") or 0 for r in rows]
        ys = list(range(len(rows)))[::-1]
        for y, r, v in zip(ys, rows, vals):
            if not r.get("supported"):
                ax.barh(y, 0.0001, color="none", edgecolor=NEUTRAL, linewidth=1)
                ax.annotate("not supported on this GPU", (0, y), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK_2)
                continue
            color = SERIES[0] if r.get("current") else SERIES[2]
            ax.barh(y, v, color=color, height=0.55, edgecolor=SURFACE, linewidth=2)
            kv = r.get("kv_capacity_tokens_est")
            txt = f"{v:,.0f} tok/s" + (f"  ·  KV {kv / 1000:,.0f}K tok" if kv else "")
            ax.annotate(txt, (v, y), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK_2)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Projected single-stream decode tok/s at first context")
        ax.set_title("What-if by weight format (current in blue)", fontsize=11, loc="left")
        ax.set_xlim(0, max(vals + [1]) * 1.45)
        ax.grid(False, axis="y")
    else:
        ax.text(0.5, 0.5, "no what-if (model unknown)", ha="center", va="center", color=INK_2, transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
