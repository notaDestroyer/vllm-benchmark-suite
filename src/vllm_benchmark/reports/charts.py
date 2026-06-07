"""Matplotlib chart generation — 5 essential benchmark visualizations.

Focused on the charts that actually matter: throughput scaling,
latency distribution, TTFT, throughput vs context, and GPU utilization.

Author: amit
License: MIT
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    safe = name.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^\w\-.]", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    return safe[:100]


def ensure_output_directory(output_dir: str = "./outputs") -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------
# Main chart generator — 5 essential charts
# ------------------------------------------------------------------

def visualize_results(
    all_results: List[Dict],
    model_name: str,
    system_info: Dict = None,
    server_info: Dict = None,
    output_tokens: int = 500,
    output_dir: str = "./outputs",
) -> str:
    """Generate 5 essential benchmark charts as a single PNG.

    Charts:
      1. Throughput vs Context Length by Concurrency (line)
      2. Latency Distribution (box plot + P99 overlay)
      3. TTFT Distribution (with UX quality zones)
      4. Throughput Heatmap (context x concurrency)
      5. GPU Utilization Timeline (if available)

    Returns:
        Path to the saved PNG file.
    """
    df = pd.DataFrame(all_results)

    has_prompt_types = "prompt_type" in df.columns
    if has_prompt_types:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        preserve_cols = ["context_length", "concurrent_users"]
        avg_cols = [c for c in numeric_cols if c not in preserve_cols]
        agg_dict = {col: "mean" for col in avg_cols}
        df_main = df.groupby(["context_length", "concurrent_users"], as_index=False).agg(agg_dict)
    else:
        df_main = df.copy()

    has_gpu = "avg_gpu_util" in df_main.columns
    context_lengths = sorted(df_main["context_length"].unique())
    context_labels = [f"{int(c / 1000)}K" for c in context_lengths]
    concurrent_users = sorted(df_main["concurrent_users"].unique())
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.titlesize": 13,
    })

    num_rows = 3 if has_gpu else 2
    fig = plt.figure(figsize=(22, 6 * num_rows + 2))
    gs = fig.add_gridspec(num_rows, 2, hspace=0.35, wspace=0.25, left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ---- 1. Throughput vs Context Length ----
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, users in enumerate(concurrent_users):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax1.plot(d["context_length"] / 1000, d["tokens_per_second"], marker="o", linewidth=2.5, markersize=8,
                 label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax1.set_xlabel("Context Length (K tokens)")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_title("Throughput vs Context Length", fontweight="bold", pad=10)
    ax1.set_xticks([c / 1000 for c in context_lengths])
    ax1.set_xticklabels(context_labels)
    ax1.legend(title="Users", loc="best", frameon=True)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#FAFAFA")

    # ---- 2. Latency Distribution (box plot) ----
    ax2 = fig.add_subplot(gs[0, 1])
    latency_data = []
    latency_labels = []
    for users in concurrent_users:
        vals = df_main[df_main["concurrent_users"] == users]["avg_latency"].values
        if len(vals) > 0:
            latency_data.append(vals)
            latency_labels.append(f"{users}u")

    if latency_data:
        bp = ax2.boxplot(latency_data, labels=latency_labels, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)

        # P99 overlay
        if "latency_p99" in df_main.columns:
            p99_by_users = df_main.groupby("concurrent_users")["latency_p99"].max()
            for i, users in enumerate(concurrent_users):
                if users in p99_by_users.index:
                    ax2.scatter(i + 1, p99_by_users[users], color="red", marker="x", s=100, zorder=5, label="P99" if i == 0 else "")

    ax2.set_xlabel("Concurrent Users")
    ax2.set_ylabel("Latency (seconds)")
    ax2.set_title("Latency Distribution by Concurrency", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax2.set_facecolor("#FAFAFA")
    if "latency_p99" in df_main.columns:
        ax2.legend(loc="upper left")

    # ---- 3. TTFT with UX Quality Zones ----
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, users in enumerate(concurrent_users):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax3.plot(d["context_length"] / 1000, d["ttft_estimate"] * 1000, marker="*", linewidth=2.5, markersize=10,
                 label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)

    ax3.axhspan(0, 200, alpha=0.08, color="green", label="Excellent (<200ms)")
    ax3.axhspan(200, 1000, alpha=0.08, color="yellow", label="Acceptable")
    ax3.axhspan(1000, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 1000 else 3000, alpha=0.08, color="red", label="Poor (>1s)")
    ax3.set_xlabel("Context Length (K tokens)")
    ax3.set_ylabel("TTFT (ms)")
    ax3.set_title("Time to First Token (UX Quality)", fontweight="bold", pad=10)
    ax3.set_xticks([c / 1000 for c in context_lengths])
    ax3.set_xticklabels(context_labels)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_facecolor("#FAFAFA")

    # ---- 4. Throughput Heatmap ----
    ax4 = fig.add_subplot(gs[1, 1])
    pivot_tp = df_main.pivot(index="context_length", columns="concurrent_users", values="tokens_per_second")
    sns.heatmap(pivot_tp, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax4,
                cbar_kws={"label": "tok/s"}, linewidths=1.5, linecolor="white",
                annot_kws={"fontsize": 10, "weight": "bold"})
    ax4.set_xlabel("Concurrent Users")
    ax4.set_ylabel("Context Length")
    ax4.set_title("Throughput Heatmap", fontweight="bold", pad=10)
    ax4.set_yticklabels([f"{int(y / 1000)}K" for y in pivot_tp.index], rotation=0)

    # ---- 5. GPU Utilization (if available) ----
    if has_gpu:
        ax5 = fig.add_subplot(gs[2, :])
        x_positions = range(len(df_main))
        x_labels = [f"{int(r['context_length'] / 1000)}K\n{int(r['concurrent_users'])}u" for _, r in df_main.iterrows()]

        ax5_twin = ax5.twinx()

        ax5.bar(x_positions, df_main["avg_gpu_util"], color="#2E86AB", alpha=0.7, label="GPU Util %", width=0.4, align="edge")
        ax5_twin.plot(x_positions, df_main["avg_power"], color="#C73E1D", marker="o", linewidth=2, label="Power (W)")

        ax5.set_xlabel("Test Configuration (Context / Users)")
        ax5.set_ylabel("GPU Utilization (%)", color="#2E86AB")
        ax5_twin.set_ylabel("Power Draw (W)", color="#C73E1D")
        ax5.set_title("GPU Utilization & Power Across Tests", fontweight="bold", pad=10)
        ax5.set_xticks(x_positions)
        ax5.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax5.set_ylim(0, 105)
        ax5.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax5.set_facecolor("#FAFAFA")

        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Suptitle
    gpu_info = []
    if system_info:
        if system_info.get("gpu_name"):
            vram = system_info.get("total_vram_gb")
            gpu_info.append(f"{system_info['gpu_name']} ({vram:.0f}GB)" if vram else system_info["gpu_name"])
    if server_info and server_info.get("version"):
        gpu_info.append(f"vLLM {server_info['version']}")
    subtitle = " | ".join(gpu_info) if gpu_info else ""
    fig.suptitle(f"{model_name}\n{subtitle}", fontsize=13, fontweight="bold", y=0.99)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = sanitize_filename(model_name)
    out = ensure_output_directory(output_dir)
    filepath = out / f"benchmark_{safe}_{ts}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(filepath)


# ------------------------------------------------------------------
# PR5 — roofline, bottleneck map, quant comparison
# ------------------------------------------------------------------

#: Stable color per governing-bottleneck primary class.
_BOTTLENECK_COLORS: Dict[str, str] = {
    "prefill_compute": "#C73E1D",
    "decode_weight_bandwidth": "#2E86AB",
    "decode_kv_bandwidth": "#6A994E",
    "decode_compute": "#F18F01",
    "kv_capacity": "#A23B72",
    "queue": "#BC4B51",
    "interconnect": "#9467BD",
    "unknown": "#888888",
}


def plot_roofline(
    results: List[Dict],
    model_profile: Optional[Dict],
    gpu_spec: Optional[Dict],
    out_path: str,
) -> str:
    """Log-log roofline plot with the ridge point and measured points.

    Plots the hardware roofline (bandwidth-bound diagonal + compute-bound
    ceiling) in (arithmetic intensity, throughput) space, marks the ridge
    point B*, and overlays measured prefill (compute) and decode
    (bandwidth) operating points derived from the run.

    Missing/empty data is handled gracefully: the function still saves a
    placeholder PNG (and never raises).

    Args:
        results: Per-cell result dicts.
        model_profile: ``ModelProfile.to_dict()`` (or ``None``).
        gpu_spec: GPU spec dict with ``hbm_bandwidth_gbps`` /
            ``peak_flops_tflops`` (or ``None``).
        out_path: Destination PNG path.

    Returns:
        The path the PNG was written to.
    """
    results = results or []
    profile = model_profile or {}
    spec = gpu_spec or {}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.set_title("Roofline", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    hbm = spec.get("hbm_bandwidth_gbps")
    peak_map = spec.get("peak_flops_tflops") or {}
    peak_flops = peak_map.get("bf16") or peak_map.get("fp8")

    if hbm and peak_flops and hbm > 0 and peak_flops > 0:
        # Ridge point: where bandwidth*intensity == peak compute.
        # bandwidth in TB/s = GB/s / 1000; intensity at ridge = peak / bw.
        bw_tbs = hbm / 1000.0
        ridge_oi = peak_flops / bw_tbs
        oi = np.logspace(np.log10(ridge_oi / 100), np.log10(ridge_oi * 100), 200)
        roof = np.minimum(bw_tbs * oi, peak_flops)
        ax.plot(oi, roof, color="#333333", linewidth=2, label="Roofline")
        ax.axvline(ridge_oi, color="#A23B72", linestyle=":", linewidth=1.5,
                   label=f"Ridge (B*~{ridge_oi:.0f})")
        ax.scatter([ridge_oi], [peak_flops], color="#A23B72", zorder=5, s=60)

        # Measured operating points (best-effort, derived from throughput).
        active = profile.get("active_params")
        if active:
            pp_vals = [
                r.get("prefill_tps_mean") or r.get("prefill_tps") or r.get("prefill_tps_p50")
                for r in results
            ]
            pp = max((v for v in pp_vals if v), default=None)
            tg = max((r.get("tokens_per_second") for r in results if r.get("tokens_per_second")),
                     default=None)
            if pp:
                # prefill achieved TFLOP/s = 2*active*pp; high intensity (compute).
                achieved = 2 * active * pp / 1e12
                ax.scatter([ridge_oi * 4], [achieved], color="#C73E1D", marker="^",
                           s=90, zorder=6, label="Prefill (compute)")
            if tg:
                achieved = 2 * active * tg / 1e12
                ax.scatter([ridge_oi / 8], [achieved], color="#2E86AB", marker="o",
                           s=90, zorder=6, label="Decode (bandwidth)")
        ax.legend(loc="lower right", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient GPU/model data for roofline",
                transform=ax.transAxes, ha="center", va="center", color="#888888")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def bottleneck_grid(
    bottlenecks: List[Dict],
) -> tuple[List[int], List[int], List[List[Optional[str]]]]:
    """Grid the governing bottleneck per (context, concurrency) cell.

    Pure helper (no rendering) so the gridding/coloring logic is unit-
    testable.  When multiple verdicts share a (context, concurrency) cell
    (e.g. several prompt types) the most-frequent ``primary`` wins; ties
    break by the precedence order of :data:`_BOTTLENECK_COLORS`.

    Args:
        bottlenecks: List of ``BottleneckVerdict.to_dict()`` dicts, each
            carrying ``cell`` = ``[context, concurrency, prompt_type]`` and
            ``primary``.

    Returns:
        ``(contexts, concurrencies, primary_grid)`` where ``primary_grid``
        is indexed ``[row=context][col=concurrency]`` and holds the primary
        class string (or ``None`` for an empty cell).
    """
    bottlenecks = bottlenecks or []
    contexts = sorted({
        v["cell"][0] for v in bottlenecks
        if v.get("cell") and v["cell"][0] is not None
    })
    concurrencies = sorted({
        v["cell"][1] for v in bottlenecks
        if v.get("cell") and len(v["cell"]) > 1 and v["cell"][1] is not None
    })

    precedence = list(_BOTTLENECK_COLORS)

    # Collect primaries per (ctx, conc).
    buckets: Dict[tuple, List[str]] = {}
    for v in bottlenecks:
        cell = v.get("cell")
        if not cell or len(cell) < 2:
            continue
        ctx, conc = cell[0], cell[1]
        if ctx is None or conc is None:
            continue
        buckets.setdefault((ctx, conc), []).append(v.get("primary") or "unknown")

    grid: List[List[Optional[str]]] = []
    for ctx in contexts:
        row: List[Optional[str]] = []
        for conc in concurrencies:
            primaries = buckets.get((ctx, conc))
            if not primaries:
                row.append(None)
                continue
            # Most frequent; tie broken by precedence order.
            counts: Dict[str, int] = {}
            for p in primaries:
                counts[p] = counts.get(p, 0) + 1
            best = max(
                counts,
                key=lambda p: (counts[p], -precedence.index(p) if p in precedence else -len(precedence)),
            )
            row.append(best)
        grid.append(row)
    return contexts, concurrencies, grid


def plot_bottleneck_map(
    bottlenecks: List[Dict],
    out_path: str,
) -> str:
    """Render the context x concurrency governing-bottleneck map.

    Each cell is colored by its governing bottleneck class; the critical
    batch ``B*`` (from any verdict that carries it) is annotated as a
    vertical crossover boundary between bandwidth- and compute-bound
    concurrencies.  Empty/missing input is handled without raising.

    Args:
        bottlenecks: List of ``BottleneckVerdict.to_dict()`` dicts.
        out_path: Destination PNG path.

    Returns:
        The path the PNG was written to.
    """
    contexts, concurrencies, grid = bottleneck_grid(bottlenecks)

    fig, ax = plt.subplots(figsize=(max(6, len(concurrencies) * 1.2 + 2),
                                    max(4, len(contexts) * 0.9 + 2)))
    ax.set_title("Governing bottleneck map", fontweight="bold")

    if not contexts or not concurrencies:
        ax.text(0.5, 0.5, "No bottleneck data", transform=ax.transAxes,
                ha="center", va="center", color="#888888")
        ax.axis("off")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(out)

    classes = [c for c in _BOTTLENECK_COLORS if any(
        c in row for row in grid
    )]
    class_idx = {c: i for i, c in enumerate(classes)}

    # Build an integer matrix (-1 for empty) for imshow with a discrete cmap.
    z = np.full((len(contexts), len(concurrencies)), -1, dtype=float)
    for r, row in enumerate(grid):
        for c, primary in enumerate(row):
            if primary is not None:
                z[r, c] = class_idx[primary]

    for r in range(len(contexts)):
        for c in range(len(concurrencies)):
            primary = grid[r][c]
            color = _BOTTLENECK_COLORS.get(primary, "#202030") if primary else "#202030"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=color, ec="white", lw=1.5))
            if primary:
                ax.text(c + 0.5, r + 0.5, primary.replace("decode_", "d.").replace("_", " "),
                        ha="center", va="center", fontsize=8, color="white")

    ax.set_xlim(0, len(concurrencies))
    ax.set_ylim(0, len(contexts))
    ax.set_xticks([i + 0.5 for i in range(len(concurrencies))])
    ax.set_xticklabels([str(u) for u in concurrencies])
    ax.set_yticks([i + 0.5 for i in range(len(contexts))])
    ax.set_yticklabels([f"{c // 1000}K" if c >= 1000 else str(c) for c in contexts])
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Context length")
    ax.invert_yaxis()

    # Annotate the critical batch B* crossover (first verdict that has it).
    bstar = next(
        (v.get("critical_batch") for v in (bottlenecks or []) if v.get("critical_batch")),
        None,
    )
    if bstar is not None:
        # Find the column boundary just past B*.
        boundary = sum(1 for u in concurrencies if u < bstar)
        if 0 < boundary < len(concurrencies):
            ax.axvline(boundary, color="yellow", linestyle="--", linewidth=2)
            ax.text(boundary, -0.15, f"B*={bstar}", color="black", fontsize=9,
                    ha="center", va="top")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def plot_quant_compare(
    comparison: Dict,
    out_path: str,
) -> str:
    """Render a quant-comparison chart: tok/s vs VRAM vs quality per run.

    Consumes the output of
    :func:`vllm_benchmark.analysis.quant_compare.compare_quant_runs`.
    Draws grouped bars of mean throughput and peak VRAM per run with a
    quality overlay.  Empty/missing data is handled without raising.

    Args:
        comparison: The ``compare_quant_runs`` result dict.
        out_path: Destination PNG path.

    Returns:
        The path the PNG was written to.
    """
    comparison = comparison or {}
    runs = comparison.get("runs") or []

    fig, ax = plt.subplots(figsize=(max(7, len(runs) * 1.5 + 2), 5))
    ax.set_title("Quantization comparison", fontweight="bold")

    if not runs:
        ax.text(0.5, 0.5, "No comparison data", transform=ax.transAxes,
                ha="center", va="center", color="#888888")
        ax.axis("off")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(out)

    labels = [r.get("label", "?") for r in runs]
    tps = [r.get("tokens_per_second_mean") or 0.0 for r in runs]
    vram = [r.get("peak_mem_mean") or 0.0 for r in runs]
    quality = [r.get("quality") for r in runs]

    x = np.arange(len(runs))
    width = 0.35
    ax.bar(x - width / 2, tps, width, label="tok/s", color="#2E86AB")
    ax.set_ylabel("Throughput (tok/s)", color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    ax_v = ax.twinx()
    ax_v.bar(x + width / 2, vram, width, label="peak VRAM", color="#A23B72", alpha=0.7)
    ax_v.set_ylabel("Peak VRAM", color="#A23B72")

    # Quality overlay (scatter on a 0-1 normalized secondary scale via annotation).
    for xi, q in zip(x, quality):
        if q is not None:
            ax.annotate(f"Q={q:.0f}", (xi, tps[xi]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#6A994E")

    fig.legend(loc="upper right", fontsize=9)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)
