"""Matplotlib chart generation — publication-quality benchmark visualizations.

Generates a multi-panel PNG with 13+ performance charts covering throughput,
latency, TTFT, ITL, GPU metrics, energy efficiency, cache analysis, and more.

Author: amit
License: MIT
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    """Sanitize string for use in filenames."""
    safe = name.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^\w\-.]", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    return safe[:100]


def ensure_output_directory(output_dir: str = "./outputs") -> Path:
    """Create output directory if it doesn't exist."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------
# Main chart generator
# ------------------------------------------------------------------

def visualize_results(
    all_results: List[Dict],
    model_name: str,
    system_info: Dict = None,
    server_info: Dict = None,
    output_tokens: int = 500,
    output_dir: str = "./outputs",
) -> str:
    """Generate multi-panel benchmark visualization as PNG.

    Returns:
        Path to the saved PNG file.
    """
    df = pd.DataFrame(all_results)

    # Average across prompt types for main visualizations
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

    # Derived columns
    if "prefill_time_estimate" in df_main.columns and "avg_prompt_tokens" in df_main.columns:
        df_main["prompt_processing_speed"] = df_main["avg_prompt_tokens"] / (df_main["prefill_time_estimate"] + 0.001)
    else:
        df_main["prompt_processing_speed"] = df_main["avg_prompt_tokens"] / (df_main["avg_latency"] * 0.15 + 0.001)

    if "decode_time_estimate" in df_main.columns and "avg_completion_tokens" in df_main.columns:
        df_main["inter_token_latency"] = (df_main["decode_time_estimate"] / (df_main["avg_completion_tokens"] + 0.001)) * 1000
    else:
        df_main["inter_token_latency"] = ((df_main["avg_latency"] * 0.85) / (df_main["avg_completion_tokens"] + 0.001)) * 1000

    # Batch efficiency
    baseline_tp = {}
    for ctx in df_main["context_length"].unique():
        single = df_main[(df_main["context_length"] == ctx) & (df_main["concurrent_users"] == 1)]
        if len(single) > 0:
            baseline_tp[ctx] = single["tokens_per_second"].values[0]

    def calc_eff(row):
        base = baseline_tp.get(row["context_length"], 1)
        if row["concurrent_users"] == 1:
            return 100.0
        return (row["tokens_per_second"] / base / row["concurrent_users"]) * 100

    df_main["batch_efficiency"] = df_main.apply(calc_eff, axis=1)

    context_lengths = sorted(df_main["context_length"].unique())
    context_labels = [f"{int(c / 1000)}K" for c in context_lengths]

    # Style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.titlesize": 13,
    })

    has_cache = "cache_hit_rate" in df.columns
    need_cache_rows = has_prompt_types and has_cache
    if need_cache_rows:
        n_ptypes = len(df["prompt_type"].unique())
        cache_rows = (n_ptypes + 1) // 2
        num_rows = 6 + cache_rows
        height_ratios = [1, 1, 1, 1, 1, 1] + [0.8] * cache_rows
        fig_height = 24 + cache_rows * 5
    elif has_prompt_types:
        num_rows = 6
        height_ratios = [1, 1, 1, 1, 1, 1]
        fig_height = 24
    else:
        num_rows = 5
        height_ratios = [1, 1, 1, 1, 1]
        fig_height = 24

    fig = plt.figure(figsize=(24, fig_height))
    gs = fig.add_gridspec(num_rows, 3, hspace=0.40, wspace=0.28, left=0.06, right=0.98, top=0.96, bottom=0.04, height_ratios=height_ratios)
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]

    # ---- GRAPH 1: Throughput landscape ----
    ax1 = fig.add_subplot(gs[0, :])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax1.plot(d["context_length"] / 1000, d["tokens_per_second"], marker="o", linewidth=3, markersize=10,
                 label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax1.set_xlabel("Context Length (K tokens)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Throughput (tokens/sec)", fontsize=13, fontweight="bold")
    ax1.set_title("Throughput vs Context Length by Concurrency", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks([c / 1000 for c in context_lengths])
    ax1.set_xticklabels(context_labels)
    ax1.legend(title="Concurrent Users", fontsize=10, title_fontsize=11, loc="best", frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#FAFAFA")

    # ---- GRAPH 2: Throughput heatmap ----
    ax2 = fig.add_subplot(gs[1, :2])
    pivot_tp = df_main.pivot(index="context_length", columns="concurrent_users", values="tokens_per_second")
    sns.heatmap(pivot_tp, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax2, cbar_kws={"label": "Tokens/sec"}, linewidths=1.5, linecolor="white", annot_kws={"fontsize": 10, "weight": "bold"})
    ax2.set_xlabel("Concurrent Users", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Context Length", fontsize=11, fontweight="bold")
    ax2.set_title("Throughput Heatmap", fontsize=12, fontweight="bold", pad=10)
    ax2.set_yticklabels([f"{int(y / 1000)}K" for y in pivot_tp.index], rotation=0)

    # ---- GRAPH 3: Latency heatmap ----
    ax3 = fig.add_subplot(gs[1, 2])
    pivot_lat = df_main.pivot(index="context_length", columns="concurrent_users", values="avg_latency")
    sns.heatmap(pivot_lat, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax3, cbar_kws={"label": "Latency (sec)"}, linewidths=1.5, linecolor="white", annot_kws={"fontsize": 10, "weight": "bold"})
    ax3.set_xlabel("Concurrent Users", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Context Length", fontsize=11, fontweight="bold")
    ax3.set_title("Latency Heatmap", fontsize=12, fontweight="bold", pad=10)
    ax3.set_yticklabels([f"{int(y / 1000)}K" for y in pivot_lat.index], rotation=0)

    def _line_plot(ax, y_col, ylabel, title, marker="s"):
        for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
            d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
            ax.plot(d["context_length"] / 1000, d[y_col], marker=marker, linewidth=2.5, markersize=9,
                    label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([c / 1000 for c in context_lengths])
        ax.set_xticklabels(context_labels)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")

    # ---- ROW 3: Throughput/user, TTFT, Latency ----
    _line_plot(fig.add_subplot(gs[2, 0]), "throughput_per_user", "Tokens/sec per User", "Throughput per User")

    ax5 = fig.add_subplot(gs[2, 1])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax5.plot(d["context_length"] / 1000, d["ttft_estimate"] * 1000, marker="*", linewidth=2.5, markersize=11,
                 label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax5.axhspan(0, 200, alpha=0.08, color="green")
    ax5.axhspan(200, 1000, alpha=0.08, color="yellow")
    ax5.axhspan(1000, 3000, alpha=0.08, color="orange")
    ax5.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("TTFT (milliseconds)", fontsize=11, fontweight="bold")
    ax5.set_title("Time to First Token (UX Quality)", fontsize=12, fontweight="bold", pad=10)
    ax5.set_xticks([c / 1000 for c in context_lengths])
    ax5.set_xticklabels(context_labels)
    ax5.legend(fontsize=9, loc="upper left")
    ax5.grid(True, alpha=0.3, linestyle="--")
    ax5.set_facecolor("#FAFAFA")

    ax6 = fig.add_subplot(gs[2, 2])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax6.plot(d["context_length"] / 1000, d["avg_latency"], marker="D", linewidth=2.5, markersize=9,
                 label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax6.axhspan(0, 2, alpha=0.08, color="green")
    ax6.axhspan(2, 5, alpha=0.08, color="yellow")
    ax6.axhspan(5, 10, alpha=0.08, color="orange")
    ax6.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Latency (seconds)", fontsize=11, fontweight="bold")
    ax6.set_title("Average Latency (UX Quality)", fontsize=12, fontweight="bold", pad=10)
    ax6.set_xticks([c / 1000 for c in context_lengths])
    ax6.set_xticklabels(context_labels)
    ax6.legend(fontsize=9, loc="upper left")
    ax6.grid(True, alpha=0.3, linestyle="--")
    ax6.set_facecolor("#FAFAFA")

    # ---- ROW 4: Prompt processing, Power, GPU clock ----
    _line_plot(fig.add_subplot(gs[3, 0]), "prompt_processing_speed", "Prompt Processing (tokens/sec)", "Prompt Processing Speed", marker="^")

    ax8 = fig.add_subplot(gs[3, 1])
    if has_gpu:
        _line_plot(ax8, "avg_power", "Power Draw (W)", "Average Power Draw")
    else:
        ax8.text(0.5, 0.5, "GPU stats not available", ha="center", va="center", fontsize=12, transform=ax8.transAxes)
        ax8.axis("off")

    ax9 = fig.add_subplot(gs[3, 2])
    if has_gpu:
        _line_plot(ax9, "avg_gpu_clock", "GPU Clock (MHz)", "GPU Clock Frequency", marker="H")
    else:
        ax9.text(0.5, 0.5, "GPU stats not available", ha="center", va="center", fontsize=12, transform=ax9.transAxes)
        ax9.axis("off")

    # ---- ROW 5: ITL, Batch scaling, Decode speed ----
    ax10 = fig.add_subplot(gs[4, 0])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax10.plot(d["context_length"] / 1000, d["inter_token_latency"], marker="v", linewidth=2.5, markersize=9,
                  label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax10.axhspan(0, 50, alpha=0.1, color="green")
    ax10.axhspan(50, 100, alpha=0.1, color="yellow")
    ax10.axhspan(100, 200, alpha=0.1, color="orange")
    ax10.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax10.set_ylabel("Inter-Token Latency (ms)", fontsize=11, fontweight="bold")
    ax10.set_title("Inter-Token Latency (UX Quality)", fontsize=12, fontweight="bold", pad=10)
    ax10.set_xticks([c / 1000 for c in context_lengths])
    ax10.set_xticklabels(context_labels)
    ax10.legend(fontsize=8, loc="upper left")
    ax10.grid(True, alpha=0.3, linestyle="--")
    ax10.set_facecolor("#FAFAFA")

    ax11 = fig.add_subplot(gs[4, 1])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        if users == 1:
            continue
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax11.plot(d["context_length"] / 1000, d["batch_efficiency"], marker="p", linewidth=2.5, markersize=9,
                  label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax11.axhline(y=100, color="green", linestyle="--", linewidth=2, alpha=0.5, label="Perfect scaling")
    ax11.axhspan(80, 150, alpha=0.08, color="green")
    ax11.axhspan(50, 80, alpha=0.08, color="yellow")
    ax11.axhspan(0, 50, alpha=0.08, color="red")
    ax11.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax11.set_ylabel("Scaling Efficiency (%)", fontsize=11, fontweight="bold")
    ax11.set_title("Batch Scaling Efficiency", fontsize=12, fontweight="bold", pad=10)
    ax11.set_xticks([c / 1000 for c in context_lengths])
    ax11.set_xticklabels(context_labels)
    ax11.legend(fontsize=8, loc="best")
    ax11.grid(True, alpha=0.3, linestyle="--")
    ax11.set_facecolor("#FAFAFA")

    ax12 = fig.add_subplot(gs[4, 2])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        d = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        if "decode_time_estimate" in d.columns:
            decode_speed = d["avg_completion_tokens"] / (d["decode_time_estimate"] + 0.001)
        else:
            decode_speed = d["avg_completion_tokens"] / (d["avg_latency"] * 0.85 + 0.001)
        ax12.plot(d["context_length"] / 1000, decode_speed, marker="o", linewidth=2.5, markersize=9,
                  label=f"{users} users", color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    ax12.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax12.set_ylabel("Decode Speed (tokens/sec)", fontsize=11, fontweight="bold")
    ax12.set_title("Decode Speed (Generation)", fontsize=12, fontweight="bold", pad=10)
    ax12.set_xticks([c / 1000 for c in context_lengths])
    ax12.set_xticklabels(context_labels)
    ax12.legend(fontsize=8, loc="best")
    ax12.grid(True, alpha=0.3, linestyle="--")
    ax12.set_facecolor("#FAFAFA")

    # ---- ROW 6: Prompt type comparison (if applicable) ----
    if has_prompt_types and df["prompt_type"].nunique() >= 2:
        ax13 = fig.add_subplot(gs[5, :])
        available_users = sorted(df["concurrent_users"].unique())
        compare_users = 20 if 20 in available_users else max(available_users)
        comp_data = df[df["concurrent_users"] == compare_users]
        if len(comp_data) > 0:
            prompt_colors = {"classic": "#2E86AB", "deterministic": "#6A994E", "madlib": "#F18F01", "random": "#C73E1D"}
            for idx, pt in enumerate(sorted(comp_data["prompt_type"].unique())):
                pdata = comp_data[comp_data["prompt_type"] == pt].sort_values("context_length")
                color = prompt_colors.get(pt, colors[idx % len(colors)])
                if "actual_prefill_time" in pdata.columns and pdata["actual_prefill_time"].sum() > 0:
                    y = pdata["actual_prefill_time"]
                else:
                    y = pdata.get("prefill_time_estimate", pdata["avg_latency"] * 0.15)
                ax13.plot(pdata["context_length"] / 1000, y, marker="o", linewidth=3, markersize=10,
                          label=pt.capitalize(), color=color, markeredgecolor="white", markeredgewidth=1.5)
            ax13.set_xlabel("Context Length (K tokens)", fontsize=13, fontweight="bold")
            ax13.set_ylabel("Prefill Time (seconds)", fontsize=13, fontweight="bold")
            ax13.set_title(f"Prefill Time by Prompt Type ({compare_users} users)", fontsize=14, fontweight="bold", pad=15)
            ax13.set_xticks([c / 1000 for c in context_lengths])
            ax13.set_xticklabels(context_labels)
            ax13.legend(title="Prompt Type", fontsize=11, title_fontsize=12, loc="best", frameon=True, shadow=True)
            ax13.grid(True, alpha=0.3, linestyle="--")
            ax13.set_facecolor("#FAFAFA")

    # ---- Cache heatmaps ----
    if need_cache_rows:
        ptypes_list = sorted(df["prompt_type"].unique())
        for idx, pt in enumerate(ptypes_list):
            row_offset = 6
            cache_row = row_offset + (idx // 2)
            if idx == len(ptypes_list) - 1 and idx % 2 == 0:
                ax_c = fig.add_subplot(gs[cache_row, :])
            elif idx % 2 == 0:
                ax_c = fig.add_subplot(gs[cache_row, :2])
            else:
                ax_c = fig.add_subplot(gs[cache_row, 2])
            pt_data = df[df["prompt_type"] == pt]
            pivot_c = pt_data.pivot(index="context_length", columns="concurrent_users", values="cache_hit_rate")
            sns.heatmap(pivot_c, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax_c, cbar_kws={"label": "Hit Rate (%)"}, linewidths=1.5, linecolor="white", annot_kws={"fontsize": 9, "weight": "bold"}, vmin=0, vmax=100)
            ax_c.set_xlabel("Concurrent Users", fontsize=11, fontweight="bold")
            ax_c.set_ylabel("Context Length", fontsize=11, fontweight="bold")
            ax_c.set_title(f"Cache Hit Rate: {pt.capitalize()}", fontsize=12, fontweight="bold", pad=10)
            ax_c.set_yticklabels([f"{int(y / 1000)}K" for y in pivot_c.index], rotation=0)

    # Suptitle
    gpu_info = []
    if system_info:
        if system_info.get("gpu_name"):
            vram = system_info.get("total_vram_gb")
            gpu_info.append(f"{system_info['gpu_name']} ({vram:.0f}GB)" if vram else system_info["gpu_name"])
        if system_info.get("cuda_version"):
            gpu_info.append(f"CUDA {system_info['cuda_version']}")
    if server_info:
        if server_info.get("version"):
            gpu_info.append(f"vLLM {server_info['version']}")
        if server_info.get("quantization"):
            gpu_info.append(server_info["quantization"])
    subtitle = " | ".join(gpu_info) if gpu_info else "Performance Benchmark"
    subtitle += f" | Output: {output_tokens} tokens"
    fig.suptitle(f"{model_name} - Performance Benchmark\n{subtitle}", fontsize=13, fontweight="bold", y=0.995,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = sanitize_filename(model_name)
    out = ensure_output_directory(output_dir)
    filepath = out / f"benchmark_{safe}_{ts}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(filepath)
