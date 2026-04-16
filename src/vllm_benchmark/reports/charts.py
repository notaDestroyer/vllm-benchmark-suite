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
