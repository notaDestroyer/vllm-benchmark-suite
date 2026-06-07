"""Report generation components."""

from vllm_benchmark.reports.card import render_result_card
from vllm_benchmark.reports.charts import (
    bottleneck_grid,
    plot_bottleneck_map,
    plot_quant_compare,
    plot_roofline,
    visualize_results,
)
from vllm_benchmark.reports.html_report import generate_html_report
from vllm_benchmark.reports.share import build_share_markdown, save_share_markdown
from vllm_benchmark.reports.terminal import (
    create_live_dashboard,
    render_bottleneck_panel,
    render_fitness_panel,
    render_model_intel_panel,
)

__all__ = [
    "visualize_results",
    "generate_html_report",
    "create_live_dashboard",
    "render_result_card",
    "build_share_markdown",
    "save_share_markdown",
    "bottleneck_grid",
    "plot_bottleneck_map",
    "plot_quant_compare",
    "plot_roofline",
    "render_bottleneck_panel",
    "render_fitness_panel",
    "render_model_intel_panel",
]
