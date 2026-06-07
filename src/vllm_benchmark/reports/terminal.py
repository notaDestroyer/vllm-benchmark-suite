"""Rich terminal output — summary tables and live dashboard.

Author: amit
License: MIT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary_table(all_results: List[Dict]) -> None:
    """Print detailed performance summary tables to console."""
    import pandas as pd

    df = pd.DataFrame(all_results)
    has_gpu = "avg_gpu_util" in df.columns
    has_energy = "watts_per_token_per_user" in df.columns
    has_cache = "cache_hit_rate" in df.columns

    console.print(f"\n{'=' * 140}")
    console.print("[bold]DETAILED PERFORMANCE SUMMARY[/bold]")
    console.print(f"{'=' * 140}")

    for context in sorted(df["context_length"].unique()):
        ctx_data = df[df["context_length"] == context].sort_values("concurrent_users")
        console.print(f"\n[bold cyan]Context Length: {context:,} tokens ({context / 1000:.0f}K)[/bold cyan]")

        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
        table.add_column("Users", justify="right")
        table.add_column("Latency(s)", justify="right")
        table.add_column("Tok/s", justify="right")
        table.add_column("Req/s", justify="right")
        table.add_column("TTFT(ms)", justify="right")
        if has_gpu:
            table.add_column("GPU%", justify="right")
            table.add_column("Temp(C)", justify="right")
            table.add_column("Power(W)", justify="right")
        if has_energy:
            table.add_column("W/tok/usr", justify="right")
        if has_cache:
            table.add_column("Cache%", justify="right")
        table.add_column("Success%", justify="right")

        for _, row in ctx_data.iterrows():
            success_rate = (row["successful"] / (row["successful"] + row["failed"])) * 100
            cols = [
                str(int(row["concurrent_users"])),
                f"{row['avg_latency']:.2f}",
                f"{row['tokens_per_second']:.1f}",
                f"{row['requests_per_second']:.2f}",
                f"{row['ttft_estimate'] * 1000:.0f}",
            ]
            if has_gpu:
                cols.extend([
                    f"{row['avg_gpu_util']:.1f}",
                    f"{row['avg_temperature']:.1f}",
                    f"{row['avg_power']:.1f}",
                ])
            if has_energy:
                cols.append(f"{row['watts_per_token_per_user']:.4f}")
            if has_cache:
                cols.append(f"{row.get('cache_hit_rate', 0):.1f}")
            cols.append(f"{success_rate:.1f}")
            table.add_row(*cols)

        console.print(table)

    # Optimal configurations
    console.print(f"\n{'=' * 100}")
    console.print("[bold]OPTIMAL CONFIGURATIONS[/bold]")
    console.print(f"{'=' * 100}")

    max_tp = df.loc[df["tokens_per_second"].idxmax()]
    console.print(
        f"\n[bold green]Maximum Throughput:[/] {max_tp['tokens_per_second']:.1f} tok/s "
        f"at {int(max_tp['concurrent_users'])} users with {max_tp['context_length'] / 1000:.0f}K context"
    )

    best_eff = df.loc[df["throughput_per_user"].idxmax()]
    console.print(
        f"[bold green]Best Efficiency:[/] {best_eff['throughput_per_user']:.1f} tok/s/user "
        f"at {int(best_eff['concurrent_users'])} users with {best_eff['context_length'] / 1000:.0f}K context"
    )

    min_lat = df.loc[df["avg_latency"].idxmin()]
    console.print(
        f"[bold green]Lowest Latency:[/] {min_lat['avg_latency']:.2f}s "
        f"at {int(min_lat['concurrent_users'])} users with {min_lat['context_length'] / 1000:.0f}K context"
    )

    if has_energy and "tokens_per_watt" in df.columns:
        console.print("\n[bold]Energy Analysis:[/]")
        console.print(f"  Best efficiency: {df['tokens_per_watt'].max():.2f} tok/W")
        console.print(f"  Avg efficiency:  {df['tokens_per_watt'].mean():.2f} tok/W")
        if "energy_watt_hours" in df.columns:
            total_wh = df["energy_watt_hours"].sum()
            console.print(f"  Total energy:    {total_wh:.4f} Wh ({total_wh * 1000:.2f} mWh)")

    if has_cache:
        console.print("\n[bold]Cache Analysis:[/]")
        console.print(f"  Best hit rate:  {df['cache_hit_rate'].max():.1f}%")
        console.print(f"  Avg hit rate:   {df['cache_hit_rate'].mean():.1f}%")
        if "prompt_type" in df.columns:
            for pt in sorted(df["prompt_type"].unique()):
                avg_cache = df[df["prompt_type"] == pt]["cache_hit_rate"].mean()
                console.print(f"  {pt.capitalize()}: {avg_cache:.1f}%")


# ------------------------------------------------------------------
# PR5 analysis panels — model intel / bottleneck / fitness
# ------------------------------------------------------------------

def _na(value: object, suffix: str = "", fmt: str = "{}") -> str:
    """Format a value with a suffix, or ``"N/A"`` when missing."""
    if value is None:
        return "N/A"
    try:
        return f"{fmt.format(value)}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def render_model_intel_panel(model_profile: Optional[Dict]) -> Panel:
    """Render the model-intelligence panel (architecture + roofline facts).

    Args:
        model_profile: ``ModelProfile.to_dict()`` (or ``None``).

    Returns:
        A rich :class:`~rich.panel.Panel` summarising the model profile.
    """
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow")

    p = model_profile or {}
    arch = "MoE" if p.get("is_moe") else "dense" if p.get("is_moe") is not None else "N/A"

    def _params(n: object) -> str:
        return _na(n / 1e9 if isinstance(n, (int, float)) else None, "B", "{:.1f}")

    table.add_row("Model", str(p.get("name") or "N/A"))
    table.add_row("Family", str(p.get("family") or "N/A"))
    table.add_row("Architecture", arch)
    table.add_row("Active params", _params(p.get("active_params")))
    table.add_row("Total params", _params(p.get("total_params")))
    table.add_row("Attention", str(p.get("attention_type") or "N/A"))
    table.add_row("Layers", _na(p.get("num_layers")))
    if p.get("is_moe"):
        table.add_row("Experts", f"{_na(p.get('experts_per_tok'))}/{_na(p.get('num_experts'))}")
    table.add_row("KV bytes/token", _na(p.get("kv_bytes_per_token")))
    table.add_row("Provenance", f"{p.get('source', 'N/A')} ({p.get('confidence', 'N/A')})")

    return Panel(table, title="[bold]Model Intelligence[/bold]", border_style="cyan")


def render_bottleneck_panel(bottlenecks: Optional[List[Dict]]) -> Panel:
    """Render the governing-bottleneck panel for a run.

    Highlights the top (highest-confidence) verdict and lists per-cell
    primaries with MBU/MFU and the recommended lever.

    Args:
        bottlenecks: List of ``BottleneckVerdict.to_dict()`` dicts.

    Returns:
        A rich :class:`~rich.panel.Panel`.
    """
    verdicts = bottlenecks or []
    if not verdicts:
        return Panel("No bottleneck verdicts available.",
                     title="[bold]Bottleneck Analysis[/bold]", border_style="magenta")

    order = {"high": 0, "medium": 1, "low": 2}
    top = min(verdicts, key=lambda v: order.get(v.get("confidence"), 3))

    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAVY)
    table.add_column("Cell (ctx/users)", style="cyan")
    table.add_column("Primary", style="yellow")
    table.add_column("MBU", justify="right")
    table.add_column("MFU", justify="right")
    table.add_column("Conf", justify="center")

    for v in verdicts:
        cell = v.get("cell") or []
        ctx = cell[0] if len(cell) > 0 else None
        users = cell[1] if len(cell) > 1 else None
        ctx_label = f"{ctx // 1000}K" if isinstance(ctx, int) and ctx >= 1000 else str(ctx)
        table.add_row(
            f"{ctx_label}/{_na(users)}",
            str(v.get("primary") or "unknown"),
            _na(v.get("mbu"), "", "{:.0%}"),
            _na(v.get("mfu"), "", "{:.0%}"),
            str(v.get("confidence") or "?"),
        )

    header = Text()
    header.append("Governing: ", style="bold")
    header.append(f"{top.get('primary', 'unknown')} ", style="bold yellow")
    header.append(f"(confidence {top.get('confidence', '?')})\n", style="dim")
    header.append("Lever: ", style="bold")
    header.append(str(top.get("lever") or "N/A"), style="green")

    from rich.console import Group
    return Panel(Group(header, table),
                 title="[bold]Bottleneck Analysis[/bold]", border_style="magenta")


def render_fitness_panel(advisory_fitness: Optional[Dict]) -> Panel:
    """Render the application-fitness panel.

    Args:
        advisory_fitness: The ``advisory["fitness"]`` dict (with a
            ``verdict`` string and per-profile ``profiles`` grades).

    Returns:
        A rich :class:`~rich.panel.Panel`.
    """
    fitness = advisory_fitness or {}
    profiles = fitness.get("profiles") or {}

    table = Table(show_header=True, header_style="bold green", box=box.SIMPLE)
    table.add_column("Profile", style="cyan")
    table.add_column("Grade", justify="center")
    table.add_column("Limiting factor", style="dim")

    grade_styles = {"Good": "green", "Marginal": "yellow", "Poor": "red", "N/A": "dim"}
    for name, grade in profiles.items():
        g = grade.get("grade", "N/A")
        table.add_row(
            name,
            f"[{grade_styles.get(g, 'white')}]{g}[/]",
            str(grade.get("limiting_factor") or ""),
        )

    verdict = fitness.get("verdict") or "N/A"
    header = Text()
    header.append("Verdict: ", style="bold")
    header.append(verdict, style="bold yellow")

    from rich.console import Group
    if not profiles:
        return Panel(header, title="[bold]Application Fitness[/bold]", border_style="green")
    return Panel(Group(header, table),
                 title="[bold]Application Fitness[/bold]", border_style="green")


# ------------------------------------------------------------------
# Live dashboard
# ------------------------------------------------------------------

def create_live_dashboard(
    test_num: int,
    total_tests: int,
    context_length: int,
    concurrent_users: int,
    elapsed_time: float,
    current_gpu: Optional[Dict] = None,
    all_results: List[Dict] = None,
    remaining_tests: List[Tuple] = None,
    all_gpu_history: List[Dict] = None,
    total_benchmark_time: float = 0,
) -> Layout:
    """Create a live dashboard layout with progress bars and status."""
    remaining_size = min(max(8, len(remaining_tests) + 3 if remaining_tests else 6), 35)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="overall_progress", size=5),
        Layout(name="current_test", size=8),
        Layout(name="remaining", size=remaining_size),
    )

    # Header
    header_text = Text()
    header_text.append("vLLM Benchmark Suite ", style="bold cyan")
    header_text.append(f"Test {test_num}/{total_tests}", style="bold yellow")
    if total_benchmark_time > 0:
        mins = int(total_benchmark_time // 60)
        secs = int(total_benchmark_time % 60)
        header_text.append(f"  |  Runtime: {mins}m {secs}s", style="dim")
    layout["header"].update(Panel(header_text, style="cyan"))

    # Progress bar
    progress_pct = (test_num / total_tests) * 100
    bar_width = 60
    filled = int((test_num / total_tests) * bar_width)
    overall_bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

    progress_text = Text()
    progress_text.append("OVERALL PROGRESS\n", style="bold green")
    progress_text.append(f"{overall_bar} ", style="green")
    progress_text.append(f"{progress_pct:.1f}%\n", style="bold green")
    progress_text.append(f"Completed: {test_num}  Remaining: {total_tests - test_num}", style="dim")
    layout["overall_progress"].update(Panel(progress_text, title="Benchmark Progress", border_style="green"))

    # Current test
    test_info = Table(show_header=False, box=box.SIMPLE, border_style="cyan")
    test_info.add_column("", style="cyan", width=15)
    test_info.add_column("", style="yellow")
    test_info.add_row("Context", f"{context_length // 1000}K tokens")
    test_info.add_row("Users", str(concurrent_users))
    test_info.add_row("Elapsed", f"{elapsed_time:.1f}s")
    if current_gpu:
        util = current_gpu.get("gpu_util", 0)
        util_color = "red" if util > 95 else "yellow" if util > 80 else "green"
        test_info.add_row("GPU", f"[{util_color}]{util:.0f}%[/{util_color}]")
    test_info.add_row("Status", "[bold yellow]RUNNING[/bold yellow]")
    layout["current_test"].update(Panel(test_info, title=f"Current Test ({test_num}/{total_tests})", border_style="yellow"))

    # Remaining queue
    if remaining_tests and len(remaining_tests) > 0:
        queue_text = Text()
        queue_text.append("Remaining tests:\n\n", style="bold blue")
        for i, test in enumerate(remaining_tests):
            ctx, users, ptype = test[0], test[1], test[2]
            iter_str = f" (iter {test[3] + 1})" if len(test) > 3 else ""
            queue_text.append(f"  {i + 1}. {ctx // 1000}K x {users} users x {ptype}{iter_str}\n", style="dim")
        layout["remaining"].update(Panel(queue_text, title=f"Queue ({len(remaining_tests)} remaining)", border_style="blue"))
    else:
        layout["remaining"].update(Panel("Final test running", title="Queue", border_style="green"))

    return layout
