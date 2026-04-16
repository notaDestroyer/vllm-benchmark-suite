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
    remaining_tests: List[Tuple[int, int, str]] = None,
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
        for i, (ctx, users, ptype) in enumerate(remaining_tests):
            queue_text.append(f"  {i + 1}. {ctx // 1000}K x {users} users x {ptype}\n", style="dim")
        layout["remaining"].update(Panel(queue_text, title=f"Queue ({len(remaining_tests)} remaining)", border_style="blue"))
    else:
        layout["remaining"].update(Panel("Final test running", title="Queue", border_style="green"))

    return layout
