"""CLI entry point for ``vllm-bench``.

Author: amit
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from statistics import mean
from typing import List, Tuple

import pandas as pd
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt
from rich.table import Table

from vllm_benchmark import __version__
from vllm_benchmark.config import (
    GPU_PRICING,
    BenchmarkConfig,
    parse_concurrency,
    parse_context_lengths,
)
from vllm_benchmark.core.async_engine import run_benchmark
from vllm_benchmark.core.benchmark import warmup_model
from vllm_benchmark.core.metrics import GPUMonitor
from vllm_benchmark.core.server import SystemInfo, VLLMServerInfo
from vllm_benchmark.reports.terminal import (
    create_live_dashboard,
)

console = Console()

DASHBOARD_REFRESH_RATE = 2  # Hz


# ------------------------------------------------------------------
# Interactive configuration (fallback when no CLI args)
# ------------------------------------------------------------------

def get_interactive_config() -> Tuple[List[int], List[int], int, List[str]]:
    """Interactive CLI for benchmark configuration."""
    console.print("\n[bold cyan]Benchmark Configuration[/bold cyan]\n")

    # Context lengths
    console.print("[bold]Select maximum context length:[/bold]")
    for i, (val, label) in enumerate([(32, "32K"), (64, "64K"), (128, "128K"), (256, "256K"), (512, "512K"), (1024, "1M")], 1):
        console.print(f"  [{i}] {label}")
    console.print()
    max_choice = IntPrompt.ask("Select max context", default=3, choices=["1", "2", "3", "4", "5", "6"])
    max_context = {1: 32, 2: 64, 3: 128, 4: 256, 5: 512, 6: 1024}[max_choice]
    all_ctx = [1, 10, 32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024]
    context_lengths = [c * 1000 for c in all_ctx if c <= max_context]

    # Users
    console.print("\n[bold]Select maximum concurrent users:[/bold]")
    for i, val in enumerate([1, 2, 5, 10, 20, 50], 1):
        console.print(f"  [{i}] {val} {'user' if val == 1 else 'users'}")
    console.print("  [7] Custom\n")
    user_choice = IntPrompt.ask("Select max users", default=4, choices=["1", "2", "3", "4", "5", "6", "7"])
    if user_choice == 7:
        max_users = IntPrompt.ask("Enter max concurrent users", default=10)
    else:
        max_users = {1: 1, 2: 2, 3: 5, 4: 10, 5: 20, 6: 50}[user_choice]
    all_u = [1, 2, 5, 10, 20, 50, 100]
    concurrent_users = [u for u in all_u if u <= max_users]

    # Output tokens
    console.print("\n[bold]Select output length:[/bold]")
    console.print("  [1] Short (150 tokens)")
    console.print("  [2] Standard (500 tokens)")
    console.print("  [3] Long (1500 tokens)")
    console.print("  [4] Custom\n")
    out_choice = IntPrompt.ask("Select output length", default=2, choices=["1", "2", "3", "4"])
    output_tokens = {1: 150, 2: 500, 3: 1500}.get(out_choice)
    if output_tokens is None:
        output_tokens = IntPrompt.ask("Enter output tokens", default=500)

    # Prompt types
    console.print("\n[bold]Select prompt types:[/bold]")
    console.print("  [1] Classic only")
    console.print("  [2] Deterministic only (high cache hit)")
    console.print("  [3] Madlib only (moderate cache hit)")
    console.print("  [4] Random only (low cache hit)")
    console.print("  [5] All three new types")
    console.print("  [6] All four types\n")
    pt_choice = IntPrompt.ask("Select prompt types", default=5, choices=["1", "2", "3", "4", "5", "6"])
    prompt_types = {
        1: ["classic"], 2: ["deterministic"], 3: ["madlib"], 4: ["random"],
        5: ["deterministic", "madlib", "random"],
        6: ["classic", "deterministic", "madlib", "random"],
    }[pt_choice]

    # Summary
    total = len(context_lengths) * len(concurrent_users) * len(prompt_types)
    config_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    config_table.add_row("Context Lengths", ", ".join(f"{c // 1000}K" for c in context_lengths))
    config_table.add_row("Concurrent Users", ", ".join(str(u) for u in concurrent_users))
    config_table.add_row("Output Tokens", str(output_tokens))
    config_table.add_row("Prompt Types", ", ".join(prompt_types))
    config_table.add_row("Total Tests", str(total))
    config_table.add_row("Est. Duration", f"{total * 30 // 60} min")
    console.print(config_table)

    if not Confirm.ask("\nProceed?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)

    return context_lengths, concurrent_users, output_tokens, prompt_types


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-bench",
        description="vLLM Benchmark Suite — Comprehensive performance testing for vLLM inference servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vllm-bench --quick                     Quick 5-minute smoke test
  vllm-bench --standard                  Standard 30-minute benchmark
  vllm-bench --thorough                  Full 2-hour benchmark
  vllm-bench --url http://gpu2:8000      Test a remote server
  vllm-bench --compare previous.json     Detect performance regressions
  vllm-bench --context-lengths 32k,64k --concurrency 1,4,8
""",
    )

    parser.add_argument("--version", action="version", version=f"vllm-bench {__version__}")

    # Connection
    conn = parser.add_argument_group("Connection")
    conn.add_argument("--url", default="http://localhost:8000", help="vLLM server URL (default: http://localhost:8000)")
    conn.add_argument("--model", default=None, help="Model name override (auto-detected if omitted)")
    conn.add_argument("--tokenizer", default=None,
                       help="Tokenizer HF repo id or local path (default: same as --model; env: VLLM_BENCH_TOKENIZER)")

    # Presets
    presets = parser.add_argument_group("Presets")
    preset_group = presets.add_mutually_exclusive_group()
    preset_group.add_argument("--quick", action="store_true", help="Quick benchmark (~5 min)")
    preset_group.add_argument("--standard", action="store_true", help="Standard benchmark (~30 min)")
    preset_group.add_argument("--thorough", action="store_true", help="Thorough benchmark (~2 hours)")

    # Test parameters
    params = parser.add_argument_group("Test parameters")
    params.add_argument("--context-lengths", default=None, help="Comma-separated context lengths (e.g. 32k,64k,128k)")
    params.add_argument("--concurrency", default=None, help="Comma-separated concurrency levels (e.g. 1,4,8,16)")
    params.add_argument("--output-tokens", type=int, default=None, help="Max output tokens per request (default: 500)")
    params.add_argument("--prompt-type", default=None, help="Prompt type: classic|deterministic|madlib|random|all")
    params.add_argument("--prompts-file", default=None, help="Path to custom prompts JSONL file")
    params.add_argument("--rps", type=float, default=None, help="Sustained requests per second mode")
    params.add_argument("--duration", type=float, default=120, help="Duration in seconds for sustained RPS mode")
    params.add_argument("--iterations", type=int, default=1, help="Iterations per config for statistical rigor (default: 1)")
    params.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Cost
    cost_group = parser.add_argument_group("Cost")
    cost_group.add_argument("--cost", type=float, default=None,
                             help="GPU cost USD/hr (auto-detected for known GPUs)")

    # Behavior
    behavior = parser.add_argument_group("Behavior")
    behavior.add_argument("-y", "--non-interactive", action="store_true", help="Skip interactive prompts, use defaults")
    behavior.add_argument("--no-warmup", action="store_true", help="Skip model warmup")
    behavior.add_argument("--no-streaming", action="store_true", help="Disable streaming TTFT measurement")

    # Output
    output = parser.add_argument_group("Output")
    output.add_argument("--output-dir", default="./outputs", help="Output directory (default: ./outputs)")
    output.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    output.add_argument("--no-charts", action="store_true", help="Skip PNG chart generation")

    # Traffic simulation
    traffic = parser.add_argument_group("Traffic simulation")
    traffic.add_argument("--traffic", choices=["poisson", "multiturn"], default=None,
                         help="Run traffic simulation instead of standard benchmark")
    traffic.add_argument("--target-rps", type=float, default=2.0, help="Target requests per second (default: 2.0)")
    traffic.add_argument("--traffic-duration", type=float, default=60.0, help="Traffic simulation duration in seconds (default: 60)")
    traffic.add_argument("--turns", type=int, default=5, help="Turns per conversation for multiturn mode (default: 5)")

    # Comparison
    comp = parser.add_argument_group("Comparison")
    comp.add_argument("--compare", default=None, metavar="FILE", help="Compare with previous results JSON for regression detection")

    return parser


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    """Main CLI entry point for vllm-bench."""
    parser = build_parser()
    args = parser.parse_args()

    # Build config
    config = BenchmarkConfig(api_url=args.url, output_dir=args.output_dir, streaming=not args.no_streaming)

    if args.model:
        config.model_name = args.model

    # Determine test matrix
    has_explicit_params = args.context_lengths or args.concurrency or args.output_tokens or args.prompt_type

    if args.quick:
        config = BenchmarkConfig.from_preset("quick", api_url=args.url, output_dir=args.output_dir, streaming=not args.no_streaming)
    elif args.standard:
        config = BenchmarkConfig.from_preset("standard", api_url=args.url, output_dir=args.output_dir, streaming=not args.no_streaming)
    elif args.thorough:
        config = BenchmarkConfig.from_preset("thorough", api_url=args.url, output_dir=args.output_dir, streaming=not args.no_streaming)
    elif has_explicit_params:
        if args.context_lengths:
            config.context_lengths = parse_context_lengths(args.context_lengths)
        if args.concurrency:
            config.concurrency_levels = parse_concurrency(args.concurrency)
        if args.output_tokens:
            config.output_tokens = args.output_tokens
        if args.prompt_type:
            if args.prompt_type == "all":
                config.prompt_types = ["classic", "deterministic", "madlib", "random"]
            else:
                config.prompt_types = [p.strip() for p in args.prompt_type.split(",")]
    elif not args.non_interactive:
        # Fall back to interactive mode
        ctx, users, out_tok, ptypes = get_interactive_config()
        config.context_lengths = ctx
        config.concurrency_levels = users
        config.output_tokens = out_tok
        config.prompt_types = ptypes

    if args.model:
        config.model_name = args.model
    tokenizer_override = args.tokenizer or os.environ.get("VLLM_BENCH_TOKENIZER")
    if tokenizer_override:
        config.tokenizer = tokenizer_override
    config.warmup = not args.no_warmup
    config.generate_html = not args.no_html
    config.generate_charts = not args.no_charts
    if args.prompts_file:
        config.prompts_file = args.prompts_file

    # Seed for reproducibility
    if args.seed is not None:
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
    if args.compare:
        config.compare_file = args.compare

    # ---- Header ----
    console.print(Panel.fit(
        f"[bold cyan]vLLM Benchmark Suite[/bold cyan]\n[dim]v{__version__}[/dim]",
        border_style="cyan",
    ))

    # ---- System detection ----
    console.print("\n[yellow]Detecting system...[/yellow]")
    system_info = SystemInfo.get_system_info()
    console.print("[yellow]Querying vLLM server...[/yellow]")
    server_info = VLLMServerInfo.get_server_info(config)
    model_name = config.model_name or server_info.get("model_name") or "unknown"

    # ---- Auto-detect GPU cost ----
    cost_per_hour = args.cost  # explicit CLI override takes priority
    if cost_per_hour is None:
        gpu_name = system_info.get("gpu_name", "")
        if gpu_name:
            gpu_lower = gpu_name.lower()
            for pricing_key, price in GPU_PRICING.items():
                if pricing_key.lower() in gpu_lower or gpu_lower in pricing_key.lower():
                    cost_per_hour = price
                    break

    # Display system panel
    sys_table = Table(show_header=False, box=box.ROUNDED, border_style="green")
    sys_table.add_column("", style="cyan bold")
    sys_table.add_column("", style="yellow")
    if system_info.get("gpu_name"):
        vram = system_info.get("total_vram_gb")
        sys_table.add_row("GPU", f"{system_info['gpu_name']} ({vram:.0f}GB)" if vram else system_info["gpu_name"])
    if system_info.get("cuda_version"):
        sys_table.add_row("CUDA", system_info["cuda_version"])
    sys_table.add_row("Model", model_name)
    if config.tokenizer:
        sys_table.add_row("Tokenizer", config.tokenizer)
    if server_info.get("version"):
        sys_table.add_row("vLLM", server_info["version"])
    if server_info.get("quantization"):
        sys_table.add_row("Quantization", server_info["quantization"])
    if server_info.get("max_model_len"):
        sys_table.add_row("Max Context", f"{server_info['max_model_len']:,} tokens")
    console.print(Panel(sys_table, title="[bold green]System[/bold green]", border_style="green"))

    # Show test plan
    iterations = args.iterations
    total_runs = config.total_tests * iterations
    console.print(f"\n[bold]Test plan:[/bold] {config.total_tests} configs x {iterations} iterations = {total_runs} runs")
    console.print(f"  Context:     {', '.join(f'{c // 1000}K' for c in config.context_lengths)}")
    console.print(f"  Concurrency: {', '.join(str(u) for u in config.concurrency_levels)}")
    console.print(f"  Prompts:     {', '.join(config.prompt_types)}")
    console.print(f"  Output:      {config.output_tokens} tokens")
    if config.streaming:
        console.print("  TTFT:        [green]Streaming (true measurement)[/green]")
    if iterations > 1:
        console.print(f"  Iterations:  {iterations} (with statistical aggregation)")
    if args.seed is not None:
        console.print(f"  Seed:        {args.seed}")

    # ---- Traffic simulation mode ----
    if args.traffic:
        from vllm_benchmark.core.prompts import generate_prompt
        from vllm_benchmark.core.traffic import TrafficConfig, format_traffic_report, run_multiturn_traffic, run_poisson_traffic

        traffic_config = TrafficConfig(
            target_rps=args.target_rps,
            duration_seconds=args.traffic_duration,
            multi_turn=(args.traffic == "multiturn"),
            turns_per_conversation=args.turns,
            initial_context_tokens=config.context_lengths[0] if config.context_lengths else 32000,
            max_tokens=config.output_tokens,
        )

        console.print(f"\n[bold yellow]Running {args.traffic} traffic simulation...[/bold yellow]")
        console.print(f"  Target RPS: {traffic_config.target_rps}")
        console.print(f"  Duration:   {traffic_config.duration_seconds}s")
        if args.traffic == "multiturn":
            console.print(f"  Turns:      {traffic_config.turns_per_conversation}")

        if args.traffic == "multiturn":
            traffic_result = run_multiturn_traffic(
                traffic_config, model_name, config.api_endpoint, generate_prompt, config.request_timeout,
            )
        else:
            traffic_result = run_poisson_traffic(
                traffic_config, model_name, config.api_endpoint, generate_prompt, config.request_timeout,
            )

        report = format_traffic_report(traffic_result, traffic_config)
        console.print(Panel(report, title="[bold]Traffic Simulation Results[/bold]", border_style="cyan"))
        console.print("\n[bold green]Traffic simulation complete.[/bold green]\n")
        sys.exit(0)

    # ---- Sustained RPS mode ----
    if args.rps is not None:
        import asyncio

        from vllm_benchmark.core.async_engine import run_sustained_benchmark

        console.print("\n[bold yellow]Running sustained RPS benchmark...[/bold yellow]")
        console.print(f"  Target RPS: {args.rps}")
        console.print(f"  Duration:   {args.duration}s")
        console.print(f"  Context:    {config.context_lengths[0] // 1000}K")
        console.print(f"  Prompt:     {config.prompt_types[0]}")
        if cost_per_hour is not None:
            console.print(f"  GPU Cost:   ${cost_per_hour:.2f}/hr")

        result = asyncio.run(run_sustained_benchmark(
            context_length=config.context_lengths[0],
            target_rps=args.rps,
            duration_seconds=args.duration,
            config=config,
            model_name=model_name,
            prompt_type=config.prompt_types[0],
            cost_per_hour=cost_per_hour,
        ))

        if result:
            rps_table = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE, title="Sustained RPS Results")
            rps_table.add_column("Metric", style="cyan")
            rps_table.add_column("Value", style="yellow", justify="right")
            rps_table.add_row("Target RPS", f"{args.rps:.1f}")
            rps_table.add_row("Actual RPS", f"{result.get('actual_rps', 0):.1f}")
            rps_table.add_row("Throughput", f"{result['tokens_per_second']:.1f} tok/s")
            rps_table.add_row("Avg Latency", f"{result['avg_latency']:.2f}s")
            rps_table.add_row("P99 Latency", f"{result.get('latency_p99', 0):.2f}s")
            rps_table.add_row("Success Rate", f"{result['successful']}/{result['total_requests']}")
            if result.get("cost_per_1m_tokens"):
                rps_table.add_row("Cost / 1M tokens", f"${result['cost_per_1m_tokens']:.4f}")
            console.print(Panel(rps_table, border_style="cyan"))
        else:
            console.print("[red]Sustained RPS benchmark returned no results.[/red]")

        console.print("\n[bold green]Sustained RPS benchmark complete.[/bold green]\n")
        sys.exit(0)

    # ---- Warmup ----
    if config.warmup:
        if not warmup_model(config, model_name):
            if not args.non_interactive and not Confirm.ask("[yellow]Warmup failed. Continue?[/yellow]", default=False):
                sys.exit(1)
        time.sleep(3)

    # ---- Benchmark loop ----
    all_results: list[dict] = []
    iteration_runs: list[list[dict]] = []  # for multi-iteration aggregation
    test_queue = [
        (ctx, users, pt, it)
        for it in range(iterations)
        for pt in config.prompt_types
        for ctx in config.context_lengths
        for users in config.concurrency_levels
    ]
    total_tests = len(test_queue)
    benchmark_start = time.time()

    console.print("\n[bold green]Starting benchmark...[/bold green]\n")

    gpu_monitor = GPUMonitor(config.gpu_poll_interval)
    gpu_monitor.start()
    all_gpu_history: list[dict] = []

    with Live(console=console, refresh_per_second=DASHBOARD_REFRESH_RATE) as live_display:
        for idx, (context, users, ptype, iteration) in enumerate(test_queue):
            current_test = idx + 1
            remaining = test_queue[idx + 1:]
            test_start = time.time()

            test_result = [None]

            def run_test(_ctx=context, _users=users, _pt=ptype):
                test_result[0] = run_benchmark(
                    _ctx, _users, config, model_name=model_name,
                    live_display=live_display, gpu_monitor=gpu_monitor, prompt_type=_pt,
                    cost_per_hour=cost_per_hour,
                )

            test_thread = threading.Thread(target=run_test)
            test_thread.start()

            while test_thread.is_alive():
                elapsed = time.time() - test_start
                total_elapsed = time.time() - benchmark_start
                current_gpu = gpu_monitor.get_gpu_stats()
                if current_gpu:
                    all_gpu_history.append(current_gpu)
                dashboard = create_live_dashboard(
                    current_test, total_tests, context, users, elapsed,
                    current_gpu, all_results, remaining, all_gpu_history, total_elapsed,
                )
                live_display.update(dashboard)
                time.sleep(0.5)

            test_thread.join()
            result = test_result[0]

            if result:
                # Assign GPU stats from this test's time window
                test_gpu = [s for s in all_gpu_history if s.get("timestamp", 0) >= test_start]
                if test_gpu:
                    result.update({
                        "avg_gpu_util": mean([s["gpu_util"] for s in test_gpu]),
                        "max_gpu_util": max([s["gpu_util"] for s in test_gpu]),
                        "avg_mem_used": mean([s["mem_used"] for s in test_gpu]),
                        "max_mem_used": max([s["mem_used"] for s in test_gpu]),
                        "avg_temperature": mean([s["temperature"] for s in test_gpu]),
                        "max_temperature": max([s["temperature"] for s in test_gpu]),
                        "avg_power": mean([s["power_draw"] for s in test_gpu]),
                        "max_power": max([s["power_draw"] for s in test_gpu]),
                        "avg_gpu_clock": mean([s["gpu_clock"] for s in test_gpu]),
                        "max_gpu_clock": max([s["gpu_clock"] for s in test_gpu]),
                        "avg_mem_clock": mean([s["mem_clock"] for s in test_gpu]),
                    })
                result["iteration"] = iteration
                all_results.append(result)

                iter_tag = f" (iter {iteration + 1})" if iterations > 1 else ""
                summary = (
                    f"[green]OK[/green] {current_test}/{total_runs} "
                    f"{context // 1000}K x {users}u x {ptype}{iter_tag}: "
                    f"[bold yellow]{result['tokens_per_second']:.1f}[/bold yellow] tok/s, "
                    f"[bold cyan]{result['avg_latency']:.2f}s[/bold cyan]"
                )
                if "avg_gpu_util" in result:
                    summary += f", [magenta]{result['avg_gpu_util']:.0f}%[/magenta] GPU"
                console.print(summary)

            if current_test < total_tests:
                time.sleep(config.pause_between_tests)

    gpu_monitor.stop()
    total_time = time.time() - benchmark_start

    if not all_results:
        console.print("[red]No successful tests. Check vLLM server.[/red]")
        sys.exit(1)

    # ---- Statistical aggregation (multi-iteration) ----
    if iterations > 1:
        from vllm_benchmark.analysis.statistics import aggregate_iterations
        # Group raw results into per-iteration lists
        iter_groups: dict[int, list[dict]] = {}
        for r in all_results:
            it = r.get("iteration", 0)
            iter_groups.setdefault(it, []).append(r)
        iteration_runs = list(iter_groups.values())
        aggregated = aggregate_iterations(iteration_runs)
        console.print(f"\n[bold]Aggregated {len(all_results)} runs into {len(aggregated)} configs with 95% CIs[/bold]")
        # Use aggregated results for scoring/reporting, keep raw for JSON
        reporting_results = aggregated
    else:
        reporting_results = all_results

    # ---- Save results ----
    from vllm_benchmark.reports.charts import ensure_output_directory, sanitize_filename

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = sanitize_filename(model_name)
    output_path = ensure_output_directory(config.output_dir)

    # Capture full environment fingerprint
    from vllm_benchmark.core.server import capture_environment
    environment = capture_environment(server_info)

    metadata = {
        "timestamp": timestamp,
        "benchmark_duration": total_time,
        "system_info": system_info,
        "server_info": server_info,
        "environment": environment,
        "configuration": {
            "context_lengths": config.context_lengths,
            "concurrent_users": config.concurrency_levels,
            "output_tokens": config.output_tokens,
            "prompt_types": config.prompt_types,
            "streaming": config.streaming,
            "iterations": iterations,
            "seed": args.seed,
        },
    }

    results_package = {"metadata": metadata, "results": all_results}
    json_file = output_path / f"benchmark_{safe_model}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results_package, f, indent=2)

    df = pd.DataFrame(all_results)
    csv_file = output_path / f"benchmark_{safe_model}_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    # ---- Score ----
    from vllm_benchmark.analysis.scoring import VLLMScore
    scorer = VLLMScore()
    score = scorer.calculate(reporting_results, gpu_name=system_info.get("gpu_name"))

    console.print(f"\n{'=' * 80}")
    console.print(Panel(scorer.format_score_display(score), title="[bold]vLLM Benchmark Score[/bold]", border_style="cyan"))

    # ---- Publish result ----
    from vllm_benchmark.analysis.publish import create_result_entry, save_result
    entry = create_result_entry(reporting_results, metadata, score)
    if entry:
        result_file = save_result(entry, config.output_dir)
        console.print(f"  Result: {result_file}")

    # ---- Statistical warnings ----
    if iterations > 1:
        from vllm_benchmark.analysis.statistics import compute_robust_stats
        tps_values = [r["tokens_per_second"] for r in all_results]
        tps_stats = compute_robust_stats(tps_values)
        if tps_stats.get("warnings"):
            console.print("\n[bold yellow]Statistical warnings:[/bold yellow]")
            for w in tps_stats["warnings"]:
                console.print(f"  [yellow]WARNING[/yellow] {w}")
        console.print(
            f"\n[bold]Throughput (95% CI):[/bold] "
            f"{tps_stats['mean']:.1f} tok/s "
            f"[{tps_stats['ci_lower']:.1f}, {tps_stats['ci_upper']:.1f}] "
            f"(CV={tps_stats['cv']:.3f}, n={tps_stats['n']})"
        )

    # ---- Diagnostics ----
    from vllm_benchmark.analysis.diagnostics import DiagnosticEngine
    engine = DiagnosticEngine()
    diagnostics = engine.analyze(reporting_results, server_info)

    if diagnostics:
        console.print("\n[bold]Diagnostics:[/bold]")
        for d in diagnostics:
            severity_icons = {
                "critical": "[bold red]CRITICAL[/]", "warning": "[yellow]WARNING[/]",
                "success": "[green]OK[/]", "info": "[blue]INFO[/]",
            }
            icon = severity_icons.get(d.severity, "[dim]?[/]")
            console.print(f"  {icon} [bold]{d.title}[/bold]: {d.message}")

    # ---- Summary table ----
    console.print(f"\n{'=' * 80}")
    summary_table = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE, title="Performance Highlights")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow", justify="right")
    summary_table.add_column("Configuration", style="green")

    rdf = pd.DataFrame(reporting_results)
    max_tp = rdf.loc[rdf["tokens_per_second"].idxmax()]
    tp_val = f"{max_tp['tokens_per_second']:.1f} tok/s"
    if "tokens_per_second_ci_lower" in max_tp:
        tp_val += f" [{max_tp['tokens_per_second_ci_lower']:.1f}, {max_tp['tokens_per_second_ci_upper']:.1f}]"
    summary_table.add_row("Peak Throughput", tp_val,
                          f"{int(max_tp['concurrent_users'])}u @ {max_tp['context_length'] // 1000}K")
    best_eff = rdf.loc[rdf["throughput_per_user"].idxmax()]
    summary_table.add_row("Best Efficiency", f"{best_eff['throughput_per_user']:.1f} tok/s/user",
                          f"{int(best_eff['concurrent_users'])}u @ {best_eff['context_length'] // 1000}K")
    min_lat = rdf.loc[rdf["avg_latency"].idxmin()]
    lat_val = f"{min_lat['avg_latency']:.2f}s"
    if "avg_latency_ci_lower" in min_lat:
        lat_val += f" [{min_lat['avg_latency_ci_lower']:.2f}, {min_lat['avg_latency_ci_upper']:.2f}]"
    summary_table.add_row("Lowest Latency", lat_val,
                          f"{int(min_lat['concurrent_users'])}u @ {min_lat['context_length'] // 1000}K")
    costs = [r.get("cost_per_1m_tokens") for r in reporting_results if r.get("cost_per_1m_tokens")]
    if costs:
        best_cost = min(costs)
        summary_table.add_row("Cost / 1M tokens", f"${best_cost:.4f}", "Best configuration")
    console.print(summary_table)

    # ---- Charts ----
    if config.generate_charts:
        console.print("\n[yellow]Generating charts...[/yellow]")
        from vllm_benchmark.reports.charts import visualize_results
        viz_file = visualize_results(reporting_results, model_name, system_info, server_info, config.output_tokens, config.output_dir)
        console.print(f"  [green]Charts:[/green] {viz_file}")

    # ---- HTML report ----
    if config.generate_html:
        console.print("[yellow]Generating HTML report...[/yellow]")
        from vllm_benchmark.reports.html_report import generate_html_report
        html_file = generate_html_report(all_results, metadata, score, diagnostics, config.output_dir)
        console.print(f"  [green]HTML:[/green] {html_file}")

    # ---- Regression detection ----
    if config.compare_file:
        console.print(f"\n[yellow]Comparing with {config.compare_file}...[/yellow]")
        from vllm_benchmark.analysis.regression import RegressionDetector
        detector = RegressionDetector()
        try:
            regressions = detector.compare(all_results, config.compare_file)
            console.print(detector.format_report(regressions))
        except Exception as e:
            console.print(f"[red]Regression detection failed: {e}[/red]")

    # ---- Methodology note ----
    console.print(f"\n[dim]Methodology: async concurrent requests via aiohttp, "
                  f"{'streaming SSE' if config.streaming else 'batch'} mode, "
                  f"{iterations} iteration(s), "
                  f"{'seeded' if args.seed else 'unseeded'}. "
                  f"Environment fingerprint: {environment.get('fingerprint', 'N/A')[:12]}[/dim]")

    # ---- Output summary ----
    console.print("\n[bold cyan]Outputs:[/bold cyan]")
    console.print(f"  JSON: {json_file}")
    console.print(f"  CSV:  {csv_file}")

    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print(f"[bold cyan]Benchmark complete in {total_time / 60:.1f} minutes[/bold cyan]")
    console.print(f"[bold green]{'=' * 80}[/bold green]\n")
