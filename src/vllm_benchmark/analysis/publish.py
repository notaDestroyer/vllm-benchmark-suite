"""Benchmark result publishing and community leaderboard.

Saves structured results for sharing and comparison. Results can be
contributed back to the repository for a community leaderboard.

Author: amit
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def create_result_entry(
    results: List[Dict],
    metadata: Dict,
    score: Optional[object] = None,
) -> Dict:
    """Create a standardized result entry for publishing.

    Args:
        results: List of benchmark result dicts.
        metadata: Benchmark metadata (system_info, server_info, etc.).
        score: Optional ScoreBreakdown object.

    Returns:
        Structured dict ready for JSON serialization.
    """
    system = metadata.get("system_info", {})
    server = metadata.get("server_info", {})

    # Extract peak metrics
    successful = [r for r in results if r.get("tokens_per_second")]
    if not successful:
        return {}

    peak_tps = max(r["tokens_per_second"] for r in successful)
    best_latency = min(r["avg_latency"] for r in successful)
    best_ttft = min(r.get("ttft_estimate", 999) for r in successful)

    # Cost if available
    costs = [r["cost_per_1m_tokens"] for r in successful if r.get("cost_per_1m_tokens")]
    best_cost = min(costs) if costs else None

    entry = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": system.get("gpu_name", "Unknown"),
            "gpu_count": system.get("gpu_count", 1),
            "vram_gb": system.get("total_vram_gb"),
            "cpu": system.get("cpu_model", "Unknown"),
            "ram_gb": system.get("total_ram_gb"),
        },
        "software": {
            "model": server.get("model_name", "Unknown"),
            "vllm_version": server.get("version"),
            "quantization": server.get("quantization"),
            "tensor_parallel": server.get("tensor_parallel_size"),
        },
        "results": {
            "peak_throughput_tps": round(peak_tps, 1),
            "best_latency_s": round(best_latency, 3),
            "best_ttft_ms": round(best_ttft * 1000, 1),
            "cost_per_1m_tokens_usd": round(best_cost, 4) if best_cost else None,
        },
        "test_matrix": {
            "context_lengths": sorted(set(r.get("context_length", 0) for r in successful)),
            "concurrency_levels": sorted(set(r.get("concurrent_users", 0) for r in successful)),
            "num_tests": len(successful),
        },
    }

    if score:
        entry["score"] = {
            "overall": score.overall,
            "grade": score.grade,
            "throughput": score.throughput,
            "latency": score.latency,
            "efficiency": score.efficiency,
            "energy": score.energy,
            "consistency": score.consistency,
        }

    # Generate a fingerprint for deduplication
    fp_str = f"{entry['hardware']['gpu']}:{entry['software']['model']}:{peak_tps:.0f}"
    entry["fingerprint"] = hashlib.sha256(fp_str.encode()).hexdigest()[:12]

    return entry


def save_result(entry: Dict, output_dir: str = "./outputs") -> str:
    """Save a result entry to a JSON file.

    Returns:
        Path to the saved file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gpu = (entry.get("hardware") or {}).get("gpu") or "unknown"
    model = (entry.get("software") or {}).get("model") or "unknown"
    gpu = gpu.replace(" ", "_")
    model = model.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = out / f"result_{gpu}_{model}_{ts}.json"
    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2)

    return str(filepath)


def format_leaderboard_row(entry: Dict) -> str:
    """Format a result entry as a markdown table row."""
    hw = entry.get("hardware", {})
    sw = entry.get("software", {})
    res = entry.get("results", {})
    sc = entry.get("score", {})

    gpu = hw.get("gpu", "?")
    model = sw.get("model", "?")
    if len(model) > 30:
        model = model[:27] + "..."
    tps = res.get("peak_throughput_tps", 0)
    ttft = res.get("best_ttft_ms", 0)
    cost = res.get("cost_per_1m_tokens_usd")
    cost_str = f"${cost:.4f}" if cost else "N/A"
    grade = sc.get("grade", "-")
    overall = sc.get("overall", 0)

    return f"| {gpu} | {model} | {tps:,.0f} | {ttft:.0f}ms | {cost_str} | {grade} ({overall:,}) |"


def generate_leaderboard_md(results_dir: str = "./outputs") -> str:
    """Generate a markdown leaderboard from all result files in a directory.

    Returns:
        Markdown string with the leaderboard table.
    """
    results_path = Path(results_dir)
    entries: list[Dict] = []

    for f in results_path.glob("result_*.json"):
        try:
            with open(f) as fh:
                entries.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue

    if not entries:
        return "No benchmark results found.\n"

    # Sort by peak throughput descending
    entries.sort(key=lambda e: e.get("results", {}).get("peak_throughput_tps", 0), reverse=True)

    lines = [
        "# vLLM Benchmark Leaderboard",
        "",
        "| GPU | Model | Peak tok/s | Best TTFT | Cost/1M tok | Score |",
        "|-----|-------|-----------|-----------|-------------|-------|",
    ]
    for entry in entries:
        lines.append(format_leaderboard_row(entry))

    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC*")
    lines.append("")

    return "\n".join(lines)
