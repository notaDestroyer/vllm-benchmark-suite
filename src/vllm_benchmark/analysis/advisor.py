"""Operating-point and configuration advisor.

Turns a benchmark run plus a :class:`ModelProfile` / GPU spec into
actionable guidance:

* knee detection (an in-tree Kneedle implementation) to find the
  latency-optimal and throughput-optimal operating points,
* a plain-English throughput explanation grounded in the model's
  active-vs-total parameters and the measured MBU/MFU, and
* configuration tips (MoE expert parallelism, GQA/MQA context headroom,
  FP8 quantization on FP8-capable GPUs, prefix caching).

The eight application-fitness profiles are intentionally **not**
implemented here — they are a later PR.  :func:`build_advisory` exposes a
clear extension point (``fitness=None``) for them.

Author: amit
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from vllm_benchmark.analysis.model_intel import ModelProfile

# ---------------------------------------------------------------------------
# Kneedle (in-tree)
# ---------------------------------------------------------------------------

def find_knee(
    x: list[float],
    y: list[float],
    curve: Literal["concave", "convex"] = "concave",
    direction: Literal["increasing", "decreasing"] = "increasing",
) -> Optional[float]:
    """Find the knee/elbow x-value of a curve (Kneedle algorithm).

    Normalizes ``(x, y)`` to the unit square, computes the difference
    between the normalized curve and the unit diagonal, and returns the
    ``x`` at which that difference is extremal — the point of maximum
    curvature.  ``curve``/``direction`` orient the difference so the knee
    is a maximum.

    Args:
        x: Monotonically increasing x values.
        y: Corresponding y values.
        curve: ``"concave"`` (diminishing returns) or ``"convex"``.
        direction: ``"increasing"`` or ``"decreasing"`` y trend.

    Returns:
        The x value at the knee, or ``None`` for degenerate input
        (fewer than 3 points or a flat curve).
    """
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    if xmax == xmin or ymax == ymin:
        return None

    xn = [(xi - xmin) / (xmax - xmin) for xi in x]
    yn = [(yi - ymin) / (ymax - ymin) for yi in y]

    # Orient so a concave-increasing curve bulges above the diagonal.
    if curve == "concave" and direction == "increasing":
        diff = [yi - xi for xi, yi in zip(xn, yn)]
    elif curve == "concave" and direction == "decreasing":
        diff = [yi - (1 - xi) for xi, yi in zip(xn, yn)]
    elif curve == "convex" and direction == "increasing":
        diff = [xi - yi for xi, yi in zip(xn, yn)]
    else:  # convex, decreasing
        diff = [(1 - xi) - yi for xi, yi in zip(xn, yn)]

    best_idx = max(range(len(diff)), key=lambda i: diff[i])
    # A truly flat curve produces a knee at an endpoint with ~0 difference.
    if abs(diff[best_idx]) < 1e-9:
        return None
    return x[best_idx]


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------

def recommend_operating_points(results: list[dict]) -> dict[str, Optional[dict]]:
    """Recommend latency-optimal and throughput-optimal cells.

    Aggregates by concurrency (mean across contexts/prompt types) and
    uses the knee of throughput-vs-concurrency for the throughput-optimal
    point and the knee of p95-latency-vs-concurrency for the latency-
    optimal point.  Falls back to the extreme cell when no knee exists.

    Returns:
        ``{"latency_optimal": cell|None, "throughput_optimal": cell|None}``.
    """
    if not results:
        return {"latency_optimal": None, "throughput_optimal": None}

    by_conc: dict[int, list[dict]] = {}
    for r in results:
        c = r.get("concurrent_users")
        if c is not None:
            by_conc.setdefault(c, []).append(r)

    concs = sorted(by_conc)
    if not concs:
        return {"latency_optimal": None, "throughput_optimal": None}

    def _mean(cells: list[dict], key: str) -> float:
        vals = [c[key] for c in cells if c.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    agg_tps = [_mean(by_conc[c], "tokens_per_second") for c in concs]
    agg_lat = [
        _mean(by_conc[c], "latency_p95") or _mean(by_conc[c], "avg_latency")
        for c in concs
    ]

    # Throughput-optimal: knee of throughput vs concurrency.
    tp_knee = find_knee([float(c) for c in concs], agg_tps, "concave", "increasing")
    if tp_knee is None:
        # Fall back to the max-throughput concurrency.
        tp_conc = concs[max(range(len(agg_tps)), key=lambda i: agg_tps[i])]
    else:
        tp_conc = int(tp_knee)
    throughput_cell = max(by_conc[tp_conc], key=lambda c: c.get("tokens_per_second", 0))

    # Latency-optimal: lowest concurrency with acceptable latency (the
    # knee of the latency curve, else the min-latency concurrency).
    lat_knee = find_knee([float(c) for c in concs], agg_lat, "convex", "increasing")
    if lat_knee is None:
        lat_conc = concs[min(range(len(agg_lat)), key=lambda i: agg_lat[i])]
    else:
        lat_conc = int(lat_knee)
    latency_cell = min(
        by_conc[lat_conc],
        key=lambda c: c.get("latency_p95") or c.get("avg_latency") or float("inf"),
    )

    return {"latency_optimal": latency_cell, "throughput_optimal": throughput_cell}


# ---------------------------------------------------------------------------
# Explanations and tips
# ---------------------------------------------------------------------------

def throughput_explanation(
    profile: Optional[ModelProfile],
    top_cell: Optional[dict],
    mbu: Optional[float],
    mfu: Optional[float],
) -> str:
    """Explain the observed throughput in terms of model & roofline.

    Mentions active-vs-total parameters (and the MoE batching caveat),
    then maps MBU/MFU into verdict bands: ``> 0.7`` -> "near-optimal",
    ``< 0.4`` -> "well below ceiling — investigate".
    """
    parts: list[str] = []
    if profile is not None:
        if profile.is_moe and profile.active_params and profile.total_params:
            parts.append(
                f"This is an MoE model: only ~{profile.active_params / 1e9:.1f}B of "
                f"{profile.total_params / 1e9:.1f}B params are active per token, so "
                f"decode is bandwidth-light but benefits strongly from batching "
                f"(more concurrency activates experts more fully)."
            )
        elif profile.active_params:
            parts.append(
                f"Dense model with ~{profile.active_params / 1e9:.1f}B active params "
                f"read from HBM every decode step."
            )

    if top_cell is not None and top_cell.get("tokens_per_second") is not None:
        parts.append(
            f"Peak observed throughput is {top_cell['tokens_per_second']:.0f} tok/s "
            f"at {top_cell.get('concurrent_users')} concurrent users."
        )

    def _band(label: str, val: Optional[float]) -> Optional[str]:
        if val is None:
            return None
        pct = val * 100
        if val > 0.7:
            return f"{label} {pct:.0f}% — near-optimal."
        if val < 0.4:
            return f"{label} {pct:.0f}% — well below ceiling; investigate."
        return f"{label} {pct:.0f}% — moderate utilization."

    mbu_s = _band("MBU", mbu)
    mfu_s = _band("MFU", mfu)
    if mbu_s:
        parts.append(mbu_s)
    if mfu_s:
        parts.append(mfu_s)

    if not parts:
        return "Insufficient data to explain throughput."
    return " ".join(parts)


def config_tips(
    profile: Optional[ModelProfile],
    server_info,
    gpu_spec: Optional[dict],
) -> list[str]:
    """Return actionable configuration tips for the deployment.

    Fires on: MoE models (expert parallelism + higher ``max_num_seqs``),
    GQA/MQA attention (push context/concurrency), a 16-bit model on an
    FP8-capable GPU (quantize for ~2x compute headroom), and prefix
    caching disabled on a cache-friendly workload.
    """
    tips: list[str] = []
    if profile is None:
        return tips

    if profile.is_moe:
        tips.append(
            "MoE model: enable expert parallelism (EP) and raise --max-num-seqs; "
            "experts are underutilized at low concurrency."
        )

    if profile.attention_type in ("GQA", "MQA"):
        tips.append(
            f"{profile.attention_type} attention keeps the KV cache small — you can "
            f"push longer context and higher concurrency before hitting KV limits."
        )

    # FP16/BF16 on an FP8-capable GPU.
    quant = getattr(server_info, "quantization", None)
    is_16bit = not quant or "16" in str(quant).lower() or str(quant).lower() in ("none", "fp16/bf16")
    fp8_capable = bool(gpu_spec and gpu_spec.get("peak_flops_tflops", {}).get("fp8"))
    if is_16bit and fp8_capable:
        peak = gpu_spec["peak_flops_tflops"]
        delta = ""
        if peak.get("fp8") and peak.get("bf16"):
            delta = f" (~{peak['fp8'] / peak['bf16']:.1f}x prefill FLOPs, ~half the weight bytes)"
        tips.append(
            f"GPU supports FP8: serving this 16-bit model in FP8 should improve "
            f"throughput{delta}."
        )

    # Prefix caching off but workload looks cache-friendly.
    prefix_caching = getattr(server_info, "prefix_caching", None)
    if prefix_caching is False:
        tips.append(
            "Prefix caching is disabled — enable it (--enable-prefix-caching) for "
            "workloads with shared/system prompts to cut prefill cost."
        )

    return tips


# ---------------------------------------------------------------------------
# Advisory aggregate
# ---------------------------------------------------------------------------

@dataclass
class Advisory:
    """Aggregate advisory output for a benchmark run.

    ``fitness`` is reserved for the (later-PR) application-fitness
    profiles and is always ``None`` here — the explicit extension point.
    """

    latency_optimal: Optional[dict] = None
    throughput_optimal: Optional[dict] = None
    explanation: str = ""
    tips: list[str] = field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    fitness: Optional[dict] = None  # extension point for app-fitness PR

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation of this advisory."""
        return {
            "latency_optimal": self.latency_optimal,
            "throughput_optimal": self.throughput_optimal,
            "explanation": self.explanation,
            "tips": list(self.tips),
            "confidence": self.confidence,
            "fitness": self.fitness,
        }


def build_advisory(
    results: list[dict],
    profile: Optional[ModelProfile],
    server_info,
    gpu_spec: Optional[dict],
    *,
    mbu: Optional[float] = None,
    mfu: Optional[float] = None,
) -> Advisory:
    """Assemble a full :class:`Advisory` from a run.

    Confidence is taken from the model profile when available
    (``confirmed`` -> high, ``inferred`` -> medium, ``heuristic`` -> low).
    """
    points = recommend_operating_points(results)
    top = points.get("throughput_optimal")
    explanation = throughput_explanation(profile, top, mbu, mfu)
    tips = config_tips(profile, server_info, gpu_spec)

    confidence: Literal["high", "medium", "low"] = "medium"
    if profile is not None:
        confidence = {
            "confirmed": "high",
            "inferred": "medium",
            "heuristic": "low",
        }.get(profile.confidence, "medium")

    return Advisory(
        latency_optimal=points.get("latency_optimal"),
        throughput_optimal=top,
        explanation=explanation,
        tips=tips,
        confidence=confidence,
    )
