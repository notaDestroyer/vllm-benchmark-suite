"""Governing-bottleneck detection for benchmark cells.

Given a per-cell benchmark result, a :class:`ModelProfile`, a GPU spec
and (optionally) server-side metrics, this module decides *what is
limiting performance* and *which lever to pull*.

The classifier follows a strict **precedence hierarchy** so that a
single governing bottleneck is reported per cell:

1. ``kv_capacity`` — the KV cache is saturated (near-full or preemptions
   observed); nothing else matters until you make room.
2. ``queue`` — requests are waiting / tail latency explodes; the server
   is admission-limited.
3. ``interconnect`` — multi-GPU tensor parallelism scales sublinearly
   beyond what the roofline predicts.
4. **physics** — otherwise place the dominant phase (prefill vs decode)
   on the roofline and pick the governing resource:
   ``prefill_compute``, ``decode_weight_bandwidth``,
   ``decode_kv_bandwidth`` or ``decode_compute``.

All roofline numbers come from :mod:`vllm_benchmark.analysis.model_intel`.
Every metric is ``None`` when its inputs are missing — the classifier
never fabricates a verdict it cannot support and lowers ``confidence``
accordingly.

Author: amit
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from vllm_benchmark.analysis.model_intel import (
    ModelProfile,
    bytes_per_param,
    critical_batch,
    mbu,
    mfu,
)

Primary = Literal[
    "prefill_compute",
    "decode_weight_bandwidth",
    "decode_kv_bandwidth",
    "decode_compute",
    "kv_capacity",
    "queue",
    "interconnect",
    "unknown",
]

#: Actionable lever per primary bottleneck.
_LEVERS: dict[str, str] = {
    "prefill_compute": "increase compute/FLOPs (faster GPU, FP8, or shard the prefill)",
    "decode_weight_bandwidth": "increase memory bandwidth (or enable FP8 KV cache)",
    "decode_kv_bandwidth": "increase memory bandwidth (or enable FP8 KV cache)",
    "decode_compute": "already compute-bound; add GPUs/replicas for more throughput",
    "kv_capacity": "add VRAM, reduce context/concurrency, or enable FP8 KV",
    "queue": "raise max_num_seqs or add a replica",
    "interconnect": "reduce TP degree or improve interconnect (NVLink)",
    "unknown": "collect streaming pp/tg metrics and a known model profile",
}


@dataclass
class BottleneckVerdict:
    """The governing bottleneck for a single benchmark cell."""

    cell: tuple
    primary: Primary
    contributions: dict[str, float]
    mbu: Optional[float] = None
    mfu: Optional[float] = None
    critical_batch: Optional[int] = None
    headroom_to_ridge: Optional[float] = None
    prefill_time_frac: Optional[float] = None
    analytic_vs_empirical_agree: Optional[bool] = None
    lever: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation of this verdict."""
        return {
            "cell": list(self.cell),
            "primary": self.primary,
            "contributions": dict(self.contributions),
            "mbu": self.mbu,
            "mfu": self.mfu,
            "critical_batch": self.critical_batch,
            "headroom_to_ridge": self.headroom_to_ridge,
            "prefill_time_frac": self.prefill_time_frac,
            "analytic_vs_empirical_agree": self.analytic_vs_empirical_agree,
            "lever": self.lever,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Batch-scaling regime (pure)
# ---------------------------------------------------------------------------

def batch_scaling_regime(
    concurrencies: list[float],
    throughputs: list[float],
) -> Literal["linear", "saturating", "collapsing"]:
    """Classify how aggregate throughput scales with concurrency.

    Compares the throughput gain over the measured concurrency span to
    the ideal linear gain:

    * ``"collapsing"`` — peak throughput is below the lowest-concurrency
      throughput (overload / thrashing).
    * ``"linear"`` — realized speedup is >= 70% of the ideal ratio
      (bandwidth-bound regime; more requests ~= more tokens/s).
    * ``"saturating"`` — otherwise (compute-bound; extra concurrency adds
      little throughput).

    Requires at least two points; degenerate input returns ``"linear"``.
    """
    if len(concurrencies) < 2 or len(throughputs) < 2:
        return "linear"
    pairs = sorted(zip(concurrencies, throughputs), key=lambda p: p[0])
    c0, t0 = pairs[0]
    cmax, _ = pairs[-1]
    peak = max(t for _, t in pairs)

    _, tmax = pairs[-1]
    if t0 <= 0 or c0 <= 0 or cmax <= c0:
        return "linear"
    # Collapse: throughput at the highest concurrency has fallen below the
    # lowest-concurrency throughput (the system is thrashing under load).
    if tmax < t0 * 0.95:
        return "collapsing"

    ideal_ratio = cmax / c0
    actual_ratio = peak / t0
    # Fraction of ideal linear scaling that was realized.
    realized = (actual_ratio - 1.0) / (ideal_ratio - 1.0) if ideal_ratio > 1 else 0.0
    if realized >= 0.70:
        return "linear"
    return "saturating"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(contributions: dict[str, float]) -> dict[str, float]:
    """Normalize a contributions dict to sum to 1.0 (empty stays empty)."""
    total = sum(v for v in contributions.values() if v > 0)
    if total <= 0:
        return {k: 0.0 for k in contributions}
    return {k: max(0.0, v) / total for k, v in contributions.items()}


def _prefill_time_frac(
    cell: dict,
    profile: ModelProfile,
) -> Optional[float]:
    """Estimate the fraction of end-to-end time spent in prefill.

    Uses measured prefill/decode throughput and token counts::

        t_prefill = prompt_tokens / prefill_tps
        t_decode  = completion_tokens / decode_tps
        frac      = t_prefill / (t_prefill + t_decode)

    Returns ``None`` if the needed measurements are absent.
    """
    prefill_tps = cell.get("prefill_tps_mean") or cell.get("prefill_tps")
    decode_tps = cell.get("decode_tps_mean") or cell.get("decode_tps") or cell.get("decode_tps_p50")
    prompt_tokens = cell.get("avg_prompt_tokens") or cell.get("context_length")
    completion_tokens = cell.get("avg_completion_tokens")

    if not prefill_tps or not decode_tps or not prompt_tokens or not completion_tokens:
        return None
    if prefill_tps <= 0 or decode_tps <= 0:
        return None
    t_prefill = prompt_tokens / prefill_tps
    t_decode = completion_tokens / decode_tps
    if t_prefill + t_decode <= 0:
        return None
    return t_prefill / (t_prefill + t_decode)


def _confidence_from_cv(cell: dict) -> Literal["high", "medium", "low"]:
    """Map measured latency coefficient-of-variation to a confidence band."""
    avg = cell.get("avg_latency")
    std = cell.get("std_latency")
    if avg and std is not None and avg > 0:
        cv = std / avg
        if cv < 0.10:
            return "high"
        if cv < 0.30:
            return "medium"
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_cell(
    result_cell: dict,
    profile: Optional[ModelProfile],
    gpu_spec: Optional[dict],
    server_metrics: Optional[dict],
    *,
    batch_scaling: Optional[str] = None,
) -> BottleneckVerdict:
    """Classify the governing bottleneck for a single benchmark cell.

    See the module docstring for the precedence hierarchy.

    Args:
        result_cell: A per-cell result dict (context_length,
            concurrent_users, prompt_type, throughputs, latencies, ...).
        profile: The model profile (may be ``None`` -> ``unknown``).
        gpu_spec: GPU spec dict (``hbm_bandwidth_gbps``,
            ``peak_flops_tflops``) or ``None``.
        server_metrics: Optional server-side metrics
            (``gpu_cache_usage_perc``, ``num_requests_waiting``,
            ``preemptions``, ...).
        batch_scaling: Optional regime string for this cell's
            (context, prompt_type) group, from :func:`classify_run`.

    Returns:
        A :class:`BottleneckVerdict`.
    """
    cell_key = (
        result_cell.get("context_length"),
        result_cell.get("concurrent_users"),
        result_cell.get("prompt_type"),
    )
    notes: list[str] = []
    server_metrics = server_metrics or {}

    profile = profile or ModelProfile(name="unknown")
    seq_len = result_cell.get("context_length")
    concurrency = result_cell.get("concurrent_users") or 1
    tensor_parallel = None

    # --- roofline scalars (None-safe) ---
    hbm = gpu_spec.get("hbm_bandwidth_gbps") if gpu_spec else None
    peak_map = gpu_spec.get("peak_flops_tflops", {}) if gpu_spec else {}
    quant = None
    for n in profile.notes:
        if "quantization=" in n:
            quant = n.split("quantization=")[-1].rstrip(".")
    dtype_key = "fp8" if quant and "fp8" in quant.lower() else "bf16"
    peak_flops = peak_map.get(dtype_key) or peak_map.get("bf16")
    bpp = bytes_per_param(quant)

    decode_tps_1u = result_cell.get("decode_tps_mean") or result_cell.get("decode_tps_p50")
    prefill_tps = result_cell.get("prefill_tps_mean") or result_cell.get("prefill_tps_p50")

    mbu_val = mbu(
        decode_tps_1u, profile.active_params, profile.kv_bytes_per_token,
        seq_len, hbm, bpp,
    ) if concurrency == 1 else None
    mfu_val = mfu(prefill_tps, profile.active_params, peak_flops)
    bstar = critical_batch(peak_flops, hbm, bpp)

    headroom: Optional[float] = None
    if bstar is not None and concurrency:
        headroom = bstar - concurrency

    prefill_frac = _prefill_time_frac(result_cell, profile)

    # ------------------------------------------------------------------
    # Precedence 1: KV capacity
    # ------------------------------------------------------------------
    cache_usage = (
        server_metrics.get("gpu_cache_usage_perc")
        if server_metrics.get("gpu_cache_usage_perc") is not None
        else result_cell.get("gpu_cache_usage_perc")
    )
    preemptions = server_metrics.get("preemptions") or result_cell.get("preemptions") or 0
    if (cache_usage is not None and cache_usage >= 0.95) or (preemptions and preemptions > 0):
        notes.append("KV cache saturated or preemptions observed.")
        return BottleneckVerdict(
            cell=cell_key,
            primary="kv_capacity",
            contributions=_normalize({"kv_capacity": 1.0}),
            mbu=mbu_val, mfu=mfu_val, critical_batch=bstar,
            headroom_to_ridge=headroom, prefill_time_frac=prefill_frac,
            lever=_LEVERS["kv_capacity"],
            confidence="high",
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Precedence 2: Queue / admission
    # ------------------------------------------------------------------
    waiting = server_metrics.get("num_requests_waiting") or result_cell.get("num_requests_waiting") or 0
    p99 = result_cell.get("latency_p99")
    p50 = result_cell.get("latency_p50") or result_cell.get("avg_latency")
    tail_ratio = (p99 / p50) if (p99 and p50 and p50 > 0) else None
    if (waiting and waiting > 0) or (tail_ratio is not None and tail_ratio > 5.0):
        notes.append(
            f"Queueing detected (waiting={waiting}, p99/p50="
            f"{tail_ratio:.1f})." if tail_ratio else "Requests waiting in queue."
        )
        return BottleneckVerdict(
            cell=cell_key,
            primary="queue",
            contributions=_normalize({"queue": 1.0}),
            mbu=mbu_val, mfu=mfu_val, critical_batch=bstar,
            headroom_to_ridge=headroom, prefill_time_frac=prefill_frac,
            lever=_LEVERS["queue"],
            confidence="high" if (waiting and waiting > 0) else "medium",
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Precedence 3: Interconnect (multi-GPU sublinear scaling)
    # ------------------------------------------------------------------
    tensor_parallel = server_metrics.get("tensor_parallel")
    if (
        tensor_parallel and tensor_parallel > 1
        and batch_scaling == "saturating"
        and bstar is not None and concurrency < bstar
    ):
        # Below B* we *expect* linear (bandwidth) scaling; saturation here
        # points at interconnect overhead rather than compute saturation.
        notes.append(
            "Sublinear scaling below the roofline ridge with TP>1 suggests "
            "interconnect overhead."
        )
        return BottleneckVerdict(
            cell=cell_key,
            primary="interconnect",
            contributions=_normalize({"interconnect": 0.6, "decode_weight_bandwidth": 0.4}),
            mbu=mbu_val, mfu=mfu_val, critical_batch=bstar,
            headroom_to_ridge=headroom, prefill_time_frac=prefill_frac,
            lever=_LEVERS["interconnect"],
            confidence="medium",
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Precedence 4: Physics (roofline placement)
    # ------------------------------------------------------------------
    if profile.active_params is None or peak_flops is None or hbm is None:
        notes.append("Insufficient model/GPU data for roofline placement.")
        return BottleneckVerdict(
            cell=cell_key,
            primary="unknown",
            contributions=_normalize({"unknown": 1.0}),
            mbu=mbu_val, mfu=mfu_val, critical_batch=bstar,
            headroom_to_ridge=headroom, prefill_time_frac=prefill_frac,
            analytic_vs_empirical_agree=None,
            lever=_LEVERS["unknown"],
            confidence="low",
            notes=notes,
        )

    confidence = _confidence_from_cv(result_cell)

    # Decide dominant phase. Default to decode unless prefill dominates time.
    prefill_dominates = prefill_frac is not None and prefill_frac >= 0.5

    primary: Primary
    contributions: dict[str, float]

    if prefill_dominates:
        primary = "prefill_compute"
        contributions = {
            "prefill_compute": prefill_frac,
            "decode_weight_bandwidth": 1.0 - prefill_frac,
        }
    else:
        # Decode-dominated: choose among the three decode regimes.
        if bstar is not None and concurrency >= bstar:
            primary = "decode_compute"
        else:
            # Below the ridge -> bandwidth-bound. Weight vs KV split by
            # which dominates bytes-per-token at this sequence length.
            weight_bytes = (profile.active_params or 0) * bpp
            kv_bytes = (profile.kv_bytes_per_token or 0) * (seq_len or 0)
            if kv_bytes > weight_bytes and kv_bytes > 0:
                primary = "decode_kv_bandwidth"
            else:
                primary = "decode_weight_bandwidth"
        # Build contributions from the (1 - prefill_frac) decode share.
        decode_share = 1.0 - (prefill_frac or 0.0)
        prefill_share = prefill_frac or 0.0
        weight_bytes = (profile.active_params or 0) * bpp
        kv_bytes = (profile.kv_bytes_per_token or 0) * (seq_len or 0)
        denom = weight_bytes + kv_bytes
        if primary == "decode_compute":
            contributions = {
                "decode_compute": decode_share,
                "prefill_compute": prefill_share,
            }
        elif denom > 0:
            contributions = {
                "decode_weight_bandwidth": decode_share * weight_bytes / denom,
                "decode_kv_bandwidth": decode_share * kv_bytes / denom,
                "prefill_compute": prefill_share,
            }
        else:
            contributions = {primary: decode_share, "prefill_compute": prefill_share}

    # Analytic-vs-empirical agreement when batch_scaling is provided.
    agree: Optional[bool] = None
    if batch_scaling is not None and bstar is not None and concurrency:
        analytic_bandwidth_bound = concurrency < bstar
        empirical_bandwidth_bound = batch_scaling == "linear"
        agree = analytic_bandwidth_bound == empirical_bandwidth_bound
        if not agree:
            confidence = "low"
            notes.append(
                "Analytic roofline placement disagrees with empirical batch "
                "scaling; lowering confidence."
            )

    return BottleneckVerdict(
        cell=cell_key,
        primary=primary,
        contributions=_normalize(contributions),
        mbu=mbu_val, mfu=mfu_val, critical_batch=bstar,
        headroom_to_ridge=headroom, prefill_time_frac=prefill_frac,
        analytic_vs_empirical_agree=agree,
        lever=_LEVERS[primary],
        confidence=confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Whole-run classification
# ---------------------------------------------------------------------------

def classify_run(
    results: list[dict],
    profile: Optional[ModelProfile],
    gpu_spec: Optional[dict],
) -> list[BottleneckVerdict]:
    """Classify every cell in a run, deriving per-group batch scaling.

    Cells are grouped by ``(context_length, prompt_type)``; within each
    group the throughput-vs-concurrency curve determines the scaling
    regime (:func:`batch_scaling_regime`) which is fed to
    :func:`classify_cell` so it can flag interconnect / agreement.

    Args:
        results: All per-cell result dicts from a benchmark run.
        profile: The model profile.
        gpu_spec: The GPU spec dict.

    Returns:
        One :class:`BottleneckVerdict` per input cell, in input order.
    """
    # Build batch-scaling regime per (context, prompt_type).
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r.get("context_length"), r.get("prompt_type"))
        groups.setdefault(key, []).append(r)

    regimes: dict[tuple, str] = {}
    for key, cells in groups.items():
        concs = [c.get("concurrent_users") or 0 for c in cells]
        tps = [c.get("tokens_per_second") or 0.0 for c in cells]
        regimes[key] = batch_scaling_regime(concs, tps)

    verdicts: list[BottleneckVerdict] = []
    for r in results:
        key = (r.get("context_length"), r.get("prompt_type"))
        server_metrics = {
            k: r.get(k)
            for k in ("gpu_cache_usage_perc", "num_requests_waiting", "preemptions", "tensor_parallel")
            if r.get(k) is not None
        }
        verdicts.append(
            classify_cell(
                r, profile, gpu_spec, server_metrics or None,
                batch_scaling=regimes.get(key),
            )
        )
    return verdicts
