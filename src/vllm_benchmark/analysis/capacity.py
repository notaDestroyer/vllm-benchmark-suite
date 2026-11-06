"""Capacity, ceilings and "how to get the most out of this model on this GPU".

Everything here is arithmetic on facts the suite already measures or derives:
the model profile (parameter counts, KV bytes per token), the GPU spec
(bandwidth, dense tensor peaks per format, memory), the server's own KV-cache
capacity when it publishes it, and the per-cell measurements.  No numbers are
invented; every projection names the formula and the inputs it used so the
report can show its work.

Scientific basis
----------------

* **Roofline model** (Williams, Waterman & Patterson, *CACM* 2009): attainable
  throughput is bounded by ``min(peak compute, bandwidth x arithmetic
  intensity)``.  For transformer decode the arithmetic intensity grows with
  batch size, so the aggregate decode ceiling at batch *N* is
  ``N * BW / (W + N * KV * L)`` tokens/s, capped by the compute ceiling.
  ``W`` is the weight bytes streamed per step, ``KV * L`` the KV-cache bytes
  read per sequence at context ``L`` (Pope et al., *Efficiently Scaling
  Transformer Inference*, 2022).
* **FLOPs per token ~= 2 x active parameters** (Kaplan et al., *Scaling Laws
  for Neural Language Models*, 2020), which gives the prefill ceiling
  ``peak_FLOPs / (2 * P_active)`` and the decode compute ceiling.
* **MBU / MFU** as defined in Databricks' *LLM Inference Performance
  Engineering: Best Practices* (2023): fraction of peak bandwidth (decode) or
  peak FLOPs (prefill) actually achieved.
* **KV-cache capacity**: vLLM allocates ``num_gpu_blocks * block_size``
  tokens after profiling (``vllm:cache_config_info``); when that is available
  the budget is *measured*, otherwise it is estimated as
  ``(memory * utilization - weights) / KV_bytes_per_token`` with the
  activation reserve vLLM's profiler would subtract left as a stated caveat.
* **Operating point**: the concurrency "sweet spot" is where added users stop
  buying throughput, read from the measured curve against the roofline's
  critical batch ``B* = peak_FLOPs / (BW * bytes_per_param) / 2``.

Author: amit
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from vllm_benchmark.analysis.bottleneck import peak_dtype_for
from vllm_benchmark.analysis.model_intel import ModelProfile, bytes_per_param, critical_batch

#: vLLM's default ``--gpu-memory-utilization``.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
#: Fraction of the post-weights budget vLLM's profiler typically reserves for
#: activations / CUDA graphs when estimating (measured budgets ignore this).
ACTIVATION_RESERVE_FRACTION = 0.08
#: Added users are "worth it" while each buys at least this fraction of the
#: throughput a perfectly linear scale-up would add.
SWEET_SPOT_MARGINAL_GAIN = 0.50

GB = 1e9


@dataclass
class Footprint:
    """Static memory accounting for the served model on one GPU."""

    quantization: Optional[str]
    bytes_per_param: float
    weight_bytes_total: Optional[int]
    weight_bytes_per_gpu: Optional[int]
    kv_bytes_per_token_total: Optional[int]
    kv_bytes_per_token_per_gpu: Optional[int]
    kv_cache_dtype: Optional[str]
    gpu_memory_bytes: Optional[int]
    gpu_memory_utilization: float
    tensor_parallel: int
    pipeline_parallel: int
    kv_budget_bytes_per_gpu: Optional[int]
    kv_capacity_tokens: Optional[int]
    kv_capacity_source: str  # "server" | "estimated" | "unavailable"
    fits_on_one_gpu: Optional[bool]
    notes: list[str] = field(default_factory=list)


@dataclass
class CellCeiling:
    """Roofline ceiling and attainment for one measured cell."""

    context_length: int
    concurrent_users: int
    prompt_type: Optional[str]
    decode_ceiling_tps: Optional[float]  # aggregate decode tok/s at this N
    decode_measured_tps: Optional[float]
    decode_measured_source: Optional[str]  # "decode_tps_mean x users" | "tokens_per_second"
    decode_attainment: Optional[float]  # measured / ceiling
    bound_by: Optional[str]  # "bandwidth" | "compute"
    prefill_ceiling_tps: Optional[float]
    prefill_measured_tps: Optional[float]
    prefill_attainment: Optional[float]


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _peak_for(gpu_spec: Optional[dict], quant: Optional[str]) -> Optional[float]:
    peaks = (gpu_spec or {}).get("peak_flops_tflops") or {}
    key = peak_dtype_for(quant)
    return _num(peaks.get(key)) or _num(peaks.get("fp8")) or _num(peaks.get("bf16"))


def _gpu_memory_bytes(gpu_spec: Optional[dict], system_info: Optional[dict]) -> Optional[int]:
    dev = (gpu_spec or {}).get("_device") or {}
    mb = _num(dev.get("memory_total_mb"))
    if mb:
        return int(mb * 1024 * 1024)
    sysinfo = system_info or {}
    total_gb = _num(sysinfo.get("total_vram_gb"))
    count = int(sysinfo.get("gpu_count") or 1) or 1
    if total_gb:
        return int(total_gb / count * 1024 ** 3)
    return None


def _kv_dtype_bytes(server_info: Any) -> float:
    kv_dtype = getattr(server_info, "kv_cache_dtype", None)
    return bytes_per_param(kv_dtype) if kv_dtype and kv_dtype != "auto" else 2.0


def compute_footprint(
    profile: Optional[ModelProfile],
    gpu_spec: Optional[dict],
    server_info: Any,
    system_info: Optional[dict],
    *,
    measured_kv_capacity_tokens: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
) -> Footprint:
    """Weights, KV bytes and the KV-token budget for the served configuration.

    Tensor parallelism shards both weights and KV cache evenly across GPUs;
    pipeline parallelism shards layers.  Per-GPU figures divide by
    ``TP * PP``.  The KV budget is measured when the server reports it
    (vLLM ``cache_config_info``, SGLang ``max_total_num_tokens``) and
    estimated from memory otherwise.
    """
    quant = getattr(server_info, "quantization", None)
    bpp = bytes_per_param(quant)
    tp = max(1, int(getattr(server_info, "tensor_parallel", None) or 1))
    pp = max(1, int(getattr(server_info, "pipeline_parallel", None) or 1))
    shards = tp * pp
    util = gpu_memory_utilization or DEFAULT_GPU_MEMORY_UTILIZATION
    notes: list[str] = []

    weight_total = None
    if profile is not None and profile.total_params:
        weight_total = int(profile.total_params * bpp)
    elif profile is not None and profile.weight_bytes:
        weight_total = int(profile.weight_bytes)

    kv_dtype = getattr(server_info, "kv_cache_dtype", None)
    kv_total = int(profile.kv_bytes_per_token) if (profile and profile.kv_bytes_per_token) else None
    if kv_total is not None and kv_dtype and kv_dtype != "auto":
        # profile.kv_bytes_per_token already used the server kv dtype when
        # available; nothing to rescale.  Keep the label for the report.
        pass

    mem = _gpu_memory_bytes(gpu_spec, system_info)
    weight_per_gpu = int(weight_total / shards) if weight_total else None
    kv_per_gpu = int(kv_total / shards) if kv_total else None

    budget = None
    fits = None
    if mem and weight_per_gpu is not None:
        usable = int(mem * util)
        fits = weight_per_gpu < usable
        budget = max(0, int((usable - weight_per_gpu) * (1 - ACTIVATION_RESERVE_FRACTION)))

    if measured_kv_capacity_tokens:
        capacity = int(measured_kv_capacity_tokens)
        source = "server"
    elif budget is not None and kv_per_gpu:
        capacity = int(budget / kv_per_gpu)
        source = "estimated"
        notes.append(
            f"KV budget estimated as (memory x {util:.0%} - weights) x {1 - ACTIVATION_RESERVE_FRACTION:.0%} "
            "activation reserve; the server's own profiler may reserve more."
        )
    else:
        capacity = None
        source = "unavailable"
        if mem is None:
            notes.append("GPU memory unknown (no NVML/nvidia-smi on this host); KV budget not estimated.")
        if weight_total is None:
            notes.append("Model parameter count unknown; weights footprint not estimated.")

    max_len = getattr(server_info, "max_model_len", None)
    if capacity and max_len and capacity < max_len:
        notes.append(
            f"KV cache holds {capacity:,} tokens, less than --max-model-len {max_len:,}: "
            "a single request at full context cannot be admitted."
        )

    return Footprint(
        quantization=quant,
        bytes_per_param=bpp,
        weight_bytes_total=weight_total,
        weight_bytes_per_gpu=weight_per_gpu,
        kv_bytes_per_token_total=kv_total,
        kv_bytes_per_token_per_gpu=kv_per_gpu,
        kv_cache_dtype=kv_dtype,
        gpu_memory_bytes=mem,
        gpu_memory_utilization=util,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        kv_budget_bytes_per_gpu=budget,
        kv_capacity_tokens=capacity,
        kv_capacity_source=source,
        fits_on_one_gpu=fits,
        notes=notes,
    )


def max_concurrency_at(capacity_tokens: Optional[int], context_length: int, output_tokens: int) -> Optional[int]:
    """How many sequences of ``context + output`` tokens fit in the KV cache."""
    if not capacity_tokens:
        return None
    per_seq = max(1, int(context_length) + int(output_tokens))
    return capacity_tokens // per_seq


def matrix_fit(
    capacity_tokens: Optional[int],
    context_lengths: list[int],
    concurrency_levels: list[int],
    output_tokens: int,
) -> dict[str, Any]:
    """Which matrix cells fit in the KV cache without queueing or preemption."""
    if not capacity_tokens:
        return {"known": False, "cells_over_capacity": [], "max_users_by_context": {}}
    max_users = {int(c): max_concurrency_at(capacity_tokens, c, output_tokens) for c in context_lengths}
    over = [
        {"context_length": int(c), "concurrent_users": int(u), "max_users": max_users[int(c)]}
        for c in context_lengths for u in concurrency_levels
        if max_users[int(c)] is not None and u > max_users[int(c)]
    ]
    return {"known": True, "cells_over_capacity": over, "max_users_by_context": max_users}


# ---------------------------------------------------------------------------
# Roofline ceilings
# ---------------------------------------------------------------------------

def decode_ceiling_tps(
    bandwidth_gbps: Optional[float],
    weight_bytes: Optional[int],
    kv_bytes_per_token: Optional[int],
    context_length: int,
    concurrency: int,
    compute_ceiling_tps: Optional[float] = None,
    *,
    gpus: int = 1,
) -> tuple[Optional[float], Optional[str]]:
    """Aggregate decode tokens/s ceiling at ``concurrency`` and its binding limit.

    ``N * BW_total / (W + N * KV * L)`` (bandwidth-bound) capped by the
    compute ceiling.  With TP the weights and KV are sharded, so the
    aggregate bandwidth is ``gpus * BW`` against the *total* bytes; the
    interconnect cost of TP is not modelled here (it shows up as the
    ``interconnect`` bottleneck class when measured).
    """
    if not bandwidth_gbps or not weight_bytes or concurrency <= 0:
        return None, None
    kv = kv_bytes_per_token or 0
    bytes_per_step = weight_bytes + concurrency * kv * max(context_length, 1)
    bw_bound = concurrency * bandwidth_gbps * GB * max(1, gpus) / bytes_per_step
    if compute_ceiling_tps and compute_ceiling_tps < bw_bound:
        return float(compute_ceiling_tps), "compute"
    return float(bw_bound), "bandwidth"


def prefill_ceiling_tps(peak_tflops: Optional[float], active_params: Optional[int], *, gpus: int = 1) -> Optional[float]:
    """Prefill tokens/s ceiling: ``peak_FLOPs / (2 * P_active)``."""
    if not peak_tflops or not active_params:
        return None
    return peak_tflops * 1e12 * max(1, gpus) / (2.0 * active_params)


def _measured_decode(cell: dict) -> tuple[Optional[float], Optional[str]]:
    users = int(cell.get("concurrent_users") or 1)
    per_stream = _num(cell.get("decode_tps_mean")) or _num(cell.get("decode_tps"))
    if per_stream:
        return per_stream * users, "decode_tps_mean x users"
    tps = _num(cell.get("tokens_per_second"))
    if tps:
        return tps, "tokens_per_second"
    return None, None


def cell_ceilings(
    results: list[dict],
    profile: Optional[ModelProfile],
    gpu_spec: Optional[dict],
    footprint: Footprint,
    *,
    output_tokens: int = 0,
) -> list[CellCeiling]:
    """Roofline ceilings and attainment for every measured generative cell."""
    if not results or profile is None or not gpu_spec:
        return []
    bw = _num((gpu_spec or {}).get("hbm_bandwidth_gbps"))
    peak = _peak_for(gpu_spec, footprint.quantization)
    gpus = footprint.tensor_parallel * footprint.pipeline_parallel
    compute_ceiling = prefill_ceiling_tps(peak, profile.active_params, gpus=gpus)
    out: list[CellCeiling] = []
    for cell in results:
        ctx = int(cell.get("context_length") or 0)
        users = int(cell.get("concurrent_users") or 1)
        # Decode reads the KV of the whole sequence; use context + half the
        # generated tokens as the mean sequence length during decode.
        eff_ctx = ctx + int(output_tokens / 2)
        ceiling, bound = decode_ceiling_tps(
            bw, footprint.weight_bytes_total, footprint.kv_bytes_per_token_total, eff_ctx, users,
            compute_ceiling, gpus=gpus,
        )
        measured, source = _measured_decode(cell)
        attain = (measured / ceiling) if (measured and ceiling) else None
        pf_measured = _num(cell.get("prefill_tps_mean")) or _num(cell.get("prefill_tps"))
        pf_attain = (pf_measured / compute_ceiling) if (pf_measured and compute_ceiling) else None
        out.append(CellCeiling(
            context_length=ctx, concurrent_users=users, prompt_type=cell.get("prompt_type"),
            decode_ceiling_tps=ceiling, decode_measured_tps=measured, decode_measured_source=source,
            decode_attainment=attain, bound_by=bound,
            prefill_ceiling_tps=compute_ceiling, prefill_measured_tps=pf_measured, prefill_attainment=pf_attain,
        ))
    return out


def context_slope(
    ceilings: list[CellCeiling],
    footprint: Footprint,
) -> list[dict]:
    """Single-stream decode ceiling vs measured across context lengths.

    Also reports the KV share of bytes per step, which is what makes decode
    slow down with context: at ``KV * L / (W + KV * L)`` of the traffic the
    slowdown is physics; a measured slope steeper than the ceiling's points
    at the server (attention kernel, paging) rather than the memory system.
    """
    rows = []
    w = footprint.weight_bytes_total or 0
    kv = footprint.kv_bytes_per_token_total or 0
    for c in sorted((c for c in ceilings if c.concurrent_users == 1), key=lambda c: c.context_length):
        kv_share = (kv * c.context_length) / (w + kv * c.context_length) if (w + kv * c.context_length) else None
        rows.append({
            "context_length": c.context_length,
            "ceiling_tps": c.decode_ceiling_tps,
            "measured_tps": c.decode_measured_tps,
            "attainment": c.decode_attainment,
            "kv_share_of_bytes": kv_share,
        })
    return rows


# ---------------------------------------------------------------------------
# What-ifs: quantization and KV dtype
# ---------------------------------------------------------------------------

_FORMATS: list[tuple[str, float, str, str]] = [
    # label, bytes/param, peak key for compute ceiling, note
    ("BF16", 2.0, "bf16", "16-bit weights"),
    ("FP8", 1.0, "fp8", "needs Ada or newer"),
    ("INT4 (AWQ/GPTQ)", 0.5, "bf16", "weight-only; dequantizes to 16-bit for the matmul"),
    ("NVFP4", 0.5, "fp4", "needs Blackwell; FP4 tensor cores"),
]


def quant_what_if(
    profile: Optional[ModelProfile],
    gpu_spec: Optional[dict],
    footprint: Footprint,
    *,
    context_length: int,
    output_tokens: int,
) -> list[dict]:
    """Project weights, KV headroom and the single-stream ceiling per weight format.

    Uses the same roofline as the measured cells so the rows are comparable
    with what was observed for the current format.  Formats the GPU cannot
    run (no FP8 peak on Ampere, no FP4 before Blackwell) are listed as
    unsupported rather than silently dropped.
    """
    if profile is None or not profile.total_params or not gpu_spec:
        return []
    bw = _num(gpu_spec.get("hbm_bandwidth_gbps"))
    peaks = gpu_spec.get("peak_flops_tflops") or {}
    gpus = footprint.tensor_parallel * footprint.pipeline_parallel
    mem = footprint.gpu_memory_bytes
    util = footprint.gpu_memory_utilization
    current_key = peak_dtype_for(footprint.quantization)
    current_bpp = footprint.bytes_per_param
    rows = []
    for label, bpp, peak_key, note in _FORMATS:
        peak = _num(peaks.get(peak_key))
        supported = peak is not None
        weights = int(profile.total_params * bpp)
        kv_tokens = None
        if mem and supported:
            usable = int(mem * util) * gpus
            kv_tokens = int(max(0, usable - weights) * (1 - ACTIVATION_RESERVE_FRACTION) / footprint.kv_bytes_per_token_total) \
                if footprint.kv_bytes_per_token_total else None
        compute = prefill_ceiling_tps(peak, profile.active_params, gpus=gpus) if supported else None
        ceiling, _ = decode_ceiling_tps(bw, weights, footprint.kv_bytes_per_token_total,
                                        context_length + output_tokens // 2, 1, compute, gpus=gpus) if supported else (None, None)
        rows.append({
            "format": label,
            "bytes_per_param": bpp,
            "supported": supported,
            "weight_bytes": weights,
            "kv_capacity_tokens_est": kv_tokens,
            "max_users_at_context": max_concurrency_at(kv_tokens, context_length, output_tokens) if kv_tokens else None,
            "single_stream_ceiling_tps": ceiling,
            "prefill_ceiling_tps": compute,
            "current": (bpp == current_bpp and peak_key == current_key),
            "note": note,
        })
    # KV-cache dtype what-if: FP8 KV halves KV bytes -> ~2x the token budget.
    if footprint.kv_bytes_per_token_total and _kv_dtype_bytes_label(footprint.kv_cache_dtype) == 2.0:
        rows.append({
            "format": "FP8 KV cache (same weights)",
            "bytes_per_param": current_bpp,
            "supported": _num(peaks.get("fp8")) is not None,
            "weight_bytes": footprint.weight_bytes_total,
            "kv_capacity_tokens_est": (footprint.kv_capacity_tokens * 2) if footprint.kv_capacity_tokens else None,
            "max_users_at_context": max_concurrency_at(
                footprint.kv_capacity_tokens * 2 if footprint.kv_capacity_tokens else None, context_length, output_tokens),
            "single_stream_ceiling_tps": decode_ceiling_tps(
                bw, footprint.weight_bytes_total, (footprint.kv_bytes_per_token_total or 0) // 2,
                context_length + output_tokens // 2, 1, None, gpus=gpus)[0],
            "prefill_ceiling_tps": None,
            "current": False,
            "note": "--kv-cache-dtype fp8: halves KV bytes per token; check quality with --quality kl",
        })
    return rows


def _kv_dtype_bytes_label(kv_dtype: Optional[str]) -> float:
    return bytes_per_param(kv_dtype) if kv_dtype and kv_dtype != "auto" else 2.0


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------

def sweet_spot(results: list[dict], bottlenecks: Optional[list[dict]], ceilings: list[CellCeiling]) -> dict[str, Any]:
    """Where adding users stops paying, from the measured curve and B*.

    For each context, walk the concurrency levels in order and report the
    last level at which the marginal throughput per added user is still at
    least :data:`SWEET_SPOT_MARGINAL_GAIN` of linear.  Compare with the
    roofline's critical batch ``B*`` (from the bottleneck verdicts) so the
    report can say whether the knee is the hardware's (compute-bound past
    B*) or the server's (knee well below B*: scheduling, queueing).
    """
    by_ctx: dict[int, list[tuple[int, float]]] = {}
    for r in results or []:
        tps = _num(r.get("tokens_per_second"))
        if tps is None:
            continue
        by_ctx.setdefault(int(r.get("context_length") or 0), []).append((int(r.get("concurrent_users") or 1), tps))
    bstar = None
    for v in bottlenecks or []:
        if v.get("critical_batch"):
            bstar = v["critical_batch"]
            break
    per_context = {}
    for ctx, pts in by_ctx.items():
        pts = sorted(set(pts))
        if len(pts) < 2:
            continue
        best_users = pts[0][0]
        for (u0, t0), (u1, t1) in zip(pts, pts[1:]):
            linear_gain = t0 / u0 * (u1 - u0) if u0 else 0
            gain = t1 - t0
            if linear_gain > 0 and gain / linear_gain >= SWEET_SPOT_MARGINAL_GAIN:
                best_users = u1
            else:
                break
        last_users, last_tps = pts[-1]
        per_context[ctx] = {
            "sweet_spot_users": best_users,
            "tested_max_users": last_users,
            "throughput_at_sweet_spot": next(t for u, t in pts if u == best_users),
            "throughput_at_max": last_tps,
            "still_scaling": best_users == last_users,
        }
    verdict = None
    if per_context:
        ctx0 = min(per_context)
        s = per_context[ctx0]
        if s["still_scaling"]:
            verdict = (f"Throughput was still scaling at {s['tested_max_users']} users ({ctx0 // 1000}K): "
                       "push concurrency higher"
                       + (f"; the roofline critical batch is about {bstar:.0f}." if bstar else "."))
        elif bstar and s["sweet_spot_users"] < bstar / 2:
            verdict = (f"Gains flatten at {s['sweet_spot_users']} users ({ctx0 // 1000}K), well below the hardware "
                       f"critical batch (~{bstar:.0f}): the limit is the server (scheduling, KV capacity, queueing), "
                       "not the GPU.")
        else:
            verdict = (f"Gains flatten at {s['sweet_spot_users']} users ({ctx0 // 1000}K)"
                       + (f", consistent with the critical batch (~{bstar:.0f}): the GPU is compute-bound past that."
                          if bstar else "."))
    return {"critical_batch": bstar, "per_context": per_context, "verdict": verdict}


# ---------------------------------------------------------------------------
# Multi-GPU and power
# ---------------------------------------------------------------------------

def multi_gpu_advice(footprint: Footprint, system_info: Optional[dict], server_info: Any, environment: Optional[dict]) -> dict:
    """Replicas vs tensor parallel for the installed GPUs.

    Without NVLink (every RTX / RTX PRO card, most PCIe datacenter cards)
    tensor parallelism pays an all-reduce over PCIe on every layer.  When
    the model fits on one GPU with usable KV room, independent replicas
    scale throughput linearly with no interconnect cost; TP is only worth it
    when the model does not fit or single-stream latency must be minimised.
    """
    count = int((system_info or {}).get("gpu_count") or 1)
    tp = footprint.tensor_parallel
    pp = footprint.pipeline_parallel
    env_gpu = ((environment or {}).get("gpu") or {})
    per_gpu = env_gpu.get("per_gpu") or []
    pcie = None
    if per_gpu:
        g = per_gpu[0]
        if g.get("pcie_gen") and g.get("pcie_width"):
            pcie = f"PCIe Gen{int(g['pcie_gen'])} x{int(g['pcie_width'])}"
    nvlink = bool(env_gpu.get("nvlink"))
    out: dict[str, Any] = {"gpu_count": count, "tensor_parallel": tp, "pipeline_parallel": pp,
                           "pcie": pcie, "nvlink": nvlink, "fits_on_one_gpu": footprint.fits_on_one_gpu,
                           "recommendation": None}
    if count <= 1:
        return out
    if footprint.fits_on_one_gpu and tp * pp > 1:
        out["recommendation"] = (
            f"The model fits on one GPU; {count} GPUs as {count} independent replicas (data parallel) "
            f"scale throughput ~{count}x with no {'NVLink' if nvlink else 'PCIe'} all-reduce cost, "
            f"versus TP={tp} which pays an interconnect round-trip per layer. Use TP only for lower "
            "single-stream latency or when KV room on one GPU is too small for your context."
        )
    elif footprint.fits_on_one_gpu and tp * pp == 1:
        out["recommendation"] = (
            f"Single-GPU serving on a {count}-GPU host: run {count} replicas behind a load balancer "
            "for ~linear aggregate throughput."
        )
    elif footprint.fits_on_one_gpu is False and footprint.weight_bytes_total and footprint.gpu_memory_bytes:
        need = math.ceil(footprint.weight_bytes_total / (footprint.gpu_memory_bytes * footprint.gpu_memory_utilization * 0.8))
        out["recommendation"] = (
            f"Weights do not fit on one GPU; at least TP={need} is required"
            + (" (PCIe only, expect sublinear scaling: check the interconnect bottleneck class)." if not nvlink else ".")
        )
    return out


def power_summary(results: list[dict], gpu_spec: Optional[dict]) -> dict:
    """Sustained clock and throttling truth from the per-cell telemetry.

    Uses ``avg_gpu_clock`` vs the device's max SM clock, the enforced power
    limit and the throttle-reason fractions the poller records.  A card that
    runs at 75% of its max clock while power-capped for most of a cell is
    power-limited: a 300 W Max-Q part looks exactly like this against a
    600 W sibling, and no amount of software tuning changes it.
    """
    dev = (gpu_spec or {}).get("_device") or {}
    max_clock = _num(dev.get("max_sm_clock_mhz"))
    clocks = [_num(r.get("avg_gpu_clock")) for r in results or []]
    clocks = [c for c in clocks if c]
    cap = [_num(r.get("power_cap_frac")) for r in results or []]
    cap = [c for c in cap if c is not None]
    therm = [_num(r.get("thermal_frac")) for r in results or []]
    therm = [t for t in therm if t is not None]
    limits = [_num(r.get("power_limit_w")) for r in results or []]
    limits = [v for v in limits if v]
    power = [_num(r.get("avg_power")) for r in results or []]
    power = [p for p in power if p]
    out: dict[str, Any] = {
        "max_sm_clock_mhz": max_clock,
        "avg_sm_clock_mhz": (sum(clocks) / len(clocks)) if clocks else None,
        "sustained_clock_pct": None,
        "power_limit_w": (max(limits) if limits else _num(dev.get("enforced_power_limit_w"))),
        "avg_power_w": (sum(power) / len(power)) if power else None,
        "power_capped_frac": (sum(cap) / len(cap)) if cap else None,
        "thermal_frac": (sum(therm) / len(therm)) if therm else None,
        "verdict": None,
    }
    if clocks and max_clock:
        out["sustained_clock_pct"] = out["avg_sm_clock_mhz"] / max_clock
    pct = out["sustained_clock_pct"]
    if pct is not None:
        if out["power_capped_frac"] is not None and out["power_capped_frac"] > 0.5:
            out["verdict"] = (f"Power-limited: ran at {pct:.0%} of max SM clock, power-capped {out['power_capped_frac']:.0%} "
                              f"of the time at {out['power_limit_w']:.0f} W. A higher power limit (or the 600 W variant) "
                              "is the lever; software tuning will not recover this.")
        elif out["thermal_frac"] is not None and out["thermal_frac"] > 0.2:
            out["verdict"] = (f"Thermally throttled {out['thermal_frac']:.0%} of the time ({pct:.0%} of max clock): "
                              "cooling / airflow is the lever.")
        elif pct < 0.85:
            out["verdict"] = (f"Sustained {pct:.0%} of max SM clock without a reported power or thermal cap: "
                              "check application clocks and persistence mode.")
        else:
            out["verdict"] = f"Clocks held at {pct:.0%} of max: not power- or thermal-limited."
    return out


# ---------------------------------------------------------------------------
# Server command
# ---------------------------------------------------------------------------

def server_command(
    footprint: Footprint,
    server_info: Any,
    *,
    context_lengths: list[int],
    output_tokens: int,
    throughput_optimal_users: Optional[int] = None,
) -> dict[str, Any]:
    """A ready-to-paste serve command sized from the capacity numbers.

    ``--max-model-len`` is the largest context in the matrix plus the
    output budget, capped by what the KV cache can hold for one sequence;
    ``--max-num-seqs`` is the number of such sequences that fit at the
    throughput-optimal context (or the largest), never below the measured
    sweet spot.  Everything is a starting point, labelled as such.
    """
    model = getattr(server_info, "model_name", None) or getattr(server_info, "served_model_path", None) or "<model>"
    backend = getattr(server_info, "backend", None) or "vllm"
    cap = footprint.kv_capacity_tokens
    ctx_max = max(context_lengths) if context_lengths else 32_000
    max_len = ctx_max + output_tokens
    if cap:
        max_len = min(max_len, cap)
    seqs = max_concurrency_at(cap, ctx_max, output_tokens) if cap else None
    if throughput_optimal_users and seqs:
        seqs = max(seqs, throughput_optimal_users)
    seqs = max(1, seqs) if seqs else None
    util = footprint.gpu_memory_utilization
    tp = footprint.tensor_parallel
    kv_fp8 = (footprint.kv_cache_dtype or "").lower().startswith("fp8")
    parts: list[str]
    if backend == "sglang":
        parts = [f"python -m sglang.launch_server --model-path {model}", f"--context-length {max_len}",
                 f"--mem-fraction-static {util:.2f}", "--enable-metrics"]
        if seqs:
            parts.append(f"--max-running-requests {seqs}")
        if tp > 1:
            parts.append(f"--tp {tp}")
        if kv_fp8:
            parts.append("--kv-cache-dtype fp8_e5m2")
    else:
        parts = [f"vllm serve {model}", f"--max-model-len {max_len}", f"--gpu-memory-utilization {util:.2f}"]
        if seqs:
            parts.append(f"--max-num-seqs {seqs}")
        if tp > 1:
            parts.append(f"--tensor-parallel-size {tp}")
        if kv_fp8:
            parts.append("--kv-cache-dtype fp8")
        if getattr(server_info, "prefix_caching", None) is False:
            parts.append("--enable-prefix-caching")
    return {
        "backend": backend,
        "command": " \\\n  ".join(parts),
        "max_model_len": max_len,
        "max_num_seqs": seqs,
        "basis": ("KV capacity " + (f"{cap:,} tokens ({footprint.kv_capacity_source})" if cap else "unknown")
                  + f"; context {ctx_max:,} + {output_tokens} output tokens"),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def analyze_capacity(
    results: Optional[list[dict]],
    profile: Optional[ModelProfile],
    gpu_spec: Optional[dict],
    server_info: Any,
    system_info: Optional[dict],
    *,
    context_lengths: list[int],
    concurrency_levels: list[int],
    output_tokens: int,
    bottlenecks: Optional[list[dict]] = None,
    environment: Optional[dict] = None,
    throughput_optimal_users: Optional[int] = None,
) -> dict[str, Any]:
    """Assemble the full capacity / ceilings / what-if analysis as a dict.

    Works before a run (``results=None``: footprint, matrix fit, what-ifs,
    server command) and after it (adds per-cell ceilings, context slope,
    sweet spot and power truth).  Never raises on missing inputs; absent
    sections are ``None`` or empty and ``notes`` says why.
    """
    results = results or []
    measured_kv = None
    for r in results:
        if _num(r.get("kv_capacity_tokens")):
            measured_kv = int(r["kv_capacity_tokens"])
            break
    raw = getattr(server_info, "raw", None) or {}
    if measured_kv is None:
        for key in ("kv_capacity_tokens", "max_total_num_tokens"):
            v = _num(raw.get(key)) or _num((raw.get("server_info") or {}).get(key)) if isinstance(raw, dict) else None
            if v:
                measured_kv = int(v)
                break
    util = None
    if isinstance(raw, dict):
        util = _num(raw.get("gpu_memory_utilization")) or _num((raw.get("server_info") or {}).get("mem_fraction_static"))

    fp = compute_footprint(profile, gpu_spec, server_info, system_info,
                           measured_kv_capacity_tokens=measured_kv, gpu_memory_utilization=util)
    fit = matrix_fit(fp.kv_capacity_tokens, context_lengths, concurrency_levels, output_tokens)
    ceilings = cell_ceilings(results, profile, gpu_spec, fp, output_tokens=output_tokens)
    ctx0 = min(context_lengths) if context_lengths else 32_000
    what_if = quant_what_if(profile, gpu_spec, fp, context_length=ctx0, output_tokens=output_tokens)
    spot = sweet_spot(results, bottlenecks, ceilings) if results else {"critical_batch": None, "per_context": {}, "verdict": None}
    bstar = spot.get("critical_batch")
    if bstar is None and gpu_spec:
        bstar = critical_batch(_peak_for(gpu_spec, fp.quantization), _num(gpu_spec.get("hbm_bandwidth_gbps")), fp.bytes_per_param)
        spot["critical_batch"] = bstar
    cmd = server_command(fp, server_info, context_lengths=context_lengths, output_tokens=output_tokens,
                         throughput_optimal_users=throughput_optimal_users)
    best_cell = None
    if ceilings:
        graded = [c for c in ceilings if c.decode_attainment is not None]
        if graded:
            b = max(graded, key=lambda c: c.decode_attainment)
            best_cell = {"context_length": b.context_length, "concurrent_users": b.concurrent_users,
                         "decode_attainment": b.decode_attainment, "bound_by": b.bound_by}
    return {
        "footprint": {
            "quantization": fp.quantization,
            "bytes_per_param": fp.bytes_per_param,
            "weight_bytes_total": fp.weight_bytes_total,
            "weight_bytes_per_gpu": fp.weight_bytes_per_gpu,
            "kv_bytes_per_token": fp.kv_bytes_per_token_total,
            "kv_cache_dtype": fp.kv_cache_dtype,
            "gpu_memory_bytes": fp.gpu_memory_bytes,
            "gpu_memory_utilization": fp.gpu_memory_utilization,
            "tensor_parallel": fp.tensor_parallel,
            "pipeline_parallel": fp.pipeline_parallel,
            "kv_budget_bytes_per_gpu": fp.kv_budget_bytes_per_gpu,
            "kv_capacity_tokens": fp.kv_capacity_tokens,
            "kv_capacity_source": fp.kv_capacity_source,
            "fits_on_one_gpu": fp.fits_on_one_gpu,
            "max_context_single_stream": fp.kv_capacity_tokens,
        },
        "matrix_fit": fit,
        "ceilings": [c.__dict__ for c in ceilings],
        "best_attainment": best_cell,
        "context_slope": context_slope(ceilings, fp),
        "what_if": what_if,
        "sweet_spot": spot,
        "multi_gpu": multi_gpu_advice(fp, system_info, server_info, environment),
        "power": power_summary(results, gpu_spec) if results else None,
        "server_command": cmd,
        "notes": list(fp.notes),
        "method": {
            "decode_ceiling": "N * BW / (W + N * KV * L), capped by peak_FLOPs / (2 * P_active)",
            "prefill_ceiling": "peak_FLOPs(format) / (2 * P_active)",
            "kv_capacity": fp.kv_capacity_source,
            "references": [
                "Williams, Waterman, Patterson. Roofline: an insightful visual performance model. CACM 2009.",
                "Pope et al. Efficiently Scaling Transformer Inference. 2022.",
                "Kaplan et al. Scaling Laws for Neural Language Models. 2020.",
                "Databricks. LLM Inference Performance Engineering: Best Practices. 2023.",
            ],
        },
    }
