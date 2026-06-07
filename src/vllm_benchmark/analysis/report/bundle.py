"""Facts-only JSON bundle for the AI analyst report.

The bundle is the *single source of truth* the language model is allowed to
draw from.  Every number, classification, bottleneck verdict, MBU/MFU,
fitness grade and recommendation it contains was already computed by an
earlier PR — this module only *assembles* and *sanitizes* those facts into
a compact dict.  Nothing here computes a new performance metric.

Three public helpers:

* :func:`build_bundle` — assemble the compact facts dict from a run.
* :func:`allowed_numbers` — the canonical set of every numeric value in a
  bundle (plus faithful unit transforms), used by the numeric verifier.
* :func:`bundle_sha256` — a stable content hash of a bundle.

Untrusted server-derived strings (model name / path) are sanitized:
control characters are stripped and the length is capped.  They are carried
as DATA and never interpreted as instructions.

Author: amit
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

#: Maximum length for any sanitized server-derived string.
_MAX_STR_LEN = 200

#: Control characters (C0/C1 except common whitespace) to strip from
#: untrusted strings before they enter the bundle.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize_text(value: Any, *, max_len: int = _MAX_STR_LEN) -> Optional[str]:
    """Sanitize an untrusted, server-derived string for safe carriage as data.

    Strips control characters, collapses interior runs of whitespace, trims
    surrounding whitespace and caps the length.  The result is meant to be
    embedded in the bundle as an opaque label — never interpreted as an
    instruction by a downstream consumer.

    Args:
        value: The raw value (typically a model name or path).  Non-string
            values other than ``None`` are coerced via :func:`str`.
        max_len: Maximum retained length.

    Returns:
        The sanitized string, or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


# ---------------------------------------------------------------------------
# Numeric collection helpers
# ---------------------------------------------------------------------------

def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    """Round a float for stable hashing / comparison (``None`` passthrough)."""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _ci_from_cell(cell: dict, key: str) -> Optional[dict]:
    """Extract a confidence interval for ``key`` from a result cell.

    Earlier PRs attach per-metric CIs either as flat ``<key>_ci_lower`` /
    ``<key>_ci_upper`` fields or inside a ``bootstrap_ci`` mapping
    (``{key: [low, point, high]}``).  Returns ``{"lower", "upper"}`` when a
    CI is derivable, else ``None``.  No CI is *computed* here — only read.
    """
    lower = cell.get(f"{key}_ci_lower")
    upper = cell.get(f"{key}_ci_upper")
    if lower is not None and upper is not None:
        return {"lower": _round(lower), "upper": _round(upper)}

    boot = cell.get("bootstrap_ci")
    if isinstance(boot, dict) and key in boot:
        triple = boot[key]
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            return {"lower": _round(triple[0]), "upper": _round(triple[2])}
    return None


def _cell_summary(cell: dict) -> dict:
    """Summarize one result cell into a facts-only aggregate.

    Carries the already-measured throughput / latency figures (with CIs
    where derivable) plus the cell coordinates.  Raw per-request data is
    never dumped.
    """
    keys = (
        "tokens_per_second",
        "prefill_tps",
        "decode_tps",
        "ttft_estimate",
        "avg_latency",
        "latency_p99",
        "cost_per_1m_tokens",
    )
    summary: dict[str, Any] = {
        "context_length": cell.get("context_length"),
        "concurrent_users": cell.get("concurrent_users"),
        "prompt_type": sanitize_text(cell.get("prompt_type")),
    }
    for key in keys:
        # Prefer a mean variant when present, else the plain key.
        val = cell.get(key)
        if val is None:
            val = cell.get(f"{key}_mean")
        if val is not None:
            summary[key] = _round(val)
            ci = _ci_from_cell(cell, key)
            if ci is not None:
                summary[f"{key}_ci"] = ci
    return summary


def _summarize_matrix(results: list[dict]) -> dict:
    """Summarize the result matrix: per-cell aggregates plus run-level peaks.

    All values are read straight from the result cells; nothing is averaged
    or recomputed.  The run-level peaks point at the already-present extreme
    cells so the prose can cite a headline figure without inventing one.
    """
    cells = [_cell_summary(r) for r in results]

    def _peak(key: str, *, want_max: bool) -> Optional[dict]:
        scored = [(r.get(key), _cell_summary(r)) for r in results if r.get(key) is not None]
        if not scored:
            return None
        chooser = max if want_max else min
        _, summary = chooser(scored, key=lambda p: p[0])
        return summary

    return {
        "cells": cells,
        "peak_throughput_cell": _peak("tokens_per_second", want_max=True),
        "lowest_latency_cell": _peak("avg_latency", want_max=False),
        "lowest_ttft_cell": _peak("ttft_estimate", want_max=False),
    }


def _summarize_model_profile(profile: Optional[dict]) -> Optional[dict]:
    """Carry the model-profile facts (architecture + provenance)."""
    if not profile:
        return None
    return {
        "name": sanitize_text(profile.get("name")),
        "family": sanitize_text(profile.get("family")),
        "is_moe": profile.get("is_moe"),
        "active_params": profile.get("active_params"),
        "total_params": profile.get("total_params"),
        "attention_type": profile.get("attention_type"),
        "num_layers": profile.get("num_layers"),
        "kv_bytes_per_token": profile.get("kv_bytes_per_token"),
        "confidence": profile.get("confidence"),
        "source": profile.get("source"),
    }


def _summarize_bottlenecks(bottlenecks: Optional[list[dict]]) -> dict:
    """Carry the bottleneck verdicts (governing + per-cell facts)."""
    verdicts = bottlenecks or []
    order = {"high": 0, "medium": 1, "low": 2}
    governing: Optional[dict] = None
    if verdicts:
        top = min(verdicts, key=lambda v: order.get(v.get("confidence"), 3))
        governing = {
            "primary": top.get("primary"),
            "lever": sanitize_text(top.get("lever"), max_len=300),
            "confidence": top.get("confidence"),
            "mbu": _round(top.get("mbu")),
            "mfu": _round(top.get("mfu")),
            "critical_batch": top.get("critical_batch"),
            "cell": list(top.get("cell")) if top.get("cell") is not None else None,
        }

    per_cell = []
    for v in verdicts:
        per_cell.append({
            "cell": list(v.get("cell")) if v.get("cell") is not None else None,
            "primary": v.get("primary"),
            "mbu": _round(v.get("mbu")),
            "mfu": _round(v.get("mfu")),
            "critical_batch": v.get("critical_batch"),
            "confidence": v.get("confidence"),
            "lever": sanitize_text(v.get("lever"), max_len=300),
        })
    return {"governing": governing, "per_cell": per_cell}


def _summarize_advisory(advisory: Optional[dict]) -> dict:
    """Carry operating points, tips, explanation and fitness from advisory."""
    adv = advisory or {}
    fitness = adv.get("fitness") or {}
    profiles = fitness.get("profiles") or {}

    fitness_out = {
        "verdict": sanitize_text(fitness.get("verdict"), max_len=400),
        "profiles": {
            name: {
                "grade": g.get("grade"),
                "limiting_factor": sanitize_text(g.get("limiting_factor"), max_len=200),
                "detail": sanitize_text(g.get("detail"), max_len=300),
                "confidence": g.get("confidence"),
            }
            for name, g in profiles.items()
        },
    } if fitness else None

    def _op_point(cell: Optional[dict]) -> Optional[dict]:
        return _cell_summary(cell) if cell else None

    return {
        "explanation": sanitize_text(adv.get("explanation"), max_len=600),
        "tips": [sanitize_text(t, max_len=400) for t in (adv.get("tips") or [])],
        "confidence": adv.get("confidence"),
        "latency_optimal": _op_point(adv.get("latency_optimal")),
        "throughput_optimal": _op_point(adv.get("throughput_optimal")),
        "fitness": fitness_out,
    }


def _summarize_hardware(metadata: dict) -> dict:
    """Carry the GPU / VRAM facts from system info."""
    system_info = metadata.get("system_info") or {}
    return {
        "gpu": sanitize_text(system_info.get("gpu_name")),
        "vram_gb": _round(system_info.get("total_vram_gb"), 2),
    }


def _summarize_quality(quality: Optional[dict]) -> Optional[dict]:
    """Carry the quality section (already-computed score / mode), if present."""
    if not quality:
        return None
    out: dict[str, Any] = {
        "mode": sanitize_text(quality.get("mode")),
        "status": sanitize_text(quality.get("status")),
    }
    for key in ("score", "perplexity", "kl_divergence"):
        if quality.get(key) is not None:
            out[key] = _round(quality.get(key))
    if quality.get("reason"):
        out["reason"] = sanitize_text(quality.get("reason"), max_len=300)
    return out


def _summarize_score(score: Any) -> Optional[dict]:
    """Carry the overall score / grade, accepting a dict or a ScoreBreakdown."""
    if score is None:
        return None
    if hasattr(score, "to_dict"):
        score = score.to_dict()
    if isinstance(score, dict):
        return {
            "overall": _round(score.get("overall"), 2),
            "grade": sanitize_text(score.get("grade")),
            "throughput": _round(score.get("throughput"), 2),
            "latency": _round(score.get("latency"), 2),
            "efficiency": _round(score.get("efficiency"), 2),
        }
    # Fall back to attribute access for a dataclass without to_dict.
    return {
        "overall": _round(getattr(score, "overall", None), 2),
        "grade": sanitize_text(getattr(score, "grade", None)),
    }


def _lowest_cost(results: list[dict]) -> Optional[float]:
    """Return the best (lowest) already-measured cost-per-1M-tokens, if any."""
    costs = [r["cost_per_1m_tokens"] for r in results if r.get("cost_per_1m_tokens") is not None]
    return _round(min(costs)) if costs else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_bundle(
    results: list[dict],
    metadata: dict,
    score: Any = None,
) -> dict:
    """Assemble the compact, facts-only bundle for the analyst report.

    Args:
        results: Per-cell generative result dicts from the run.
        metadata: Run metadata carrying ``model_profile``, ``bottlenecks``,
            ``advisory``, ``quality``, ``server_info`` and ``system_info``.
        score: Optional overall score (a ``ScoreBreakdown`` or a dict).

    Returns:
        A JSON-serializable dict of facts.  No performance metric is
        computed here; values are copied (and unit-faithful) from the
        inputs.  Untrusted strings are sanitized and carried as data.
    """
    results = results or []
    server_info = metadata.get("server_info") or {}

    bundle = {
        "hardware": _summarize_hardware(metadata),
        "server": {
            "model_name": sanitize_text(server_info.get("model_name")),
            "backend": sanitize_text(server_info.get("backend")),
            "quantization": sanitize_text(server_info.get("quantization")),
            "max_model_len": server_info.get("max_model_len"),
        },
        "model_profile": _summarize_model_profile(metadata.get("model_profile")),
        "matrix": _summarize_matrix(results),
        "bottlenecks": _summarize_bottlenecks(metadata.get("bottlenecks")),
        "advisory": _summarize_advisory(metadata.get("advisory")),
        "quality": _summarize_quality(metadata.get("quality")),
        "cost": {"best_cost_per_1m_tokens": _lowest_cost(results)},
        "score": _summarize_score(score),
    }
    return bundle


def _collect_numbers(obj: Any, out: set[float]) -> None:
    """Recursively collect every numeric leaf from a bundle into ``out``."""
    if isinstance(obj, bool):
        # bool is a subclass of int — not a metric.
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def allowed_numbers(bundle: dict) -> set[float]:
    """Return the canonical set of numeric values the report may cite.

    Includes every numeric leaf in ``bundle`` plus faithful unit transforms
    so the verifier accepts correctly-rendered units without treating them
    as invented figures:

    * a value ``x`` also as ``x * 1000`` (seconds → milliseconds) and
      ``x / 1000`` (milliseconds → seconds),
    * a value ``x`` also as ``x * 100`` (fraction → percent) and
      ``x / 100`` (percent → fraction),
    * a parameter count also as its billions form (``x / 1e9``).

    The transforms are deliberately generous: the verifier only needs a
    *candidate* match, and over-permitting a faithful unit rendering is far
    safer than rejecting one.

    Args:
        bundle: A bundle produced by :func:`build_bundle`.

    Returns:
        A set of allowed float values, rounded for stable comparison.
    """
    base: set[float] = set()
    _collect_numbers(bundle, base)

    expanded: set[float] = set()
    for x in base:
        expanded.add(round(x, 6))
        expanded.add(round(x * 1000.0, 6))   # s -> ms
        expanded.add(round(x / 1000.0, 6))   # ms -> s
        expanded.add(round(x * 100.0, 6))    # fraction -> percent
        expanded.add(round(x / 100.0, 6))    # percent -> fraction
        expanded.add(round(x / 1e9, 6))      # params -> billions
        expanded.add(round(x / 1e6, 6))      # params -> millions
    return expanded


def bundle_sha256(bundle: dict) -> str:
    """Return a stable SHA-256 hex digest of a bundle.

    The bundle is serialized with sorted keys and compact separators so the
    digest is deterministic for equal content regardless of insertion order.
    """
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
