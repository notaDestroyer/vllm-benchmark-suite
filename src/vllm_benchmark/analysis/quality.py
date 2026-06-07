"""Quality measurement for inference servers (probe / perplexity / KL).

This module measures *output quality* independently of raw performance.  It
is deliberately kept in its own results section: quality numbers are never
folded into the performance score.

Four modes are dispatched by :func:`run_quality`:

* ``off`` — quality measurement disabled (the default); returns a small
  ``{"mode": "off", "status": "disabled"}`` marker.
* ``probe`` — send a bundled deterministic eval set at ``temperature=0`` and
  grade each response by exact/regex match.  Reports an overall score plus
  per-category pass rates each with a Wilson 95% confidence interval.
  Backend-agnostic; needs no logprobs.
* ``perplexity`` — compute corpus perplexity from prompt logprobs.  Gated on
  backend support: if the endpoint cannot return prompt logprobs the result
  is ``status == "skipped"`` with a reason — numbers are never fabricated.
* ``kl`` — token-level KL divergence and top-1 agreement between a quantized
  endpoint and a reference endpoint (``ref_url``) over identical prompts.
  Skipped (with a reason) when ``ref_url`` is missing or logprobs are
  unsupported.  Tokenizer mismatch between the two endpoints is detected and
  reported.

The pure scoring/maths helpers (:func:`grade_probe_response`,
:func:`perplexity_from_logprobs`, :func:`kl_divergence`,
:func:`top1_agreement`) carry no network dependency and are unit-tested
directly.

Author: amit
License: MIT
"""

from __future__ import annotations

import importlib.resources
import json
import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import requests

from vllm_benchmark.analysis.statistics import wilson_interval

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.config import BenchmarkConfig
    from vllm_benchmark.core.backends.base import ServerInfo


# ---------------------------------------------------------------------------
# Bundled data loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_probes() -> list[dict]:
    """Load the bundled deterministic probe set.

    Returns:
        A list of probe dicts, each with ``id``, ``category``, ``prompt``,
        ``expected`` and ``match`` (``"exact"`` or ``"regex"``; regex probes
        also carry a ``pattern``).
    """
    resource = importlib.resources.files("vllm_benchmark.data.quality").joinpath(
        "probes.json"
    )
    with resource.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("probes", []))


@lru_cache(maxsize=1)
def load_perplexity_corpus() -> list[str]:
    """Load the bundled perplexity corpus as a list of non-empty lines."""
    resource = importlib.resources.files("vllm_benchmark.data.quality").joinpath(
        "perplexity_corpus.txt"
    )
    text = resource.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Pure grading / maths helpers
# ---------------------------------------------------------------------------


def grade_probe_response(response: str, expected: str, match: str) -> bool:
    """Grade a single probe response against an expected value.

    Pure and network-free.

    Args:
        response: The model's response text.
        expected: For ``match == "exact"`` the exact (case-sensitive,
            whitespace-stripped) target string; for ``match == "regex"`` a
            regular expression pattern that must be *found* anywhere in the
            response.
        match: ``"exact"`` or ``"regex"``.

    Returns:
        ``True`` if the response satisfies the grading rule.

    Raises:
        ValueError: If ``match`` is not a recognised mode.
    """
    if response is None:
        return False

    if match == "exact":
        return response.strip() == expected.strip()

    if match == "regex":
        try:
            return re.search(expected, response) is not None
        except re.error:
            return False

    raise ValueError(f"Unsupported match mode: {match!r}")


def perplexity_from_logprobs(token_logprobs: list[float]) -> float:
    """Compute perplexity from per-token log-probabilities.

    Method & assumptions: ``PPL = exp(-mean(log p))`` with fixed
    length-normalization over the supplied per-token natural-log
    probabilities.  ``None`` entries (commonly the first token, which has no
    conditioning context) are ignored.

    Args:
        token_logprobs: Natural-log probabilities of the observed tokens.

    Returns:
        The corpus/sequence perplexity, or ``float("inf")`` when there are
        no usable log-probabilities.
    """
    usable = [lp for lp in token_logprobs if lp is not None and math.isfinite(lp)]
    if not usable:
        return float("inf")
    mean_logprob = sum(usable) / len(usable)
    return math.exp(-mean_logprob)


def kl_divergence(p_logprobs: dict[str, float], q_logprobs: dict[str, float]) -> float:
    """Token-level KL divergence ``D_KL(P || Q)`` from log-prob maps.

    Method & assumptions: ``P`` and ``Q`` are discrete distributions over a
    shared support of token strings, supplied as natural-log probabilities.
    ``D_KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))``.  Tokens present in
    ``P`` but absent from ``Q`` are assigned a small floor probability so the
    result stays finite (a one-sided smoothing); ``P`` is renormalized over
    its own support before the sum.

    Args:
        p_logprobs: Reference distribution as ``{token: log_prob}``.
        q_logprobs: Comparison distribution as ``{token: log_prob}``.

    Returns:
        The non-negative KL divergence in nats.  Empty ``P`` -> ``0.0``.
    """
    if not p_logprobs:
        return 0.0

    # Renormalize P over its own support.
    p_probs = {tok: math.exp(lp) for tok, lp in p_logprobs.items()}
    total = sum(p_probs.values())
    if total <= 0:
        return 0.0
    p_probs = {tok: pr / total for tok, pr in p_probs.items()}

    # Floor for tokens missing from Q (one-sided smoothing).
    floor = 1e-12
    kl = 0.0
    for tok, p in p_probs.items():
        if p <= 0:
            continue
        q_lp = q_logprobs.get(tok)
        q = math.exp(q_lp) if q_lp is not None else floor
        if q <= 0:
            q = floor
        kl += p * (math.log(p) - math.log(q))
    return max(0.0, kl)


def top1_agreement(p_top: list[str], q_top: list[str]) -> float:
    """Fraction of positions where the top-1 tokens agree.

    Args:
        p_top: Argmax token at each position for distribution ``P``.
        q_top: Argmax token at each position for distribution ``Q``.

    Returns:
        Agreement rate in ``[0, 1]`` over the overlapping prefix.  Returns
        ``0.0`` when either sequence is empty.
    """
    n = min(len(p_top), len(q_top))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if p_top[i] == q_top[i])
    return matches / n


# ---------------------------------------------------------------------------
# HTTP helpers (monkeypatchable for tests)
# ---------------------------------------------------------------------------


def _post_json(url: str, body: dict, timeout: float) -> Optional[dict]:
    """POST ``body`` to ``url`` and return the parsed JSON, or ``None``.

    The sole network boundary of the quality module; monkeypatched in
    tests.  Never raises — connection/HTTP errors yield ``None``.
    """
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _chat_completion(
    endpoint: str,
    model_name: str,
    prompt: str,
    *,
    max_tokens: int = 64,
    timeout: float = 60.0,
) -> Optional[str]:
    """Send a deterministic (``temperature=0``) chat request, return content."""
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    payload = _post_json(endpoint, body, timeout)
    if not payload:
        return None
    choices = payload.get("choices") or [{}]
    message = choices[0].get("message", {})
    return message.get("content") or ""


# ---------------------------------------------------------------------------
# Mode: probe
# ---------------------------------------------------------------------------


def _chat_endpoint(api_url: str) -> str:
    """Return the chat-completions endpoint for a base URL."""
    return f"{api_url.rstrip('/')}/v1/chat/completions"


def _completions_endpoint(api_url: str) -> str:
    """Return the text-completions endpoint for a base URL."""
    return f"{api_url.rstrip('/')}/v1/completions"


def _run_probe(
    config: "BenchmarkConfig",
    server_info: "ServerInfo",
    model_name: str,
) -> dict:
    """Run the bundled probe set and roll up per-category Wilson CIs."""
    endpoint = _chat_endpoint(config.api_url)
    timeout = float(getattr(config, "request_timeout", 60))
    probes = load_probes()

    per_probe: list[dict] = []
    for probe in probes:
        content = _chat_completion(
            endpoint,
            model_name,
            probe["prompt"],
            max_tokens=64,
            timeout=timeout,
        )
        if content is None:
            # Network failure -> count as a non-pass but record it.
            per_probe.append(
                {
                    "id": probe.get("id"),
                    "category": probe.get("category", "uncategorized"),
                    "passed": False,
                    "error": "request_failed",
                }
            )
            continue
        pattern = probe["pattern"] if probe.get("match") == "regex" else probe.get("expected", "")
        passed = grade_probe_response(content, pattern, probe.get("match", "exact"))
        per_probe.append(
            {
                "id": probe.get("id"),
                "category": probe.get("category", "uncategorized"),
                "passed": bool(passed),
            }
        )

    categories: dict[str, dict] = {}
    for cell in per_probe:
        cat = categories.setdefault(cell["category"], {"successes": 0, "n": 0})
        cat["n"] += 1
        if cell["passed"]:
            cat["successes"] += 1

    category_report: dict[str, dict] = {}
    for cat, agg in categories.items():
        low, point, high = wilson_interval(agg["successes"], agg["n"], confidence=0.95)
        category_report[cat] = {
            "passed": agg["successes"],
            "total": agg["n"],
            "pass_rate": point,
            "ci_low": low,
            "ci_high": high,
        }

    total_n = sum(c["n"] for c in categories.values())
    total_pass = sum(c["successes"] for c in categories.values())
    overall_low, overall_point, overall_high = wilson_interval(
        total_pass, total_n, confidence=0.95
    )

    return {
        "mode": "probe",
        "status": "ok",
        "temperature": 0.0,
        "score": round(overall_point * 100.0, 2),
        "passed": total_pass,
        "total": total_n,
        "overall_ci_low": overall_low,
        "overall_ci_high": overall_high,
        "categories": category_report,
        "probes": per_probe,
    }


# ---------------------------------------------------------------------------
# Mode: perplexity
# ---------------------------------------------------------------------------


def _supports_prompt_logprobs(server_info: "ServerInfo") -> bool:
    """Heuristic: only vLLM exposes ``prompt_logprobs`` for prompt PPL."""
    backend = getattr(server_info, "backend", "unknown")
    return backend == "vllm"


def _request_prompt_logprobs(
    endpoint: str,
    model_name: str,
    text: str,
    *,
    timeout: float,
) -> Optional[list[float]]:
    """Request prompt logprobs for ``text`` via the completions endpoint.

    Uses vLLM's ``prompt_logprobs`` extension with ``max_tokens=0``.  Returns
    the per-token natural-log probabilities of the prompt tokens, or ``None``
    when the response does not carry usable prompt logprobs (never raises).
    """
    body = {
        "model": model_name,
        "prompt": text,
        "max_tokens": 0,
        "temperature": 0.0,
        "echo": True,
        "prompt_logprobs": 0,
        "logprobs": 0,
    }
    payload = _post_json(endpoint, body, timeout)
    if not payload:
        return None
    choices = payload.get("choices") or [{}]
    choice = choices[0]

    # vLLM returns prompt_logprobs as a list of {token: {"logprob": ...}} dicts
    # (the first entry is None as it has no context).
    prompt_lps = choice.get("prompt_logprobs")
    if isinstance(prompt_lps, list):
        out: list[float] = []
        for entry in prompt_lps:
            if not isinstance(entry, dict):
                continue
            # The entry maps token-id/string -> {"logprob": value, ...};
            # the sampled prompt token's logprob is the (single) max-prob one.
            best = None
            for info in entry.values():
                if isinstance(info, dict) and "logprob" in info:
                    lp = info["logprob"]
                    if best is None or lp > best:
                        best = lp
            if best is not None:
                out.append(float(best))
        return out or None

    # Fallback: OpenAI-style echo+logprobs returns token_logprobs directly.
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        token_lps = logprobs.get("token_logprobs")
        if isinstance(token_lps, list):
            return [float(lp) for lp in token_lps if lp is not None]
    return None


def _run_perplexity(
    config: "BenchmarkConfig",
    server_info: "ServerInfo",
    model_name: str,
) -> dict:
    """Compute corpus perplexity from prompt logprobs (backend-gated)."""
    if not _supports_prompt_logprobs(server_info):
        return {
            "mode": "perplexity",
            "status": "skipped",
            "reason": (
                f"backend '{getattr(server_info, 'backend', 'unknown')}' does not "
                "expose prompt logprobs"
            ),
        }

    endpoint = _completions_endpoint(config.api_url)
    timeout = float(getattr(config, "request_timeout", 60))
    corpus = load_perplexity_corpus()

    all_logprobs: list[float] = []
    n_samples = 0
    for line in corpus:
        lps = _request_prompt_logprobs(endpoint, model_name, line, timeout=timeout)
        if lps is None:
            # First failure on a supposedly-supported backend -> skip honestly.
            return {
                "mode": "perplexity",
                "status": "skipped",
                "reason": "endpoint did not return usable prompt logprobs",
            }
        all_logprobs.extend(lps)
        n_samples += 1

    if not all_logprobs:
        return {
            "mode": "perplexity",
            "status": "skipped",
            "reason": "no prompt logprobs collected",
        }

    ppl = perplexity_from_logprobs(all_logprobs)
    return {
        "mode": "perplexity",
        "status": "ok",
        "perplexity": ppl,
        "n_samples": n_samples,
        "n_tokens": len(all_logprobs),
    }


# ---------------------------------------------------------------------------
# Mode: kl
# ---------------------------------------------------------------------------


def _request_token_distributions(
    endpoint: str,
    model_name: str,
    prompt: str,
    *,
    timeout: float,
    top_logprobs: int = 5,
) -> Optional[dict]:
    """Request generated token top-k logprob distributions for ``prompt``.

    Returns a dict with ``top`` (list of argmax tokens per position) and
    ``dists`` (list of ``{token: logprob}`` maps), or ``None`` when the
    endpoint does not return usable logprobs.
    """
    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs": top_logprobs,
    }
    payload = _post_json(endpoint, body, timeout)
    if not payload:
        return None
    choices = payload.get("choices") or [{}]
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    tokens = logprobs.get("tokens")
    top_lps = logprobs.get("top_logprobs")
    if not isinstance(tokens, list) or not isinstance(top_lps, list):
        return None

    top: list[str] = []
    dists: list[dict[str, float]] = []
    for tok, dist in zip(tokens, top_lps):
        top.append(tok)
        if isinstance(dist, dict):
            dists.append({k: float(v) for k, v in dist.items()})
        else:
            dists.append({})
    return {"top": top, "dists": dists}


def _tokenizers_match(quant: "ServerInfo", ref: "ServerInfo") -> bool:
    """Best-effort check that two endpoints share a tokenizer/model.

    Compares the served model names; differing model names imply a likely
    tokenizer mismatch that would make token-level KL meaningless.
    """
    a = getattr(quant, "model_name", None)
    b = getattr(ref, "model_name", None)
    if a is None or b is None:
        return True  # cannot determine -> do not block
    return a == b


def _run_kl(
    config: "BenchmarkConfig",
    server_info: "ServerInfo",
    model_name: str,
    ref_url: Optional[str],
) -> dict:
    """Token-level KL / top-1 agreement vs a reference endpoint."""
    if not ref_url:
        return {
            "mode": "kl",
            "status": "skipped",
            "reason": "no reference endpoint (ref_url) provided",
        }

    from vllm_benchmark.core.backends.detect import detect_backend

    timeout = float(getattr(config, "request_timeout", 60))
    quant_endpoint = _completions_endpoint(config.api_url)
    ref_endpoint = _completions_endpoint(ref_url)

    # Detect the reference server and check tokenizer compatibility.
    try:
        ref_backend = detect_backend(ref_url, forced=None)
        ref_info = ref_backend.server_info(_url_only_cfg(config, ref_url))
    except Exception as exc:
        return {
            "mode": "kl",
            "status": "skipped",
            "reason": f"could not query reference endpoint: {exc}",
        }

    tokenizer_mismatch = not _tokenizers_match(server_info, ref_info)

    corpus = load_perplexity_corpus()
    kls: list[float] = []
    agreements: list[float] = []
    for prompt in corpus:
        q_dist = _request_token_distributions(
            quant_endpoint, model_name, prompt, timeout=timeout
        )
        r_dist = _request_token_distributions(
            ref_endpoint, ref_info.model_name or model_name, prompt, timeout=timeout
        )
        if q_dist is None or r_dist is None:
            return {
                "mode": "kl",
                "status": "skipped",
                "reason": "an endpoint did not return usable logprobs",
                "tokenizer_mismatch": tokenizer_mismatch,
            }
        n = min(len(r_dist["dists"]), len(q_dist["dists"]))
        for i in range(n):
            kls.append(kl_divergence(r_dist["dists"][i], q_dist["dists"][i]))
        agreements.append(top1_agreement(r_dist["top"], q_dist["top"]))

    if not kls:
        return {
            "mode": "kl",
            "status": "skipped",
            "reason": "no comparable token positions",
            "tokenizer_mismatch": tokenizer_mismatch,
        }

    mean_kl = sum(kls) / len(kls)
    mean_agreement = sum(agreements) / len(agreements)
    return {
        "mode": "kl",
        "status": "ok",
        "mean_kl": mean_kl,
        "top1_agreement": mean_agreement,
        "n_positions": len(kls),
        "tokenizer_mismatch": tokenizer_mismatch,
    }


def _url_only_cfg(config: "BenchmarkConfig", url: str) -> Any:
    """Return a lightweight config-like object overriding only ``api_url``.

    The :class:`Backend` endpoint helpers only read ``cfg.api_url``, so a
    tiny shim is sufficient and avoids mutating the caller's config.
    """

    class _Cfg:
        api_url = url
        request_timeout = getattr(config, "request_timeout", 60)

    return _Cfg()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run_quality(
    mode: str,
    config: "BenchmarkConfig",
    server_info: "ServerInfo",
    *,
    ref_url: Optional[str] = None,
) -> dict:
    """Run quality measurement in the requested ``mode``.

    Args:
        mode: One of ``"off"``, ``"probe"``, ``"perplexity"`` or ``"kl"``.
        config: Benchmark configuration (provides ``api_url`` and timeouts).
        server_info: Normalized :class:`ServerInfo` for the server under test.
        ref_url: Reference endpoint base URL, required for ``kl`` mode.

    Returns:
        A self-describing result dict.  Always carries a ``mode`` key and a
        ``status`` of ``"ok"``, ``"skipped"`` or ``"disabled"``.  Capability
        gaps yield ``status == "skipped"`` with a ``reason`` — quality numbers
        are never fabricated.
    """
    model_name = (
        getattr(config, "model_name", None)
        or getattr(server_info, "model_name", None)
        or "unknown"
    )

    if mode == "off":
        return {"mode": "off", "status": "disabled"}
    if mode == "probe":
        return _run_probe(config, server_info, model_name)
    if mode == "perplexity":
        return _run_perplexity(config, server_info, model_name)
    if mode == "kl":
        return _run_kl(config, server_info, model_name, ref_url)

    return {
        "mode": mode,
        "status": "skipped",
        "reason": f"unknown quality mode: {mode!r}",
    }
