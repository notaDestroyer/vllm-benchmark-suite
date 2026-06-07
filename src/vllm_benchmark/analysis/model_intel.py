"""Model intelligence and roofline (MBU/MFU) analysis.

This module turns a backend :class:`ServerInfo` into a rich
:class:`ModelProfile` describing the served model's architecture
(layers, attention type, MoE structure, parameter counts, KV-cache cost)
and provides the analytic building blocks for roofline / arithmetic-
intensity analysis used by the bottleneck detector.

Provenance is tracked honestly: every profile records where its facts
came from (live HuggingFace fetch, the curated offline knowledge base,
the server itself, or pure name heuristics) and how confident we are.
Network access is optional and *never* fatal — :func:`build_profile`
always returns a profile, falling back through the KB and name parser.

Roofline math (all units made explicit in the docstrings):

* **MBU** (Memory Bandwidth Utilization) compares measured single-user
  decode throughput against the bandwidth-bound ceiling, i.e. how many
  tokens/s the HBM bandwidth alone would allow given the bytes that must
  be read per generated token (weights + KV cache).
* **MFU** (Model FLOPs Utilization) compares measured prefill throughput
  against the compute-bound ceiling derived from the GPU's peak FLOPS.
  The ``2 * active_params`` flops/token figure is the standard forward-
  pass approximation and is accurate to roughly +/-15% (it ignores
  attention score FLOPs, layernorm, and the LM head, and assumes a
  single fused multiply-add counts as 2 FLOPs).

Author: amit
License: MIT
"""

from __future__ import annotations

import importlib.resources
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Optional

from vllm_benchmark.core.backends.base import ServerInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bytes per parameter / per KV element by dtype or quantization name.
_BYTES_BY_DTYPE = {
    "fp8": 1.0,
    "f8": 1.0,
    "int8": 1.0,
    "i8": 1.0,
    "int4": 0.5,
    "i4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "f16": 2.0,
    "half": 2.0,
    "fp32": 4.0,
    "float32": 4.0,
    "f32": 4.0,
}


# ---------------------------------------------------------------------------
# ModelProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """A normalized, provenance-aware description of a served model.

    Numeric architecture fields are ``None`` when unknown.  The
    ``param_source``, ``confidence`` and ``source`` fields document how
    the profile was derived so downstream consumers can weight it.
    """

    name: str
    family: Optional[str] = None
    is_moe: Optional[bool] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_type: Optional[Literal["MHA", "GQA", "MQA"]] = None
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None
    num_experts: Optional[int] = None
    experts_per_tok: Optional[int] = None
    num_shared_experts: Optional[int] = None
    total_params: Optional[int] = None
    active_params: Optional[int] = None
    param_source: Optional[
        Literal["safetensors_index", "config_estimate", "kb", "name_heuristic"]
    ] = None
    weight_bytes: Optional[int] = None
    kv_bytes_per_token: Optional[int] = None
    confidence: Literal["confirmed", "inferred", "heuristic"] = "heuristic"
    source: Literal["hf_live", "kb", "server", "name"] = "name"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation of this profile."""
        return {
            "name": self.name,
            "family": self.family,
            "is_moe": self.is_moe,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "attention_type": self.attention_type,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "num_experts": self.num_experts,
            "experts_per_tok": self.experts_per_tok,
            "num_shared_experts": self.num_shared_experts,
            "total_params": self.total_params,
            "active_params": self.active_params,
            "param_source": self.param_source,
            "weight_bytes": self.weight_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "confidence": self.confidence,
            "source": self.source,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Data-file loaders (packaged JSON via importlib.resources)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_json_data(filename: str) -> dict:
    """Load a packaged JSON data file from ``vllm_benchmark/data``."""
    resource = importlib.resources.files("vllm_benchmark.data").joinpath(filename)
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_models_kb() -> dict:
    """Return the curated offline model knowledge base (without README)."""
    data = dict(_load_json_data("models_kb.json"))
    data.pop("_README", None)
    return data


@lru_cache(maxsize=1)
def load_gpu_specs() -> dict:
    """Return the GPU hardware specifications (without README key)."""
    data = dict(_load_json_data("gpu_specs.json"))
    data.pop("_README", None)
    return data


def match_gpu_spec(gpu_name: Optional[str]) -> Optional[dict]:
    """Fuzzy-match a detected GPU name to a spec entry.

    Args:
        gpu_name: Raw GPU name, e.g. ``"NVIDIA H100 80GB HBM3"``.

    Returns:
        The matching spec dict (with an added ``"_matched_key"``), or
        ``None`` when no entry matches.
    """
    if not gpu_name:
        return None
    specs = load_gpu_specs()
    cleaned = gpu_name.lower().replace("nvidia ", "")
    # Normalize separators so "A100-SXM4-80GB" and "A100 80GB" tokenize alike.
    norm = re.sub(r"[-_/]", " ", cleaned)
    detected_tokens = set(norm.split())

    best_key: Optional[str] = None
    best_score = 0.0
    for key in specs:
        key_norm = re.sub(r"[-_/]", " ", key.lower())
        key_tokens = [t for t in key_norm.split() if t]
        if not key_tokens:
            continue
        # Whole-key substring match is the strongest signal.
        if key_norm in norm:
            score = 100 + len(key_norm)
        else:
            matched = sum(1 for t in key_tokens if t in detected_tokens)
            if matched == 0:
                continue
            # Require the model token (first) to be present.
            if key_tokens[0] not in detected_tokens:
                continue
            score = matched
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is None:
        return None
    spec = dict(specs[best_key])
    spec["_matched_key"] = best_key
    return spec


# ---------------------------------------------------------------------------
# Name parsing heuristics
# ---------------------------------------------------------------------------

_FAMILY_PATTERNS = [
    ("llama", "llama"),
    ("mixtral", "mixtral"),
    ("mistral", "mistral"),
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
    ("gemma", "gemma"),
    ("phi", "phi"),
    ("falcon", "falcon"),
    ("yi", "yi"),
    ("command-r", "command-r"),
]

_QUANT_PATTERNS = [
    ("fp8", "FP8"),
    ("awq", "AWQ"),
    ("gptq", "GPTQ"),
    ("int4", "INT4"),
    ("int8", "INT8"),
    ("w4a16", "INT4"),
    ("w8a8", "INT8"),
]


def parse_model_name(name: Optional[str]) -> dict:
    """Extract architecture hints from a model repo id / name string.

    Recognizes family, parameter size (``"7B"``, ``"70B"``), MoE shapes
    (``"8x7B"`` -> 8 experts, ``"30B-A3B"`` -> 30B total / 3B active),
    quantization (FP8/AWQ/GPTQ/INT4/INT8) and the instruct-vs-base tag.

    Args:
        name: Model name or HuggingFace repo id.

    Returns:
        A dict of best-effort hints; missing values are omitted.  Always
        contains the original ``"name"``.
    """
    out: dict[str, Any] = {"name": name or ""}
    if not name:
        return out
    lower = name.lower()

    for needle, fam in _FAMILY_PATTERNS:
        if needle in lower:
            out["family"] = fam
            break

    for needle, quant in _QUANT_PATTERNS:
        if needle in lower:
            out["quant"] = quant
            break

    # MoE "AxB" shape, e.g. 8x7b -> 8 experts of ~7B each.
    moe_match = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b", lower)
    if moe_match:
        out["is_moe"] = True
        out["num_experts"] = int(moe_match.group(1))
        out["expert_size_b"] = float(moe_match.group(2))

    # "30B-A3B" style total/active hint.
    active_match = re.search(r"(\d+(?:\.\d+)?)\s*b[-_ ]*a(\d+(?:\.\d+)?)\s*b", lower)
    if active_match:
        out["is_moe"] = True
        out["total_params"] = int(float(active_match.group(1)) * 1e9)
        out["active_params"] = int(float(active_match.group(2)) * 1e9)
    elif "is_moe" not in out:
        # Plain "<N>B" dense size hint (avoid matching the MoE forms above).
        size_match = re.search(r"(?<![\dx])(\d+(?:\.\d+)?)\s*b\b", lower)
        if size_match:
            out["total_params"] = int(float(size_match.group(1)) * 1e9)

    if "instruct" in lower or "-it" in lower or "chat" in lower:
        out["variant"] = "instruct"
    else:
        out["variant"] = "base"

    return out


# ---------------------------------------------------------------------------
# Pure architecture helpers
# ---------------------------------------------------------------------------

def bytes_per_param(quant_or_dtype: Optional[str]) -> float:
    """Return bytes per parameter for a dtype/quantization label.

    FP8/INT8 -> 1.0, INT4/AWQ/GPTQ -> 0.5, BF16/FP16 -> 2.0, FP32 -> 4.0.
    Unknown / ``None`` defaults to 2.0 (the common 16-bit serving case).
    """
    if not quant_or_dtype:
        return 2.0
    key = str(quant_or_dtype).lower().strip()
    if key in _BYTES_BY_DTYPE:
        return _BYTES_BY_DTYPE[key]
    # Substring fallback (handles "fp8_e4m3", "compressed-tensors-int4", ...).
    for needle, val in _BYTES_BY_DTYPE.items():
        if needle in key:
            return val
    return 2.0


def attention_type_of(
    num_attention_heads: Optional[int],
    num_kv_heads: Optional[int],
) -> Optional[Literal["MHA", "GQA", "MQA"]]:
    """Classify the attention scheme from head counts.

    * ``num_kv_heads == 1`` -> ``"MQA"`` (multi-query).
    * ``num_kv_heads == num_attention_heads`` -> ``"MHA"`` (multi-head).
    * otherwise -> ``"GQA"`` (grouped-query).

    Returns ``None`` if either count is missing.
    """
    if not num_attention_heads or not num_kv_heads:
        return None
    if num_kv_heads == 1:
        return "MQA"
    if num_kv_heads >= num_attention_heads:
        return "MHA"
    return "GQA"


def kv_bytes_per_token(
    num_layers: Optional[int],
    num_kv_heads: Optional[int],
    head_dim: Optional[int],
    kv_dtype: Optional[str],
) -> Optional[int]:
    """Bytes of KV cache consumed per token across all layers.

    Formula: ``2 * layers * kv_heads * head_dim * bytes(kv_dtype)`` where
    the leading 2 accounts for storing both the K and V tensors.

    Returns ``None`` if any structural input is missing.
    """
    if not num_layers or not num_kv_heads or not head_dim:
        return None
    bpp = bytes_per_param(kv_dtype)
    return int(2 * num_layers * num_kv_heads * head_dim * bpp)


def compute_active_params(config_or_kb: dict) -> Optional[int]:
    """Estimate the number of *active* parameters per forward token.

    For dense models this is the total parameter count.  For MoE models
    it is the always-on backbone (attention + router + embeddings +
    shared experts) plus the parameters of the ``experts_per_tok`` routed
    experts that actually fire for a given token::

        active = total - (num_experts - experts_per_tok - shared) * per_expert

    where ``per_expert`` is estimated from ``moe_intermediate_size``
    (falling back to ``intermediate_size``) and ``hidden_size`` as the
    three projection matrices of a SwiGLU expert (``3 * hidden * inter``).

    Args:
        config_or_kb: A HF ``config.json``-like dict or a KB entry.  May
            include ``total_params`` directly.

    Returns:
        Active parameter count, or ``None`` if it cannot be estimated.
    """
    c = config_or_kb
    num_experts = c.get("num_experts") or c.get("num_local_experts")
    experts_per_tok = c.get("experts_per_tok") or c.get("num_experts_per_tok")
    is_moe = bool(num_experts) and bool(experts_per_tok)
    total = c.get("total_params")

    if not is_moe:
        # Dense: active == total.
        return int(total) if total else None

    shared = c.get("num_shared_experts") or 0
    hidden = c.get("hidden_size")
    inter = c.get("moe_intermediate_size") or c.get("intermediate_size")
    layers = c.get("num_layers") or c.get("num_hidden_layers")

    # Direct KB active count is most trustworthy.
    if c.get("active_params"):
        return int(c["active_params"])

    if total and hidden and inter and layers:
        # 3 matrices per SwiGLU expert (gate, up, down).
        per_expert = 3 * hidden * inter
        inactive_experts = max(0, int(num_experts) - int(experts_per_tok) - int(shared))
        inactive_params = inactive_experts * per_expert * int(layers)
        active = int(total) - inactive_params
        if active > 0:
            return active
    return None


# ---------------------------------------------------------------------------
# HuggingFace config / safetensors fetch (mockable, never fatal)
# ---------------------------------------------------------------------------

def _hf_fetch_config(
    repo_id: str,
    hf_token: Optional[str],
) -> tuple[Optional[dict], Optional[int]]:
    """Download ``config.json`` and the safetensors total size for a repo.

    Wraps :func:`huggingface_hub.hf_hub_download` so tests can monkeypatch
    it.  Any failure (offline, gated, missing file) returns ``(None, ...)``
    without raising.

    Returns:
        ``(config_dict_or_None, total_size_bytes_or_None)``.
    """
    config: Optional[dict] = None
    total_size: Optional[int] = None
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None, None

    token = hf_token or os.environ.get("HF_TOKEN")

    try:
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        config = None

    try:
        idx_path = hf_hub_download(
            repo_id=repo_id, filename="model.safetensors.index.json", token=token
        )
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        meta = index.get("metadata", {})
        if meta.get("total_size"):
            total_size = int(meta["total_size"])
    except Exception:
        total_size = None

    return config, total_size


def _profile_from_config(name: str, config: dict, server: ServerInfo) -> ModelProfile:
    """Build a profile from a HF ``config.json``-like dict."""
    num_layers = config.get("num_hidden_layers") or config.get("num_layers")
    hidden = config.get("hidden_size")
    n_heads = config.get("num_attention_heads")
    n_kv = config.get("num_key_value_heads")
    if n_kv is None:
        # MHA models often omit num_key_value_heads.
        n_kv = n_heads
    head_dim = config.get("head_dim")
    if not head_dim and hidden and n_heads:
        head_dim = hidden // n_heads

    num_experts = config.get("num_experts") or config.get("num_local_experts")
    experts_per_tok = config.get("experts_per_tok") or config.get("num_experts_per_tok")
    is_moe = bool(num_experts) and bool(experts_per_tok)

    kv_dtype = server.kv_cache_dtype or server.dtype
    kvbpt = kv_bytes_per_token(num_layers, n_kv, head_dim, kv_dtype)

    profile = ModelProfile(
        name=name,
        family=parse_model_name(name).get("family"),
        is_moe=is_moe,
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        attention_type=attention_type_of(n_heads, n_kv),
        vocab_size=config.get("vocab_size"),
        max_position_embeddings=config.get("max_position_embeddings"),
        rope_theta=config.get("rope_theta"),
        rope_scaling=config.get("rope_scaling"),
        num_experts=num_experts,
        experts_per_tok=experts_per_tok,
        num_shared_experts=config.get("num_shared_experts")
        or config.get("shared_expert_intermediate_size") and 1
        or 0,
        kv_bytes_per_token=kvbpt,
        confidence="confirmed",
        source="hf_live",
        param_source="config_estimate",
    )
    return profile


def _kb_lookup(name: str) -> Optional[tuple[str, dict]]:
    """Return ``(matched_key, kb_entry)`` for the longest matching KB key."""
    if not name:
        return None
    lower = name.lower()
    kb = load_models_kb()
    best: Optional[str] = None
    for key in kb:
        if key in lower:
            if best is None or len(key) > len(best):
                best = key
    if best is None:
        return None
    return best, kb[best]


def _profile_from_kb(name: str, entry: dict, server: ServerInfo) -> ModelProfile:
    """Build a profile from a KB entry."""
    n_heads = entry.get("num_attention_heads")
    n_kv = entry.get("num_kv_heads")
    hidden = entry.get("hidden_size")
    head_dim = hidden // n_heads if hidden and n_heads else None
    kv_dtype = server.kv_cache_dtype or server.dtype
    kvbpt = kv_bytes_per_token(entry.get("num_layers"), n_kv, head_dim, kv_dtype)
    return ModelProfile(
        name=name,
        family=entry.get("family"),
        is_moe=entry.get("is_moe"),
        num_layers=entry.get("num_layers"),
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        attention_type=attention_type_of(n_heads, n_kv),
        num_experts=entry.get("num_experts"),
        experts_per_tok=entry.get("experts_per_tok"),
        num_shared_experts=entry.get("num_shared_experts") or 0,
        total_params=entry.get("total_params"),
        active_params=entry.get("active_params"),
        kv_bytes_per_token=kvbpt,
        confidence="inferred",
        source="kb",
        param_source="kb",
    )


def _profile_from_name(name: str, server: ServerInfo) -> ModelProfile:
    """Build a low-confidence profile from name heuristics only."""
    hints = parse_model_name(name)
    return ModelProfile(
        name=name,
        family=hints.get("family"),
        is_moe=hints.get("is_moe"),
        num_experts=hints.get("num_experts"),
        total_params=hints.get("total_params"),
        active_params=hints.get("active_params"),
        confidence="heuristic",
        source="name",
        param_source="name_heuristic",
        notes=["Profile derived from name heuristics only; architecture unknown."],
    )


def _finalize_params(profile: ModelProfile, config: Optional[dict], total_size: Optional[int]) -> None:
    """Fill total/active params + weight bytes and cross-check against KB."""
    # Total params from safetensors index (most authoritative).
    if total_size and profile.total_params is None:
        # safetensors total_size is bytes; convert with the serving dtype.
        # We cannot know per-param bytes here precisely, so record the byte
        # size directly and leave param-count to KB/config where possible.
        profile.weight_bytes = int(total_size)

    # Active params estimate from config.
    if config is not None:
        cfg = dict(config)
        cfg.setdefault("num_layers", profile.num_layers)
        cfg.setdefault("hidden_size", profile.hidden_size)
        if profile.total_params is not None:
            cfg["total_params"] = profile.total_params
        est = compute_active_params(cfg)
        if est is not None:
            profile.active_params = est

    # Cross-check active estimate against KB published value.
    kb_match = _kb_lookup(profile.name)
    if kb_match is not None:
        _, entry = kb_match
        if profile.total_params is None and entry.get("total_params"):
            profile.total_params = entry["total_params"]
        kb_active = entry.get("active_params")
        if (
            kb_active
            and profile.active_params
            and abs(profile.active_params - kb_active) / kb_active > 0.10
        ):
            profile.confidence = "inferred"
            profile.notes.append(
                f"Active-param estimate {profile.active_params:,} disagrees with "
                f"KB value {kb_active:,} by >10%; downgraded confidence."
            )
            profile.active_params = kb_active
        elif profile.active_params is None and kb_active:
            profile.active_params = kb_active

    # Dense fallback: active == total.
    if profile.active_params is None and profile.is_moe is False and profile.total_params:
        profile.active_params = profile.total_params

    # Derive weight_bytes from params when we have a dtype.
    if profile.weight_bytes is None and profile.total_params:
        # Use bytes-per-param of serving dtype if available; default 2.
        profile.weight_bytes = int(profile.total_params * 2)


def build_profile(
    server_info: ServerInfo,
    *,
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> ModelProfile:
    """Build a :class:`ModelProfile` from server info, never raising.

    Resolution order:

    1. Pipeline server hints (model name / path).
    2. Live HuggingFace fetch of ``config.json`` + safetensors index
       (only when ``allow_network`` is true).  Network/gated failures are
       swallowed.
    3. Curated offline KB.
    4. Pure name heuristics.

    The returned profile always reflects the *best available* source and
    records its ``confidence`` and ``source`` honestly.

    Args:
        server_info: Normalized server description from a backend.
        hf_token: Optional HuggingFace token (else ``HF_TOKEN`` env).
        allow_network: When ``False``, skip the live fetch entirely.

    Returns:
        A populated :class:`ModelProfile` (possibly mostly ``None`` with
        ``confidence == "heuristic"``).
    """
    name = server_info.model_name or server_info.served_model_path or "unknown"

    config: Optional[dict] = None
    total_size: Optional[int] = None
    if allow_network and name and name != "unknown":
        try:
            config, total_size = _hf_fetch_config(name, hf_token)
        except Exception:
            config, total_size = None, None

    if config is not None:
        profile = _profile_from_config(name, config, server_info)
    else:
        kb_match = _kb_lookup(name)
        if kb_match is not None:
            profile = _profile_from_kb(name, kb_match[1], server_info)
        else:
            profile = _profile_from_name(name, server_info)

    # Overlay confirmed server facts where present.
    if server_info.quantization:
        profile.notes.append(f"Server reports quantization={server_info.quantization}.")

    try:
        _finalize_params(profile, config, total_size)
    except Exception:
        # Never let param accounting raise.
        pass

    return profile


# ---------------------------------------------------------------------------
# Roofline metrics (MBU / MFU / critical batch)
# ---------------------------------------------------------------------------

def mbu(
    measured_single_user_decode_tps: Optional[float],
    active_params: Optional[int],
    kv_bytes_per_token_val: Optional[int],
    seq_len: Optional[int],
    hbm_bandwidth_gbps: Optional[float],
    bytes_per_param_val: Optional[float],
) -> Optional[float]:
    """Memory Bandwidth Utilization for single-user decode.

    Each generated token must read the model weights plus the accumulated
    KV cache from HBM::

        bytes_per_token = active_params * bytes_per_param
                          + kv_bytes_per_token * seq_len
        ceiling_tps     = (hbm_bandwidth_gbps * 1e9) / bytes_per_token
        MBU             = measured_decode_tps / ceiling_tps

    Args:
        measured_single_user_decode_tps: Observed decode tokens/s at B=1.
        active_params: Active parameters per token.
        kv_bytes_per_token_val: KV bytes read per token per cache slot.
        seq_len: Current sequence length (KV slots in cache).
        hbm_bandwidth_gbps: Peak HBM bandwidth in GB/s (1 GB = 1e9 bytes).
        bytes_per_param_val: Bytes per weight element.

    Returns:
        The utilization ratio, or ``None`` if any input is missing /
        non-positive.  No value is fabricated.
    """
    if (
        measured_single_user_decode_tps is None
        or active_params is None
        or kv_bytes_per_token_val is None
        or seq_len is None
        or hbm_bandwidth_gbps is None
        or bytes_per_param_val is None
    ):
        return None
    bytes_per_token = active_params * bytes_per_param_val + kv_bytes_per_token_val * seq_len
    if bytes_per_token <= 0 or hbm_bandwidth_gbps <= 0:
        return None
    hbm_bps = hbm_bandwidth_gbps * 1e9
    ceiling = hbm_bps / bytes_per_token
    if ceiling <= 0:
        return None
    return measured_single_user_decode_tps / ceiling


def mfu(
    measured_prefill_tps: Optional[float],
    active_params: Optional[int],
    peak_flops_tflops_for_dtype: Optional[float],
) -> Optional[float]:
    """Model FLOPs Utilization for the prefill (compute-bound) phase.

    Uses the standard forward-pass approximation of ``2 * active_params``
    FLOPs per token (one multiply + one add per parameter).  This ignores
    attention-score FLOPs, normalization and the LM head, and is accurate
    to roughly +/-15%::

        flops_per_token = 2 * active_params
        ceiling_tps     = (peak_flops_tflops * 1e12) / flops_per_token
        MFU             = measured_prefill_tps / ceiling_tps

    The ``peak_flops_tflops_for_dtype`` must match the compute dtype of
    the run (e.g. FP8 peak for an FP8-served model).

    Returns:
        The utilization ratio, or ``None`` if any input is missing /
        non-positive.
    """
    if (
        measured_prefill_tps is None
        or active_params is None
        or peak_flops_tflops_for_dtype is None
    ):
        return None
    if active_params <= 0 or peak_flops_tflops_for_dtype <= 0:
        return None
    flops_per_token = 2 * active_params
    peak_flops = peak_flops_tflops_for_dtype * 1e12
    ceiling = peak_flops / flops_per_token
    if ceiling <= 0:
        return None
    return measured_prefill_tps / ceiling


def critical_batch(
    peak_flops_tflops: Optional[float],
    hbm_bandwidth_gbps: Optional[float],
    bytes_per_param_val: Optional[float],
) -> Optional[int]:
    """Critical batch size B* where decode transitions compute<->bandwidth.

    The ridge point of the roofline (the batch size at which the GEMM
    becomes compute-bound rather than weight-bandwidth-bound) is the
    hardware's operational-intensity break-even scaled by bytes/param::

        ops_per_byte = peak_flops / hbm_bandwidth        (FLOP/byte)
        B*           = round(ops_per_byte * bytes_per_param / 2)

    The ``/2`` reflects the 2 FLOPs per parameter per token convention.
    Units are converted consistently (TFLOPS->FLOPS, GB/s->bytes/s).

    Returns:
        The integer critical batch size, or ``None`` on missing inputs.
    """
    if (
        peak_flops_tflops is None
        or hbm_bandwidth_gbps is None
        or bytes_per_param_val is None
    ):
        return None
    if peak_flops_tflops <= 0 or hbm_bandwidth_gbps <= 0 or bytes_per_param_val <= 0:
        return None
    peak_flops = peak_flops_tflops * 1e12
    hbm_bps = hbm_bandwidth_gbps * 1e9
    ops_per_byte = peak_flops / hbm_bps
    return int(round(ops_per_byte * bytes_per_param_val / 2))
