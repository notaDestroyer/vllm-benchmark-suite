"""KB head_dim correctness, MLA KV-cache math and quantization-aware weight bytes.

``hidden_size // num_attention_heads`` is wrong for Gemma-2 (256 / 128)
and Qwen3-MoE (128), and meaningless for DeepSeek-V3's MLA, which caches a
single compressed latent per layer.  Those errors propagated into
``kv_bytes_per_token`` and therefore MBU and the bottleneck verdicts.
"""

from __future__ import annotations

import pytest

from vllm_benchmark.analysis import model_intel as mi
from vllm_benchmark.core.backends.base import ServerInfo


def _kb_profile(name: str, **server) -> mi.ModelProfile:
    info = ServerInfo(backend="vllm", model_name=name, kv_cache_dtype="bf16", **server)
    return mi.build_profile(info, allow_network=False)


@pytest.mark.parametrize(
    "name, head_dim, kv_bytes",
    [
        ("google/gemma-2-9b-it", 256, 2 * 42 * 8 * 256 * 2),
        ("google/gemma-2-27b-it", 128, 2 * 46 * 16 * 128 * 2),
        ("Qwen/Qwen3-30B-A3B", 128, 2 * 48 * 4 * 128 * 2),
        ("Qwen/Qwen3-235B-A22B", 128, 2 * 94 * 4 * 128 * 2),
        ("meta-llama/Llama-3-8B", 128, 2 * 32 * 8 * 128 * 2),
        ("microsoft/Phi-3-mini-4k-instruct", 96, 2 * 32 * 32 * 96 * 2),
    ],
)
def test_kb_head_dim_and_kv_bytes(name, head_dim, kv_bytes):
    p = _kb_profile(name)
    assert p.source == "kb"
    assert p.head_dim == head_dim
    assert p.kv_bytes_per_token == kv_bytes


def test_deepseek_v3_uses_mla_latent():
    p = _kb_profile("deepseek-ai/DeepSeek-V3")
    assert p.source == "kb"
    # 61 layers * (512 + 64) latent * 2 bytes, no K/V doubling.
    assert p.kv_bytes_per_token == 61 * 576 * 2
    assert p.head_dim == 192


def test_every_kb_entry_has_explicit_head_dim():
    kb = mi.load_models_kb()
    for key, entry in kb.items():
        assert entry.get("head_dim"), key


def test_mla_helpers():
    assert mi.mla_latent_dim({"kv_lora_rank": 512, "qk_rope_head_dim": 64}) == 576
    assert mi.mla_latent_dim({"kv_lora_rank": 512}) == 512
    assert mi.mla_latent_dim({"hidden_size": 4096}) is None
    assert mi.mla_kv_bytes_per_token(61, 576, "bf16") == 61 * 576 * 2
    assert mi.mla_kv_bytes_per_token(61, 576, "fp8") == 61 * 576
    assert mi.mla_kv_bytes_per_token(None, 576, "bf16") is None


def test_hf_config_with_mla_fields():
    cfg = {
        "architectures": ["DeepseekV3ForCausalLM"], "num_hidden_layers": 4, "hidden_size": 1024,
        "num_attention_heads": 8, "num_key_value_heads": 8, "kv_lora_rank": 512,
        "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
    }
    info = ServerInfo(backend="vllm", model_name="x/mla-test", kv_cache_dtype="bf16")
    p = mi._profile_from_config("x/mla-test", cfg, info)
    assert p.kv_bytes_per_token == 4 * 576 * 2


@pytest.mark.parametrize(
    "quant, bpp",
    [(None, 2.0), ("FP16/BF16", 2.0), ("FP8", 1.0), ("fp8_e4m3", 1.0), ("AWQ", 0.5), ("GPTQ", 0.5), ("INT8", 1.0)],
)
def test_weight_bytes_follow_quantization(quant, bpp):
    p = _kb_profile("meta-llama/Llama-3-8B", quantization=quant)
    assert p.weight_bytes == int(p.total_params * bpp)
    if bpp != 2.0:
        assert any("weight_bytes assumes" in n for n in p.notes)
