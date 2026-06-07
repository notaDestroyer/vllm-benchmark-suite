"""Coverage-oriented tests for model_intel internals.

Exercises the HuggingFace fetch wrapper (with a fake ``huggingface_hub``
module injected into ``sys.modules``), the data-file loaders, ``to_dict``,
GPU matching edge cases, and the param-finalization branches.

Author: amit
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from vllm_benchmark.analysis import model_intel as mi
from vllm_benchmark.core.backends.base import ServerInfo

FIXTURES = Path(__file__).parent / "fixtures" / "configs"


# ---------------------------------------------------------------------------
# Loaders & to_dict
# ---------------------------------------------------------------------------

def test_loaders_strip_readme() -> None:
    assert "_README" not in mi.load_gpu_specs()
    assert "_README" not in mi.load_models_kb()
    assert "H100 SXM" in mi.load_gpu_specs()
    assert "llama-3-8b" in mi.load_models_kb()


def test_profile_to_dict_keys() -> None:
    p = mi.ModelProfile(name="x", num_layers=32, notes=["n"])
    d = p.to_dict()
    assert d["name"] == "x"
    assert d["num_layers"] == 32
    assert d["notes"] == ["n"]
    assert set(d) >= {"family", "attention_type", "confidence", "source"}


def test_match_gpu_spec_none_and_unknown() -> None:
    assert mi.match_gpu_spec(None) is None
    assert mi.match_gpu_spec("") is None
    assert mi.match_gpu_spec("Intel Arc A770") is None


def test_match_gpu_spec_l4_not_l40s() -> None:
    # "L4" must not greedily match "L40S".
    spec = mi.match_gpu_spec("NVIDIA L4")
    assert spec is not None
    assert spec["_matched_key"] == "L4"


# ---------------------------------------------------------------------------
# _hf_fetch_config with a fake huggingface_hub
# ---------------------------------------------------------------------------

class _FakeHub:
    def __init__(self, files: dict[str, str]):
        self._files = files

    def hf_hub_download(self, repo_id, filename, token=None):
        if filename not in self._files:
            raise FileNotFoundError(filename)
        return self._files[filename]


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Install a fake huggingface_hub module with writable files."""
    cfg = json.loads((FIXTURES / "llama3_8b.json").read_text())
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    idx_path = tmp_path / "model.safetensors.index.json"
    idx_path.write_text(json.dumps({"metadata": {"total_size": 16_060_000_000}}))

    files = {
        "config.json": str(cfg_path),
        "model.safetensors.index.json": str(idx_path),
    }
    mod = types.ModuleType("huggingface_hub")
    hub = _FakeHub(files)
    mod.hf_hub_download = hub.hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return cfg


def test_hf_fetch_config_success(fake_hub) -> None:
    config, total = mi._hf_fetch_config("meta-llama/Meta-Llama-3-8B", hf_token="tok")
    assert config is not None
    assert config["num_hidden_layers"] == 32
    assert total == 16_060_000_000


def test_hf_fetch_config_no_index(monkeypatch, tmp_path) -> None:
    cfg = json.loads((FIXTURES / "llama3_8b.json").read_text())
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    mod = types.ModuleType("huggingface_hub")
    mod.hf_hub_download = _FakeHub({"config.json": str(cfg_path)}).hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    config, total = mi._hf_fetch_config("repo", None)
    assert config is not None
    assert total is None  # index download raised -> None


def test_hf_fetch_config_no_hub(monkeypatch) -> None:
    # Simulate huggingface_hub being unimportable.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    config, total = mi._hf_fetch_config("repo", None)
    assert config is None and total is None


def test_build_profile_hf_live_through_fake_hub(fake_hub) -> None:
    info = ServerInfo(backend="vllm", model_name="meta-llama/Meta-Llama-3-8B",
                      kv_cache_dtype="bf16", quantization="FP8")
    profile = mi.build_profile(info, allow_network=True, hf_token="tok")
    assert profile.source == "hf_live"
    assert profile.weight_bytes == 16_060_000_000
    # Quantization note recorded from server.
    assert any("quantization=FP8" in n for n in profile.notes)


# ---------------------------------------------------------------------------
# Finalize-params branches
# ---------------------------------------------------------------------------

def test_finalize_weight_bytes_from_params() -> None:
    # KB dense model -> weight_bytes derived from total_params * 2.
    info = ServerInfo(backend="vllm", model_name="meta-llama/Llama-3-8B", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=False)
    assert profile.weight_bytes == profile.total_params * 2


def test_finalize_kb_supplies_active_when_missing(monkeypatch) -> None:
    # A config (no params) for a model whose KB entry has active_params.
    cfg = json.loads((FIXTURES / "qwen3_30b_a3b.json").read_text())
    monkeypatch.setattr(mi, "_hf_fetch_config", lambda r, t: (cfg, None))
    info = ServerInfo(backend="vllm", model_name="Qwen/Qwen3-30B-A3B", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=True)
    assert profile.total_params == 30_500_000_000
    assert profile.active_params is not None


# ---------------------------------------------------------------------------
# parse_model_name extra branches
# ---------------------------------------------------------------------------

def test_parse_model_name_mqa_falcon_family() -> None:
    parsed = mi.parse_model_name("tiiuae/falcon-40b")
    assert parsed["family"] == "falcon"
    assert parsed["total_params"] == 40_000_000_000


def test_compute_active_params_moe_missing_dims_returns_none() -> None:
    # MoE flagged but no hidden/inter/layers and no active_params -> None.
    out = mi.compute_active_params(
        {"num_experts": 8, "experts_per_tok": 2, "total_params": 1}
    )
    assert out is None


def test_attention_type_none_on_missing() -> None:
    assert mi.attention_type_of(None, 8) is None
    assert mi.attention_type_of(32, None) is None


def test_roofline_negative_inputs_return_none() -> None:
    # active_params <= 0 in mfu, and zero bytes/param in critical_batch.
    assert mi.mfu(100.0, -1, 989.0) is None
    assert mi.critical_batch(989.0, 3350.0, 0.0) is None


def test_build_profile_mha_config_no_kv_heads(monkeypatch) -> None:
    # gpt_neox/dense_mha fixture has num_key_value_heads set; craft one
    # without it to exercise the n_kv = n_heads fallback.
    cfg = {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
    }
    monkeypatch.setattr(mi, "_hf_fetch_config", lambda r, t: (cfg, None))
    info = ServerInfo(backend="vllm", model_name="gpt2-medium", kv_cache_dtype="fp16")
    profile = mi.build_profile(info, allow_network=True)
    assert profile.num_kv_heads == 12  # fell back to attention heads
    assert profile.attention_type == "MHA"
    assert profile.head_dim == 64  # 768 / 12


def test_finalize_params_never_raises(monkeypatch) -> None:
    # Force _finalize_params to blow up and confirm build_profile swallows it.
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mi, "_finalize_params", boom)
    info = ServerInfo(backend="vllm", model_name="meta-llama/Llama-3-8B", kv_cache_dtype="bf16")
    profile = mi.build_profile(info, allow_network=False)
    assert profile is not None
