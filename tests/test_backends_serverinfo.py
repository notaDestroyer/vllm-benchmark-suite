"""Tests for backend ``server_info`` field mapping.

Feeds representative mocked endpoint bodies and asserts that the
normalized :class:`ServerInfo` fields are populated correctly for both
the vLLM and SGLang backends.

Author: amit
"""

from unittest.mock import MagicMock, patch

from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core.backends import SGLangBackend, VLLMBackend


def _resp(status: int = 200, json_data=None, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.return_value = json_data if json_data is not None else {}
    return r


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


_VLLM_METRICS = """# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
vllm:gpu_cache_usage_perc 0.42
vllm:num_requests_running 7.0
vllm:spec_decode_num_accepted_tokens_total 800.0
vllm:spec_decode_num_draft_tokens_total 1000.0
"""


def test_vllm_server_info_maps_fields():
    def fake_get(url, timeout=5):
        if url.endswith("/v1/models"):
            return _resp(200, {"data": [{"id": "meta-llama/Llama-3.1-8B-FP8", "max_model_len": 131072}]})
        if url.endswith("/version"):
            return _resp(200, {"version": "0.6.3"})
        if url.endswith("/metrics"):
            return _resp(200, text=_VLLM_METRICS)
        return _resp(404)

    cfg = BenchmarkConfig(api_url="http://localhost:8000")
    with patch("requests.get", side_effect=fake_get):
        info = VLLMBackend(cfg.api_url).server_info(cfg)

    assert info.backend == "vllm"
    assert info.backend_version == "0.6.3"
    assert info.model_name == "meta-llama/Llama-3.1-8B-FP8"
    assert info.max_model_len == 131072
    assert info.quantization == "FP8"
    assert info.raw["kv_cache_usage"] == 0.42
    assert info.raw["running_requests"] == 7
    # Speculative acceptance rate = accepted / draft = 800 / 1000
    assert info.speculative is not None
    assert abs(info.speculative["acceptance_rate"] - 0.8) < 1e-9


def test_vllm_server_info_no_speculative_when_absent():
    metrics = "vllm:gpu_cache_usage_perc 0.1\n"

    def fake_get(url, timeout=5):
        if url.endswith("/v1/models"):
            return _resp(200, {"data": [{"id": "some-model"}]})
        if url.endswith("/metrics"):
            return _resp(200, text=metrics)
        return _resp(404)

    cfg = BenchmarkConfig(api_url="http://localhost:8000")
    with patch("requests.get", side_effect=fake_get):
        info = VLLMBackend(cfg.api_url).server_info(cfg)

    assert info.speculative is None
    assert info.quantization == "FP16/BF16"


# ---------------------------------------------------------------------------
# SGLang
# ---------------------------------------------------------------------------


def test_sglang_server_info_maps_fields():
    server_info_body = {
        "tp_size": 4,
        "dp_size": 2,
        "max_running_requests": 256,
        "attention_backend": "flashinfer",
        "version": "0.4.1",
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
    }
    model_info_body = {
        "model_path": "/models/Qwen2.5-7B-AWQ",
        "tokenizer_path": "/models/Qwen2.5-7B-AWQ",
        "is_generation": True,
    }

    def fake_get(url, timeout=5):
        if url.endswith("/v1/models"):
            return _resp(404)
        if url.endswith("/get_server_info"):
            return _resp(200, server_info_body)
        if url.endswith("/get_model_info"):
            return _resp(200, model_info_body)
        return _resp(404)

    cfg = BenchmarkConfig(api_url="http://localhost:30000")
    with patch("requests.get", side_effect=fake_get):
        info = SGLangBackend(cfg.api_url).server_info(cfg)

    assert info.backend == "sglang"
    assert info.backend_version == "0.4.1"
    assert info.tensor_parallel == 4
    assert info.max_num_seqs == 256
    assert info.served_model_path == "/models/Qwen2.5-7B-AWQ"
    assert info.model_name == "/models/Qwen2.5-7B-AWQ"
    assert info.task == "generate"
    assert info.quantization == "AWQ"
    assert info.raw["dp_size"] == 2
    assert info.raw["attention_backend"] == "flashinfer"
    assert info.raw["tokenizer_path"] == "/models/Qwen2.5-7B-AWQ"
    # speculative_* fields collected
    assert info.speculative is not None
    assert info.speculative["speculative_algorithm"] == "EAGLE"
    assert info.speculative["speculative_num_steps"] == 3


def test_sglang_server_info_embedding_task():
    def fake_get(url, timeout=5):
        if url.endswith("/get_server_info"):
            return _resp(200, {"tp_size": 1})
        if url.endswith("/get_model_info"):
            return _resp(200, {"model_path": "/models/bge", "is_generation": False})
        return _resp(404)

    cfg = BenchmarkConfig(api_url="http://localhost:30000")
    with patch("requests.get", side_effect=fake_get):
        info = SGLangBackend(cfg.api_url).server_info(cfg)

    assert info.task == "embed"
    assert info.tensor_parallel == 1
