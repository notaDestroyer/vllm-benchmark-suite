"""Tests for backend detection.

Mocks ``requests.get`` responses to exercise vLLM, SGLang, ambiguous
(unknown), and error/timeout paths without a live server.

Author: amit
"""

from unittest.mock import MagicMock, patch

import requests

from vllm_benchmark.core.backends import (
    OpenAICompatBackend,
    SGLangBackend,
    VLLMBackend,
    detect_backend,
)


def _resp(status: int = 200, json_data=None, text: str = "") -> MagicMock:
    """Build a fake ``requests.Response``."""
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.return_value = json_data if json_data is not None else {}
    return r


# ---------------------------------------------------------------------------
# vLLM detection
# ---------------------------------------------------------------------------


def test_detect_vllm_via_version():
    """A /version endpoint returning a vLLM version detects vLLM."""
    def fake_get(url, timeout=5.0):
        if url.endswith("/version"):
            return _resp(200, {"version": "0.6.3"})
        return _resp(404)

    with patch("requests.get", side_effect=fake_get):
        assert VLLMBackend.detect("http://localhost:8000") is True


def test_detect_vllm_via_metrics():
    """A /metrics body with vllm: lines detects vLLM when /version absent."""
    metrics = "# HELP vllm:num_requests_running\nvllm:num_requests_running 3.0\n"

    def fake_get(url, timeout=5.0):
        if url.endswith("/version"):
            return _resp(404)
        if url.endswith("/metrics"):
            return _resp(200, text=metrics)
        return _resp(404)

    with patch("requests.get", side_effect=fake_get):
        assert VLLMBackend.detect("http://localhost:8000") is True


def test_detect_backend_auto_picks_vllm():
    def fake_get(url, timeout=5.0):
        if url.endswith("/version"):
            return _resp(200, {"version": "0.6.3"})
        return _resp(404)

    with patch("requests.get", side_effect=fake_get):
        backend = detect_backend("http://localhost:8000")
    assert isinstance(backend, VLLMBackend)
    assert backend.name == "vllm"


# ---------------------------------------------------------------------------
# SGLang detection
# ---------------------------------------------------------------------------


def test_detect_sglang_via_endpoints():
    """Both /get_model_info and /get_server_info returning 200 detects SGLang."""
    def fake_get(url, timeout=5.0):
        if url.endswith("/get_model_info"):
            return _resp(200, {"model_path": "/models/x"})
        if url.endswith("/get_server_info"):
            return _resp(200, {"tp_size": 2})
        return _resp(404)

    with patch("requests.get", side_effect=fake_get):
        assert SGLangBackend.detect("http://localhost:30000") is True


def test_detect_sglang_via_metrics():
    metrics = "sglang:num_running_reqs 5\n"

    def fake_get(url, timeout=5.0):
        if url.endswith("/get_model_info") or url.endswith("/get_server_info"):
            return _resp(404)
        if url.endswith("/metrics"):
            return _resp(200, text=metrics)
        return _resp(404)

    with patch("requests.get", side_effect=fake_get):
        assert SGLangBackend.detect("http://localhost:30000") is True


def test_detect_backend_auto_picks_sglang():
    """When vLLM probes fail but SGLang endpoints respond, pick SGLang."""
    def fake_get(url, timeout=5.0):
        if url.endswith("/version"):
            return _resp(404)
        if url.endswith("/metrics"):
            return _resp(404)
        if url.endswith("/get_model_info"):
            return _resp(200, {"model_path": "/models/x"})
        if url.endswith("/get_server_info"):
            return _resp(200, {"tp_size": 2})
        return _resp(404)

    with patch("requests.get", side_effect=fake_get):
        backend = detect_backend("http://localhost:30000")
    assert isinstance(backend, SGLangBackend)
    assert backend.name == "sglang"


# ---------------------------------------------------------------------------
# Ambiguous / unknown
# ---------------------------------------------------------------------------


def test_detect_backend_unknown():
    """A server matching neither backend yields a generic OpenAI-compat one."""
    with patch("requests.get", return_value=_resp(404)):
        backend = detect_backend("http://localhost:9999")
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.name == "unknown"


def test_detect_vllm_returns_false_when_nothing_matches():
    with patch("requests.get", return_value=_resp(404)):
        assert VLLMBackend.detect("http://localhost:9999") is False
        assert SGLangBackend.detect("http://localhost:9999") is False


# ---------------------------------------------------------------------------
# Timeouts / exceptions must not raise
# ---------------------------------------------------------------------------


def test_detect_handles_timeout_without_raising():
    with patch("requests.get", side_effect=requests.exceptions.Timeout):
        assert VLLMBackend.detect("http://localhost:8000") is False
        assert SGLangBackend.detect("http://localhost:8000") is False
        backend = detect_backend("http://localhost:8000")
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.name == "unknown"


def test_detect_handles_connection_error_without_raising():
    with patch("requests.get", side_effect=ConnectionError):
        backend = detect_backend("http://localhost:8000")
    assert backend.name == "unknown"


# ---------------------------------------------------------------------------
# Forced backend
# ---------------------------------------------------------------------------


def test_forced_vllm_skips_probing():
    # No requests.get patch — forced path must not probe the network.
    backend = detect_backend("http://localhost:8000", forced="vllm")
    assert isinstance(backend, VLLMBackend)


def test_forced_sglang_skips_probing():
    backend = detect_backend("http://localhost:8000", forced="sglang")
    assert isinstance(backend, SGLangBackend)
