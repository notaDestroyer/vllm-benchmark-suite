"""Tests for the embeddings workload.

Covers the ``is_embedding_model`` detection table and the batch-size sweep
with a monkeypatched HTTP boundary, asserting docs/sec & latency are
computed and that TTFT / decode fields are deliberately absent.
"""

from __future__ import annotations

import asyncio

import pytest

from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core.backends.base import ServerInfo
from vllm_benchmark.core.workloads import embeddings as emb_mod
from vllm_benchmark.core.workloads.embeddings import (
    EmbeddingsWorkload,
    is_embedding_model,
)

# ---------------------------------------------------------------------------
# is_embedding_model table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BAAI/bge-large-en-v1.5", True),
        ("intfloat/e5-large-v2", True),
        ("thenlper/gte-base", True),
        ("sentence-transformers/all-MiniLM-L6-v2", True),
        ("nomic-ai/nomic-embed-text-v1", True),
        ("BAAI/bge-reranker-large", True),
        ("bert-base-uncased", True),
        ("xlm-roberta-base", True),
        ("jinaai/jina-embeddings-v2", True),
        # Negatives — generative models must not match.
        ("meta-llama/Llama-3-8B-Instruct", False),
        ("mistralai/Mistral-7B-v0.1", False),
        ("Qwen/Qwen2.5-7B", False),
        ("google/gemma-2-9b", False),
        ("", False),
        (None, False),
    ],
)
def test_is_embedding_model_table(name, expected):
    assert is_embedding_model(name) is expected


def test_short_token_word_boundary():
    """Short tokens like 'e5'/'gte' should not match inside unrelated names."""
    assert is_embedding_model("some-e5model-fused") is False
    assert is_embedding_model("agte-7b") is False
    assert is_embedding_model("e5-mistral-7b") is True


# ---------------------------------------------------------------------------
# applicable()
# ---------------------------------------------------------------------------


def test_applicable_by_task():
    wl = EmbeddingsWorkload()
    assert wl.applicable(ServerInfo(backend="vllm", task="embed")) is True
    assert wl.applicable(ServerInfo(backend="vllm", task="rerank")) is True
    assert wl.applicable(ServerInfo(backend="vllm", task="generate")) is False


def test_applicable_by_model_name():
    wl = EmbeddingsWorkload()
    si = ServerInfo(backend="vllm", task="unknown", model_name="BAAI/bge-small-en")
    assert wl.applicable(si) is True
    si2 = ServerInfo(backend="vllm", task="unknown", model_name="meta-llama/Llama-3-8B")
    assert wl.applicable(si2) is False


# ---------------------------------------------------------------------------
# run() / summarize() with mocked HTTP boundary
# ---------------------------------------------------------------------------


def _fake_post_factory():
    """Return a fake ``_post_embeddings`` that fabricates a fast response."""

    async def fake_post(session, endpoint, model_name, inputs, request_timeout):
        # Pretend each request took a fixed, fast time and returned vectors.
        return {
            "success": True,
            "duration": 0.05,
            "num_docs": len(inputs),
            "body": {"data": [{"embedding": [0.0, 1.0]} for _ in inputs]},
        }

    return fake_post


def test_run_computes_docs_and_latency(monkeypatch):
    monkeypatch.setattr(emb_mod, "_post_embeddings", _fake_post_factory())
    # Avoid loading a real tokenizer.
    monkeypatch.setattr(emb_mod, "count_tokens", lambda text, model: max(1, len(text) // 4))

    config = BenchmarkConfig(concurrency_levels=[1, 4, 8])
    wl = EmbeddingsWorkload()
    si = ServerInfo(backend="vllm", task="embed", model_name="BAAI/bge-small-en")

    cells = asyncio.run(wl.run(config, "BAAI/bge-small-en", server_info=si))
    assert len(cells) == 3
    for cell in cells:
        assert cell["docs_per_second"] > 0
        assert cell["tokens_per_second"] > 0
        assert cell["avg_latency"] > 0
        assert "latency_p50" in cell and "latency_p99" in cell
        # Single forward pass — no TTFT/decode concept.
        assert "ttft" not in cell
        assert "ttft_estimate" not in cell
        assert "decode_tps" not in cell
        assert "decode_tps_mean" not in cell

    summary = wl.summarize(cells)
    assert summary["cells"] == 3
    assert summary["peak_docs_per_second"] > 0
    assert summary["best_avg_latency"] > 0


def test_run_skips_failed_requests(monkeypatch):
    async def failing_post(session, endpoint, model_name, inputs, request_timeout):
        return {"success": False, "duration": 0.01, "num_docs": len(inputs), "error": "HTTP 500"}

    monkeypatch.setattr(emb_mod, "_post_embeddings", failing_post)
    monkeypatch.setattr(emb_mod, "count_tokens", lambda text, model: 5)

    config = BenchmarkConfig(concurrency_levels=[1, 4])
    wl = EmbeddingsWorkload()
    cells = asyncio.run(wl.run(config, "BAAI/bge-small-en"))
    assert cells == []
    assert wl.summarize(cells) == {"workload": "embeddings", "cells": 0}


def test_rerank_probe_detected(monkeypatch):
    monkeypatch.setattr(emb_mod, "_post_embeddings", _fake_post_factory())
    monkeypatch.setattr(emb_mod, "count_tokens", lambda text, model: 5)

    config = BenchmarkConfig(concurrency_levels=[1])
    wl = EmbeddingsWorkload()
    si = ServerInfo(backend="vllm", task="rerank", model_name="BAAI/bge-reranker-large")
    cells = asyncio.run(wl.run(config, "BAAI/bge-reranker-large", server_info=si))
    assert any(c.get("rerank") for c in cells)
    summary = wl.summarize(cells)
    assert summary["rerank_probed"] is True
