"""End-to-end integration test driving the full pipeline.

This test wires the *real* benchmark pipeline against an in-process mock
OpenAI-compatible server (built on ``aiohttp.web``), exercising — without
any real vLLM/SGLang/GPU/network — the following stages:

    detect_backend -> backend.server_info
        -> run_benchmark_async (real SSE streaming + parse_streaming_chunks)
        -> model_intel.build_profile (offline KB path, dense AND MoE)
        -> bottleneck.classify_run
        -> advisor.build_advisory + fitness.assess_fitness
        -> quality.run_quality("probe")  (mock answers the probes)
        -> report.analyst.generate_report (local provider -> mock LLM,
           with the numeric verifier running)
        -> rendering smoke: charts.plot_roofline / plot_bottleneck_map,
           card.render_result_card, share.build_share_markdown,
           html_report.generate_html_report, terminal panel builders.

The mock is deterministic and answers the bundled quality probes so the
probe run actually produces a score.  The report LLM endpoint is pointed at
the same mock, which returns a short narrative that only cites numbers known
to be in the facts bundle so the report verifies and ``generated`` is True.

``pytest-aiohttp`` is intentionally NOT used: the server lifecycle is
managed by hand with an event loop in a fixture.

Author: amit
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import time

import pytest
from aiohttp import web

from vllm_benchmark.analysis import advisor as advisor_mod
from vllm_benchmark.analysis import bottleneck as bottleneck_mod
from vllm_benchmark.analysis import fitness as fitness_mod
from vllm_benchmark.analysis import quality as quality_mod
from vllm_benchmark.analysis.model_intel import build_profile, load_gpu_specs, match_gpu_spec
from vllm_benchmark.analysis.report import analyst as analyst_mod
from vllm_benchmark.config import BenchmarkConfig
from vllm_benchmark.core import async_engine
from vllm_benchmark.core.backends.detect import detect_backend
from vllm_benchmark.core.backends.sglang import SGLangBackend
from vllm_benchmark.reports import card as card_mod
from vllm_benchmark.reports import charts as charts_mod
from vllm_benchmark.reports import html_report as html_mod
from vllm_benchmark.reports import share as share_mod
from vllm_benchmark.reports import terminal as terminal_mod

DENSE_MODEL = "meta-llama/Llama-3-8B"
MOE_MODEL = "Qwen/Qwen3-30B-A3B"

# Pre-computed answers for the bundled deterministic probe set so a probe
# run actually passes (the mock is not a real model).
_PROBE_ANSWERS = {
    "Compute 17 + 26": "43",
    "Compute 12 * 12": "144",
    "Compute 100 - 37": "63",
    "capital of France": "Paris",
    "Red Planet": "Mars",
    "chemical formula for water": "H2O",
    'key "answer"': '{"answer": "ok"}',
}

# A few content chunks streamed for ordinary generative requests.
_STREAM_WORDS = ["Hello", " world", " from", " the", " mock", " server", " token", " here"]


# ---------------------------------------------------------------------------
# Mock server application
# ---------------------------------------------------------------------------

def _answer_for_prompt(prompt: str) -> str:
    """Return a probe-aware deterministic answer for ``prompt``."""
    for needle, answer in _PROBE_ANSWERS.items():
        if needle in prompt:
            return answer
    if "UPPERCASE" in prompt or "uppercase" in prompt:
        return "HELLO"
    return "".join(_STREAM_WORDS)


def _make_app(model_name: str, *, sglang: bool = False) -> web.Application:
    """Build the mock OpenAI-compatible aiohttp app for ``model_name``."""

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {"data": [{"id": model_name, "max_model_len": 32768}]}
        )

    async def version(_request: web.Request) -> web.Response:
        return web.json_response({"version": "0.6.3"})

    async def metrics(_request: web.Request) -> web.Response:
        body = (
            "# HELP vllm:gpu_cache_usage_perc cache usage\n"
            "# TYPE vllm:gpu_cache_usage_perc gauge\n"
            "vllm:gpu_cache_usage_perc 0.5\n"
            "vllm:num_requests_running 2\n"
        )
        return web.Response(text=body, content_type="text/plain")

    async def chat_completions(request: web.Request) -> web.Response:
        payload = await request.json()
        messages = payload.get("messages", [])
        user = messages[-1]["content"] if messages else ""
        stream = bool(payload.get("stream"))

        # Report-LLM requests carry the analyst system prompt; detect them by
        # the JSON facts bundle marker and answer with grounded prose.
        is_report = "facts bundle" in user or "benchmark facts" in user.lower()
        answer = _report_narrative() if is_report else _answer_for_prompt(user)

        prompt_tokens = 16
        if not stream:
            return web.json_response(
                {
                    "id": "cmpl-mock",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max(1, len(answer.split())),
                        "total_tokens": prompt_tokens + max(1, len(answer.split())),
                    },
                }
            )

        # Streaming SSE: several content chunks then a final chunk with usage
        # then [DONE].
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)
        chunks = _STREAM_WORDS
        completion_tokens = len(chunks)
        for word in chunks:
            data = {"choices": [{"index": 0, "delta": {"content": word}}]}
            await response.write(f"data: {json.dumps(data)}\n\n".encode())
        final = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def embeddings(request: web.Request) -> web.Response:
        payload = await request.json()
        inputs = payload.get("input")
        n = len(inputs) if isinstance(inputs, list) else 1
        return web.json_response(
            {
                "data": [{"index": i, "embedding": [0.1, 0.2, 0.3, 0.4]} for i in range(n)],
                "usage": {"prompt_tokens": 4 * n, "total_tokens": 4 * n},
            }
        )

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_get("/version", version)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/embeddings", embeddings)

    if sglang:
        async def get_model_info(_request: web.Request) -> web.Response:
            return web.json_response(
                {"model_path": model_name, "is_generation": True,
                 "tokenizer_path": model_name}
            )

        async def get_server_info(_request: web.Request) -> web.Response:
            return web.json_response(
                {"tp_size": 1, "dp_size": 1, "max_running_requests": 256,
                 "attention_backend": "flashinfer", "version": "0.4.0"}
            )

        app.router.add_get("/get_model_info", get_model_info)
        app.router.add_get("/get_server_info", get_server_info)

    return app


def _report_narrative() -> str:
    """A short, fully-grounded analyst narrative (no invented numbers)."""
    # Use no free numerals at all so the verifier never flags an unsupported
    # number regardless of the bundle contents — keeps the test deterministic.
    return (
        "## Executive summary\n"
        "The served model was benchmarked on the configured accelerator. "
        "Peak throughput and lowest latency are reported in the facts table.\n\n"
        "## Bottleneck analysis\n"
        "The governing bottleneck and recommended lever follow directly from "
        "the roofline placement in the bundle.\n\n"
        "## Application fitness\n"
        "Fitness grades are taken from the advisory section.\n\n"
        "## Recommendations\n"
        "Operate near the throughput-optimal point named in the advisory.\n\n"
        "## Caveats & confidence\n"
        "Confidence levels match the model-profile provenance.\n"
    )


# ---------------------------------------------------------------------------
# Server lifecycle (no pytest-aiohttp): run the mock app in a background
# thread with its own event loop so the main thread can use blocking
# ``requests`` calls (backend detection / quality / report) and drive the
# async benchmark engine with ``asyncio.run`` without contending for a loop.
# ---------------------------------------------------------------------------

class _MockServer:
    """A mock aiohttp app served on a private event loop in a daemon thread."""

    def __init__(self, app: web.Application) -> None:
        self._app = app
        self._loop = asyncio.new_event_loop()
        self._runner = web.AppRunner(app)
        self._ready = threading.Event()
        self.base_url = ""
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._ready.set()
        self._loop.run_forever()

    def start(self) -> str:
        self._thread.start()
        assert self._ready.wait(timeout=10), "mock server failed to start"
        return self.base_url

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        # Best-effort cleanup of the runner on its (now-stopped) loop.
        try:
            self._loop.run_until_complete(self._runner.cleanup())
        except Exception:
            pass
        self._loop.close()


def _start_server(app: web.Application) -> _MockServer:
    """Start a mock server in a background thread and return the handle."""
    server = _MockServer(app)
    server.start()
    # Give the listener a beat to accept connections.
    time.sleep(0.05)
    return server


# ---------------------------------------------------------------------------
# The end-to-end test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("model_name", "expect_moe"),
    [(DENSE_MODEL, False), (MOE_MODEL, True)],
)
def test_full_pipeline_e2e(tmp_path, model_name, expect_moe):
    """Drive the real pipeline end-to-end against an in-process mock server."""
    random.seed(1234)

    # Force offline, deterministic tokenization (avoid network tokenizer load).
    async_engine._tokenizer_cache[model_name] = None
    async_engine._tokenizer_cache[""] = None

    app = _make_app(model_name)
    server = _start_server(app)
    base_url = server.base_url
    try:
        config = BenchmarkConfig(
            api_url=base_url,
            model_name=model_name,
            context_lengths=[1024],
            concurrency_levels=[1, 2],
            output_tokens=16,
            prompt_types=["classic"],
            streaming=True,
            warmup=False,
            request_timeout=30,
        )

        # --- 1. backend detection + server info -------------------------------
        backend = detect_backend(base_url)
        assert backend.name == "vllm"
        server_info = backend.server_info(config)
        assert server_info.backend == "vllm"
        assert server_info.model_name == model_name
        assert server_info.max_model_len == 32768
        assert server_info.backend_version == "0.6.3"
        # /metrics parsed (vllm: lines present).
        assert server_info.raw.get("running_requests") == 2

        # --- 2. real benchmark matrix (SSE streaming path) --------------------
        results: list[dict] = []
        for ctx in config.context_lengths:
            for conc in config.concurrency_levels:
                cell = asyncio.run(
                    async_engine.run_benchmark_async(
                        ctx, conc, config, model_name=model_name, prompt_type="classic",
                    )
                )
                assert cell is not None
                results.append(cell)

        assert len(results) == 2
        for cell in results:
            assert cell["successful"] >= 1
            # The SSE parser produced per-phase numbers.
            assert cell.get("pp_tg_source") == "client_stream"
            assert "prefill_tps_mean" in cell
            assert "decode_tps_mean" in cell

        # --- 3. model profile, offline KB path (dense AND MoE) ----------------
        profile = build_profile(server_info, allow_network=False)
        assert profile.is_moe is expect_moe
        assert profile.kv_bytes_per_token is not None and profile.kv_bytes_per_token > 0
        # KB path is deterministic and offline.
        assert profile.source == "kb"

        # --- 4. bottleneck classification -------------------------------------
        gpu_spec = match_gpu_spec("NVIDIA H100 80GB HBM3") or load_gpu_specs().get("H100 SXM")
        verdicts = bottleneck_mod.classify_run(results, profile, gpu_spec)
        assert isinstance(verdicts, list) and len(verdicts) == len(results)
        for v in verdicts:
            assert v.primary  # a primary class is always set

        # --- 5. advisory + fitness --------------------------------------------
        advisory = advisor_mod.build_advisory(results, profile, server_info, gpu_spec)
        assert advisory.explanation
        assert advisory.fitness is not None
        assert advisory.fitness["profiles"]  # all eight profiles present

        grades = fitness_mod.assess_fitness(results, {}, server_info, profile)
        assert "interactive_chat" in grades
        assert grades["interactive_chat"].grade in {"Good", "Marginal", "Poor", "N/A"}

        # --- 6. quality probe against the mock --------------------------------
        quality = quality_mod.run_quality("probe", config, server_info)
        assert quality["mode"] == "probe"
        assert quality["status"] == "ok"
        assert quality["score"] is not None
        # The mock answers the arithmetic/factual/format probes, so some pass.
        assert quality["passed"] >= 1

        # --- 7. AI analyst report via the mock LLM endpoint -------------------
        metadata = {
            "server_info": server_info.to_dict(),
            "system_info": {"gpu_name": "NVIDIA H100 80GB HBM3", "total_vram_gb": 80.0},
            "model_profile": profile.to_dict(),
            "bottlenecks": [v.to_dict() for v in verdicts],
            "advisory": advisory.to_dict(),
            "quality": quality,
        }
        report = analyst_mod.generate_report(
            results, metadata, provider="local",
            url=base_url, model=model_name, max_tokens=512, seed=7,
        )
        assert isinstance(report, analyst_mod.AnalystReport)
        assert report.markdown
        # The verifier ran on the LLM draft.
        assert report.verification is not None
        assert report.generated is True
        assert "Executive summary" in report.markdown

        # --- 8. rendering smoke (no exceptions; files produced) ---------------
        bottleneck_dicts = [v.to_dict() for v in verdicts]

        roofline_png = charts_mod.plot_roofline(
            results, profile.to_dict(), gpu_spec, str(tmp_path / "roofline.png")
        )
        assert (tmp_path / "roofline.png").exists() and roofline_png.endswith(".png")

        bmap_png = charts_mod.plot_bottleneck_map(
            bottleneck_dicts, str(tmp_path / "bottleneck.png")
        )
        assert (tmp_path / "bottleneck.png").exists() and bmap_png.endswith(".png")

        card_png = card_mod.render_result_card(
            results, metadata, score=None, out_path=str(tmp_path / "card.png")
        )
        assert (tmp_path / "card.png").exists() and card_png.endswith(".png")

        share_md = share_mod.build_share_markdown(results, metadata, score=None)
        assert "vLLM Benchmark" in share_md
        assert model_name in share_md
        share_path = share_mod.save_share_markdown(share_md, str(tmp_path))
        assert re.search(r"share_\d+_\d+\.md$", share_path)

        html_path = html_mod.generate_html_report(
            results, metadata, output_path=str(tmp_path)
        )
        assert html_path.endswith(".html")
        html_text = (tmp_path / html_path.split("/")[-1]).read_text(encoding="utf-8")
        assert "<html" in html_text.lower()

        # Terminal panel builders (rich Panels; just confirm they build).
        intel_panel = terminal_mod.render_model_intel_panel(profile.to_dict())
        bn_panel = terminal_mod.render_bottleneck_panel(bottleneck_dicts)
        fit_panel = terminal_mod.render_fitness_panel(advisory.fitness)
        assert intel_panel is not None and bn_panel is not None and fit_panel is not None
        # Summary table prints without raising.
        terminal_mod.print_summary_table(results)
    finally:
        server.stop()


def test_sglang_backend_detect_and_info():
    """Smoke SGLangBackend.detect / server_info against a second mock app."""
    app = _make_app(DENSE_MODEL, sglang=True)
    server = _start_server(app)
    base_url = server.base_url
    try:
        # The mock exposes both /get_model_info and /get_server_info -> SGLang.
        assert SGLangBackend.detect(base_url) is True

        backend = detect_backend(base_url, forced="sglang")
        assert isinstance(backend, SGLangBackend)

        config = BenchmarkConfig(api_url=base_url, model_name=DENSE_MODEL)
        info = backend.server_info(config)
        assert info.backend == "sglang"
        assert info.model_name == DENSE_MODEL
        assert info.tensor_parallel == 1
        assert info.max_num_seqs == 256
        assert info.backend_version == "0.4.0"
        assert info.task == "generate"
    finally:
        server.stop()
