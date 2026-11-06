"""The HTML report escapes server-derived strings and reports the real version."""

from __future__ import annotations

from vllm_benchmark import __version__
from vllm_benchmark.analysis.diagnostics import Diagnostic
from vllm_benchmark.reports import html_report
from vllm_benchmark.reports.html_report import _script_json, generate_html_report

EVIL = '<script>alert("x")</script>'


def _meta(model_name: str) -> dict:
    return {
        "system_info": {"gpu_name": f"GPU {EVIL}", "total_vram_gb": 80.0},
        "server_info": {"model_name": model_name, "version": "1.0", "quantization": EVIL},
        "model_profile": {"name": model_name, "family": EVIL, "is_moe": False, "source": "kb", "confidence": "inferred"},
        "bottlenecks": [{"cell": [32000, 1, "classic"], "primary": EVIL, "lever": f"Lever {EVIL}",
                         "confidence": "high", "mbu": 0.5, "mfu": 0.2}],
        "advisory": {"fitness": {"verdict": f"Verdict {EVIL}",
                                 "profiles": {EVIL: {"grade": "Good", "limiting_factor": EVIL}}}},
    }


def _results():
    return [{"context_length": 32000, "concurrent_users": 1, "prompt_type": "classic",
             "tokens_per_second": 10.0, "avg_latency": 1.0, "ttft_estimate": 0.1,
             "throughput_per_user": 10.0, "successful": 1, "failed": 0,
             "note": EVIL}]


def test_report_escapes_all_external_strings(tmp_path):
    name = f"org/model-{EVIL}"
    diags = [Diagnostic(severity="warning", title=f"T {EVIL}", message=f"M {EVIL}", metric="x")]
    path = generate_html_report(_results(), _meta(name), None, diags, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    # The raw payload must never appear outside the (escaped) JSON blob.
    assert EVIL not in text
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in text
    # Results JSON is still present and the closing-tag sequence is neutralised.
    assert "<\\/script>" in text
    # A real HTML parser sees exactly the two intended <script> elements
    # (Plotly CDN + inline data), i.e. the payload did not inject a third.
    from html.parser import HTMLParser

    class _Count(HTMLParser):
        n = 0

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.n += 1

    parser = _Count()
    parser.feed(text)
    assert parser.n == 2


def test_script_json_neutralises_closing_tag():
    out = _script_json({"k": "</script><b>"})
    assert "</script>" not in out
    import json
    assert json.loads(out) == {"k": "</script><b>"}


def test_footer_uses_package_version(tmp_path):
    path = generate_html_report(_results(), _meta("m"), None, None, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert f"vLLM Benchmark Suite v{__version__}" in text
    assert "v3.0.0" not in text


def test_esc_handles_none_and_numbers():
    assert html_report._esc(None) == ""
    assert html_report._esc(5) == "5"
    assert html_report._esc("a<b") == "a&lt;b"
