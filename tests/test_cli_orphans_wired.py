"""Previously orphaned functions are reachable from the CLI."""

from __future__ import annotations

import json

from vllm_benchmark import cli


def test_parser_accepts_new_flags():
    args = cli.build_parser().parse_args(["--leaderboard"])
    assert args.leaderboard == "./outputs"
    args = cli.build_parser().parse_args(["--leaderboard", "/tmp/x", "--detailed-summary"])
    assert args.leaderboard == "/tmp/x" and args.detailed_summary is True
    assert cli.build_parser().parse_args([]).leaderboard is None


def test_run_leaderboard_writes_markdown(tmp_path):
    entry = {"hardware": {"gpu": "H100"}, "software": {"model": "m"},
             "results": {"peak_throughput_tps": 1234.0, "best_ttft_ms": 80.0}, "score": {"grade": "A", "overall": 8000}}
    (tmp_path / "result_a.json").write_text(json.dumps(entry))
    path = cli._run_leaderboard(str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert "| H100 | m | 1,234 |" in text
    assert path.endswith("LEADERBOARD.md")


def test_write_analysis_charts(tmp_path):
    results = [{"context_length": 32000, "concurrent_users": u, "prompt_type": "classic",
                "tokens_per_second": 100.0 * u, "prefill_tps_mean": 5000.0, "decode_tps_mean": 40.0}
               for u in (1, 4)]
    profile = {"active_params": 8e9, "kv_bytes_per_token": 131072, "is_moe": False}
    spec = {"hbm_bandwidth_gbps": 3350, "peak_flops_tflops": {"bf16": 989}}
    verdicts = [{"cell": [32000, u, "classic"], "primary": "decode_weight_bandwidth", "confidence": "high",
                 "critical_batch": 300.0} for u in (1, 4)]
    written = cli._write_analysis_charts(results, profile, spec, verdicts, str(tmp_path), "ts")
    assert len(written) == 2
    assert all(open(p, "rb").read(8).startswith(b"\x89PNG") for p in written)


def test_write_analysis_charts_skips_without_inputs(tmp_path):
    assert cli._write_analysis_charts([], None, None, [], str(tmp_path), "ts") == []
