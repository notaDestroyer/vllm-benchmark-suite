"""Capacity, roofline ceilings, what-ifs, operating point, power truth.

Fixtures are grounded in public numbers: Llama-3-8B (8.03B params, 32 layers,
8 KV heads x 128 head_dim -> 131072 KV bytes/token in BF16) on an RTX PRO 6000
Blackwell (96 GB, 1792 GB/s, 504 BF16 / 1008 FP8 / 2015 FP4 TFLOPS dense).
"""

from __future__ import annotations

import pytest

from vllm_benchmark.analysis import capacity as cap
from vllm_benchmark.analysis.model_intel import ModelProfile
from vllm_benchmark.core.backends.base import ServerInfo

GB = 1e9
PRO6000 = {
    "hbm_bandwidth_gbps": 1792.0,
    "peak_flops_tflops": {"bf16": 503.8, "fp8": 1007.6, "fp4": 2015.2, "int8": 1007.6},
    "_device": {"memory_total_mb": 97887.0, "max_sm_clock_mhz": 2617, "enforced_power_limit_w": 600.0},
}
A100_NO_FP8 = {
    "hbm_bandwidth_gbps": 2039.0,
    "peak_flops_tflops": {"bf16": 312.0, "fp8": None, "fp4": None, "int8": 624.0},
    "_device": {"memory_total_mb": 81920.0, "max_sm_clock_mhz": 1410},
}


def _llama8b() -> ModelProfile:
    return ModelProfile(name="meta-llama/Llama-3-8B", is_moe=False, total_params=8_030_000_000,
                        active_params=8_030_000_000, num_layers=32, num_kv_heads=8, head_dim=128,
                        kv_bytes_per_token=131072, confidence="confirmed")


def _server(**kw) -> ServerInfo:
    base = dict(backend="vllm", model_name="meta-llama/Llama-3-8B", quantization="FP16/BF16", max_model_len=32768)
    base.update(kw)
    return ServerInfo(**base)


def _cell(ctx, users, decode_per_stream, prefill=None, tps=None, **extra):
    c = {"context_length": ctx, "concurrent_users": users, "prompt_type": "classic",
         "decode_tps_mean": decode_per_stream, "tokens_per_second": tps or decode_per_stream * users}
    if prefill:
        c["prefill_tps_mean"] = prefill
    c.update(extra)
    return c


# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------

def test_footprint_bf16_on_pro6000_estimated():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(), {"gpu_count": 1})
    assert fp.weight_bytes_total == 16_060_000_000
    assert fp.kv_bytes_per_token_total == 131072
    assert fp.gpu_memory_bytes == int(97887.0 * 1024 * 1024)
    assert fp.fits_on_one_gpu is True
    assert fp.kv_capacity_source == "estimated"
    # (0.9 * 102.6 GB - 16.06 GB) * 0.92 / 131072 B ~= 536K tokens
    assert 480_000 < fp.kv_capacity_tokens < 600_000
    assert any("estimated" in n for n in fp.notes)


def test_footprint_prefers_server_measured_capacity():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(), {}, measured_kv_capacity_tokens=441_344)
    assert fp.kv_capacity_tokens == 441_344 and fp.kv_capacity_source == "server"


def test_footprint_tensor_parallel_shards_weights_and_kv():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(tensor_parallel=2), {"gpu_count": 2})
    assert fp.weight_bytes_per_gpu == 16_060_000_000 // 2
    assert fp.kv_bytes_per_token_per_gpu == 131072 // 2
    assert fp.tensor_parallel == 2


def test_footprint_quantized_weights_and_too_big_model():
    fp8 = cap.compute_footprint(_llama8b(), PRO6000, _server(quantization="FP8"), {})
    assert fp8.weight_bytes_total == 8_030_000_000 and fp8.bytes_per_param == 1.0
    big = ModelProfile(name="x/405b", total_params=405e9, active_params=405e9, kv_bytes_per_token=131072, confidence="inferred")
    fp = cap.compute_footprint(big, PRO6000, _server(), {})
    assert fp.fits_on_one_gpu is False and fp.kv_capacity_tokens == 0


def test_footprint_without_memory_or_profile():
    fp = cap.compute_footprint(_llama8b(), {"hbm_bandwidth_gbps": 1.0, "peak_flops_tflops": {}}, _server(), {})
    assert fp.kv_capacity_source == "unavailable" and fp.gpu_memory_bytes is None
    fp2 = cap.compute_footprint(None, PRO6000, _server(), {})
    assert fp2.weight_bytes_total is None and fp2.kv_capacity_source == "unavailable"


def test_footprint_falls_back_to_system_vram_per_gpu():
    spec = {"hbm_bandwidth_gbps": 1792.0, "peak_flops_tflops": {"bf16": 503.8}}
    fp = cap.compute_footprint(_llama8b(), spec, _server(), {"total_vram_gb": 192.0, "gpu_count": 2})
    assert fp.gpu_memory_bytes == int(96.0 * 1024 ** 3)


def test_footprint_warns_when_capacity_below_max_model_len():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(max_model_len=1_000_000), {}, measured_kv_capacity_tokens=500_000)
    assert any("cannot be admitted" in n for n in fp.notes)


# ---------------------------------------------------------------------------
# Matrix fit
# ---------------------------------------------------------------------------

def test_matrix_fit_and_max_concurrency():
    assert cap.max_concurrency_at(441_344, 32_000, 500) == 13
    fit = cap.matrix_fit(441_344, [32_000, 128_000], [1, 4, 8, 16], 500)
    assert fit["known"] and fit["max_users_by_context"] == {32_000: 13, 128_000: 3}
    over = {(c["context_length"], c["concurrent_users"]) for c in fit["cells_over_capacity"]}
    assert over == {(32_000, 16), (128_000, 4), (128_000, 8), (128_000, 16)}
    assert cap.matrix_fit(None, [1], [1], 1)["known"] is False
    assert cap.max_concurrency_at(None, 1, 1) is None


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------

def test_decode_ceiling_formula_and_compute_cap():
    w, kv = 16_060_000_000, 131072
    # N=1 at 32K: 1792e9 / (16.06e9 + 131072*32000) = 1792e9 / 20.25e9 = 88.5 tok/s
    c1, b1 = cap.decode_ceiling_tps(1792.0, w, kv, 32_000, 1)
    assert c1 == pytest.approx(88.5, rel=0.01) and b1 == "bandwidth"
    # N=64: 64*1792e9 / (16.06e9 + 64*131072*32000) = 114.7e12/284.5e9 = 403 tok/s, still bandwidth-bound
    c64, b64 = cap.decode_ceiling_tps(1792.0, w, kv, 32_000, 64, compute_ceiling_tps=31_000)
    assert c64 == pytest.approx(403, rel=0.02) and b64 == "bandwidth"
    # Short context, huge batch: compute cap binds.
    c, b = cap.decode_ceiling_tps(1792.0, w, 0, 1, 10_000, compute_ceiling_tps=31_000)
    assert c == 31_000 and b == "compute"
    # TP=2 doubles aggregate bandwidth.
    c2, _ = cap.decode_ceiling_tps(1792.0, w, kv, 32_000, 1, gpus=2)
    assert c2 == pytest.approx(2 * c1, rel=0.001)
    assert cap.decode_ceiling_tps(None, w, kv, 1, 1) == (None, None)


def test_prefill_ceiling_uses_format_peak():
    # 503.8e12 / (2 * 8.03e9) = 31,370 tok/s
    assert cap.prefill_ceiling_tps(503.8, 8_030_000_000) == pytest.approx(31_370, rel=0.01)
    assert cap.prefill_ceiling_tps(None, 1) is None


def test_cell_ceilings_attainment_and_sources():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(), {})
    cells = [_cell(32_000, 1, 60.0, prefill=15_000.0), _cell(32_000, 4, 50.0), {"context_length": 32_000, "concurrent_users": 8,
                                                                               "tokens_per_second": 300.0}]
    out = cap.cell_ceilings(cells, _llama8b(), PRO6000, fp, output_tokens=0)
    assert len(out) == 3
    c1 = out[0]
    assert c1.decode_ceiling_tps == pytest.approx(88.5, rel=0.01)
    assert c1.decode_measured_tps == 60.0 and c1.decode_measured_source == "decode_tps_mean x users"
    assert c1.decode_attainment == pytest.approx(60 / 88.5, rel=0.01)
    assert c1.prefill_attainment == pytest.approx(15_000 / 31_370, rel=0.01)
    assert out[1].decode_measured_tps == 200.0
    assert out[2].decode_measured_source == "tokens_per_second"
    assert cap.cell_ceilings([], _llama8b(), PRO6000, fp) == []
    assert cap.cell_ceilings(cells, None, PRO6000, fp) == []


def test_fp8_quant_uses_fp8_peak_and_half_weights():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(quantization="FP8"), {})
    out = cap.cell_ceilings([_cell(32_000, 1, 100.0, prefill=20_000.0)], _llama8b(), PRO6000, fp)
    # 1792e9 / (8.03e9 + 4.19e9) = 146.6 tok/s single stream
    assert out[0].decode_ceiling_tps == pytest.approx(146.6, rel=0.01)
    assert out[0].prefill_ceiling_tps == pytest.approx(1007.6e12 / (2 * 8.03e9), rel=0.01)


def test_context_slope_kv_share():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(), {})
    cells = [_cell(4_000, 1, 95.0), _cell(32_000, 1, 70.0), _cell(128_000, 1, 40.0), _cell(32_000, 4, 50.0)]
    rows = cap.context_slope(cap.cell_ceilings(cells, _llama8b(), PRO6000, fp), fp)
    assert [r["context_length"] for r in rows] == [4_000, 32_000, 128_000]
    # KV share grows with context: 0.5GB/16.6GB, 4.2/20.3, 16.8/32.8
    assert rows[0]["kv_share_of_bytes"] == pytest.approx(0.032, abs=0.005)
    assert rows[2]["kv_share_of_bytes"] == pytest.approx(0.51, abs=0.02)
    assert rows[1]["ceiling_tps"] > rows[2]["ceiling_tps"]


# ---------------------------------------------------------------------------
# What-ifs
# ---------------------------------------------------------------------------

def test_quant_what_if_marks_current_and_unsupported():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(), {})
    rows = cap.quant_what_if(_llama8b(), PRO6000, fp, context_length=32_000, output_tokens=500)
    by = {r["format"]: r for r in rows}
    assert by["BF16"]["current"] and not by["FP8"]["current"]
    assert by["NVFP4"]["supported"] and by["NVFP4"]["weight_bytes"] == 4_015_000_000
    assert by["NVFP4"]["single_stream_ceiling_tps"] > by["FP8"]["single_stream_ceiling_tps"] > by["BF16"]["single_stream_ceiling_tps"]
    assert by["INT4 (AWQ/GPTQ)"]["prefill_ceiling_tps"] == by["BF16"]["prefill_ceiling_tps"]  # dequantizes for matmul
    assert by["FP8 KV cache (same weights)"]["kv_capacity_tokens_est"] == fp.kv_capacity_tokens * 2
    # Ampere: no FP8/FP4 tensor cores -> unsupported rows, not dropped.
    rows_a100 = cap.quant_what_if(_llama8b(), A100_NO_FP8, cap.compute_footprint(_llama8b(), A100_NO_FP8, _server(), {}),
                                  context_length=32_000, output_tokens=500)
    a = {r["format"]: r for r in rows_a100}
    assert not a["FP8"]["supported"] and not a["NVFP4"]["supported"] and a["BF16"]["supported"]
    assert cap.quant_what_if(None, PRO6000, fp, context_length=1, output_tokens=1) == []


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------

def test_sweet_spot_detects_knee_and_relates_to_bstar():
    cells = [_cell(32_000, 1, 80.0), _cell(32_000, 4, 70.0), _cell(32_000, 8, 50.0), _cell(32_000, 16, 26.0)]
    # aggregate: 80, 280, 400, 416.  1->4 adds 200 vs 240 linear (83%): keep going;
    # 4->8 adds 120 vs 280 linear (43%, below the 50% rule): the knee is at 4.
    spot = cap.sweet_spot(cells, [{"critical_batch": 280.0}], [])
    assert spot["per_context"][32_000]["sweet_spot_users"] == 4
    assert spot["per_context"][32_000]["throughput_at_sweet_spot"] == 280.0
    assert "well below" in spot["verdict"] and "server" in spot["verdict"]
    still = cap.sweet_spot([_cell(32_000, 1, 80.0), _cell(32_000, 4, 79.0)], None, [])
    assert still["per_context"][32_000]["still_scaling"] and "still scaling" in still["verdict"]
    assert cap.sweet_spot([], None, [])["verdict"] is None


# ---------------------------------------------------------------------------
# Multi-GPU and power
# ---------------------------------------------------------------------------

def test_multi_gpu_prefers_replicas_when_model_fits():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(tensor_parallel=2), {"gpu_count": 2})
    env = {"gpu": {"per_gpu": [{"pcie_gen": 5, "pcie_width": 16}]}}
    adv = cap.multi_gpu_advice(fp, {"gpu_count": 2}, _server(tensor_parallel=2), env)
    assert adv["pcie"] == "PCIe Gen5 x16" and "replicas" in adv["recommendation"] and "TP=2" in adv["recommendation"]
    single = cap.multi_gpu_advice(cap.compute_footprint(_llama8b(), PRO6000, _server(), {"gpu_count": 4}),
                                  {"gpu_count": 4}, _server(), None)
    assert "4 replicas" in single["recommendation"]
    assert cap.multi_gpu_advice(fp, {"gpu_count": 1}, _server(), None)["recommendation"] is None


def test_multi_gpu_requires_tp_when_too_big():
    big = ModelProfile(name="x/405b", total_params=405e9, active_params=405e9, kv_bytes_per_token=131072, confidence="inferred")
    fp = cap.compute_footprint(big, PRO6000, _server(), {"gpu_count": 8})
    adv = cap.multi_gpu_advice(fp, {"gpu_count": 8}, _server(), None)
    assert "TP=" in adv["recommendation"] and "PCIe only" in adv["recommendation"]


def test_power_summary_verdicts():
    capped = [_cell(32_000, 1, 50.0, avg_gpu_clock=1950.0, power_cap_frac=0.9, thermal_frac=0.0, power_limit_w=300.0, avg_power=298.0)]
    p = cap.power_summary(capped, PRO6000)
    assert p["sustained_clock_pct"] == pytest.approx(1950 / 2617, rel=0.01)
    assert "Power-limited" in p["verdict"] and "300 W" in p["verdict"]
    hot = [_cell(32_000, 1, 50.0, avg_gpu_clock=2000.0, power_cap_frac=0.0, thermal_frac=0.5)]
    assert "Thermally" in cap.power_summary(hot, PRO6000)["verdict"]
    fine = [_cell(32_000, 1, 50.0, avg_gpu_clock=2550.0, power_cap_frac=0.0, thermal_frac=0.0)]
    assert "not power- or thermal-limited" in cap.power_summary(fine, PRO6000)["verdict"]
    assert cap.power_summary([], PRO6000)["verdict"] is None


# ---------------------------------------------------------------------------
# Server command
# ---------------------------------------------------------------------------

def test_server_command_vllm_and_sglang():
    fp = cap.compute_footprint(_llama8b(), PRO6000, _server(prefix_caching=False), {}, measured_kv_capacity_tokens=441_344)
    cmd = cap.server_command(fp, _server(prefix_caching=False), context_lengths=[32_000, 128_000], output_tokens=500,
                             throughput_optimal_users=4)
    assert cmd["max_model_len"] == 128_500 and cmd["max_num_seqs"] == 4  # 441344 // 128500 = 3 -> raised to sweet spot 4
    assert "vllm serve meta-llama/Llama-3-8B" in cmd["command"]
    assert "--max-model-len 128500" in cmd["command"] and "--enable-prefix-caching" in cmd["command"]
    fp2 = cap.compute_footprint(_llama8b(), PRO6000, _server(backend="sglang", tensor_parallel=2, kv_cache_dtype="fp8_e5m2"), {},
                                measured_kv_capacity_tokens=1_000_000)
    cmd2 = cap.server_command(fp2, _server(backend="sglang", tensor_parallel=2, kv_cache_dtype="fp8_e5m2"),
                              context_lengths=[32_000], output_tokens=500)
    assert cmd2["command"].startswith("python -m sglang.launch_server")
    assert "--tp 2" in cmd2["command"] and "--kv-cache-dtype fp8_e5m2" in cmd2["command"] and "--max-running-requests 30" in cmd2["command"]


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def test_analyze_capacity_pre_run_and_post_run():
    srv = _server()
    pre = cap.analyze_capacity(None, _llama8b(), PRO6000, srv, {"gpu_count": 1},
                               context_lengths=[32_000, 128_000], concurrency_levels=[1, 4, 16], output_tokens=500)
    assert pre["ceilings"] == [] and pre["power"] is None
    assert pre["footprint"]["kv_capacity_source"] == "estimated"
    assert pre["server_command"]["command"].startswith("vllm serve")
    assert pre["sweet_spot"]["critical_batch"]  # analytic B* even without results
    assert len(pre["method"]["references"]) == 4

    cells = [_cell(32_000, 1, 60.0, prefill=15_000.0, kv_capacity_tokens=441_344),
             _cell(32_000, 4, 50.0), _cell(32_000, 16, 20.0)]
    post = cap.analyze_capacity(cells, _llama8b(), PRO6000, srv, {"gpu_count": 1},
                                context_lengths=[32_000], concurrency_levels=[1, 4, 16], output_tokens=500,
                                bottlenecks=[{"critical_batch": 280.0}])
    assert post["footprint"]["kv_capacity_source"] == "server" and post["footprint"]["kv_capacity_tokens"] == 441_344
    assert len(post["ceilings"]) == 3 and post["best_attainment"]["context_length"] == 32_000
    assert post["context_slope"][0]["measured_tps"] == 60.0
    assert post["matrix_fit"]["cells_over_capacity"] == [{"context_length": 32_000, "concurrent_users": 16, "max_users": 13}]
    import json
    json.dumps(post)  # serialisable for the results JSON


def test_analyze_capacity_reads_sglang_raw_capacity():
    srv = _server(backend="sglang")
    srv.raw["kv_capacity_tokens"] = 250_000
    srv.raw["gpu_memory_utilization"] = 0.85
    out = cap.analyze_capacity(None, _llama8b(), PRO6000, srv, {}, context_lengths=[32_000], concurrency_levels=[1], output_tokens=0)
    assert out["footprint"]["kv_capacity_tokens"] == 250_000 and out["footprint"]["gpu_memory_utilization"] == 0.85


def test_analyze_capacity_never_raises_on_nothing():
    out = cap.analyze_capacity(None, None, None, _server(), None, context_lengths=[], concurrency_levels=[], output_tokens=0)
    assert out["footprint"]["kv_capacity_source"] == "unavailable" and out["what_if"] == []
