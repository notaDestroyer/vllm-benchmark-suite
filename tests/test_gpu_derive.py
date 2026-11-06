"""NVML-derived GPU specs: bandwidth and dense tensor peaks from device facts."""

from __future__ import annotations

import sys
import types

import pytest

from vllm_benchmark.analysis import model_intel as mi
from vllm_benchmark.analysis.bottleneck import peak_dtype_for
from vllm_benchmark.core.backends.openai_compat import infer_quantization
from vllm_benchmark.core.hardware import gpu_derive as gd
from vllm_benchmark.core.hardware.gpu_derive import (
    NvmlDevice,
    arch_for,
    derive_bandwidth_gbps,
    derive_peak_flops,
    derive_spec,
    is_geforce,
    merge_with_table,
    read_nvml_devices,
)


def _dev(name, cc, cores, sm_mhz, mem_mhz, bus, **kw) -> NvmlDevice:
    base = dict(index=0, name=name, compute_capability=cc, cuda_cores=cores, max_sm_clock_mhz=sm_mhz,
                max_mem_clock_mhz=mem_mhz, bus_width_bits=bus, memory_total_mb=None,
                power_limit_w=None, enforced_power_limit_w=None)
    base.update(kw)
    return NvmlDevice(**base)


# Datasheet cross-checks (dense, FP32-accumulate figures that inference kernels achieve).
CARDS = {
    "A100 SXM 80GB": (_dev("NVIDIA A100-SXM4-80GB", (8, 0), 6912, 1410, 1593, 5120), 2039, 312, None, None),
    "RTX 4090": (_dev("NVIDIA GeForce RTX 4090", (8, 9), 16384, 2520, 10501, 384), 1008, 165, 330, None),
    "L40S": (_dev("NVIDIA L40S", (8, 9), 18176, 2520, 9001, 384), 864, 366, 733, None),
    "RTX 5090": (_dev("NVIDIA GeForce RTX 5090", (12, 0), 21760, 2407, 14001, 512), 1792, 210, 419, 838),
    "RTX PRO 6000 Blackwell": (_dev("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0), 24064, 2617, 14001, 512),
                               1792, 504, 1008, 2015),
    "H100 SXM": (_dev("NVIDIA H100 80GB HBM3", (9, 0), 16896, 1980, 2619, 5120), 3352, 1070, 2140, None),
}


@pytest.mark.parametrize("label", list(CARDS))
def test_derivation_matches_known_cards(label):
    dev, bw, bf16, fp8, fp4 = CARDS[label]
    assert derive_bandwidth_gbps(dev.bus_width_bits, dev.max_mem_clock_mhz) == pytest.approx(bw, rel=0.01)
    peaks = derive_peak_flops(dev.name, dev.compute_capability, dev.cuda_cores, dev.max_sm_clock_mhz)
    assert peaks["bf16"] == pytest.approx(bf16, rel=0.01)
    if fp8 is None:
        assert peaks["fp8"] is None
    else:
        assert peaks["fp8"] == pytest.approx(fp8, rel=0.01)
    if fp4 is None:
        assert peaks["fp4"] is None
    else:
        assert peaks["fp4"] == pytest.approx(fp4, rel=0.01)


def test_geforce_halving_only_on_consumer_parts():
    pro = derive_peak_flops("NVIDIA RTX PRO 6000 Blackwell", (12, 0), 24064, 2617)
    con = derive_peak_flops("NVIDIA GeForce RTX PRO 6000", (12, 0), 24064, 2617)
    assert con["bf16"] == pytest.approx(pro["bf16"] / 2, rel=0.001)
    assert is_geforce("NVIDIA GeForce RTX 3090") and is_geforce("NVIDIA TITAN RTX")
    assert not is_geforce("NVIDIA A100-SXM4-80GB") and not is_geforce(None)
    # Datacenter Ampere is never halved even if someone names it oddly.
    assert derive_peak_flops("GeForce A100", (8, 0), 6912, 1410)["bf16"] == pytest.approx(312, rel=0.01)


def test_int8_rates():
    assert derive_peak_flops("A100", (8, 0), 6912, 1410)["int8"] == pytest.approx(624, rel=0.01)
    assert derive_peak_flops("L40S", (8, 9), 18176, 2520)["int8"] == pytest.approx(733, rel=0.01)


def test_arch_for_fallbacks():
    assert arch_for((8, 6)).family.startswith("Ampere")
    assert arch_for((8, 8)).family == arch_for((8, 9)).family  # nearest known minor
    assert arch_for((12, 1)).family == arch_for((12, 0)).family
    assert arch_for((7, 5)) is None  # Turing unsupported
    assert arch_for(None) is None


def test_missing_inputs_return_none():
    assert derive_bandwidth_gbps(None, 1000) is None
    assert derive_bandwidth_gbps(384, 0) is None
    assert derive_peak_flops("x", (8, 9), None, 2520) is None
    assert derive_peak_flops("x", (7, 5), 4608, 1545) is None
    assert derive_spec(_dev("x", None, None, None, None, None)) is None


def test_derive_spec_shape():
    dev, *_ = CARDS["RTX PRO 6000 Blackwell"]
    dev = _dev(dev.name, dev.compute_capability, dev.cuda_cores, dev.max_sm_clock_mhz, dev.max_mem_clock_mhz,
               dev.bus_width_bits, memory_total_mb=97887.0, power_limit_w=600.0, enforced_power_limit_w=300.0)
    spec = derive_spec(dev)
    assert spec["_source"] == "nvml"
    assert spec["hbm_bandwidth_gbps"] == pytest.approx(1792, rel=0.01)
    assert spec["peak_flops_tflops"]["fp4"] == pytest.approx(2015, rel=0.01)
    d = spec["_device"]
    assert d["sm_count"] == 188 and d["architecture"] == "Blackwell (GB20x)"
    assert d["enforced_power_limit_w"] == 300.0 and d["geforce_halved_rate"] is False
    assert spec["_device"]["compute_capability"] == [12, 0]


# ---------------------------------------------------------------------------
# Merge with the curated table
# ---------------------------------------------------------------------------

def test_merge_table_wins_when_agreeing():
    derived = {"hbm_bandwidth_gbps": 3352.0, "peak_flops_tflops": {"bf16": 1070.0, "fp8": 2140.0, "fp4": None, "int8": 2140.0},
               "_source": "nvml", "_device": {"sm_count": 132}}
    table = {"hbm_bandwidth_gbps": 3350.0, "peak_flops_tflops": {"bf16": 989.0, "fp8": 1979.0, "int8": 1979.0}, "_matched_key": "H100 SXM"}
    out = merge_with_table(derived, table)
    assert out["peak_flops_tflops"]["bf16"] == 989.0  # datasheet basis kept (8% apart)
    assert out["_source"] == "table (nvml-confirmed)"
    assert out["_device"]["sm_count"] == 132 and out["_notes"] == []


def test_merge_nvml_wins_when_table_is_off_and_adds_fp4():
    derived = {"hbm_bandwidth_gbps": 1792.0, "peak_flops_tflops": {"bf16": 503.8, "fp8": 1007.6, "fp4": 2015.2, "int8": 1007.6},
               "_source": "nvml", "_device": {}}
    table = {"hbm_bandwidth_gbps": 1792.0, "peak_flops_tflops": {"bf16": 251.0, "fp8": 503.0, "int8": 503.0},
             "_matched_key": "RTX PRO 6000 Blackwell"}
    out = merge_with_table(derived, table)
    assert out["peak_flops_tflops"]["bf16"] == 503.8
    assert out["peak_flops_tflops"]["fp4"] == 2015.2
    assert out["_source"] == "nvml (table overridden)"
    assert any("bf16 table=251.0 nvml=503.8" in n for n in out["_notes"])
    assert out["_matched_key"] == "RTX PRO 6000 Blackwell"


def test_merge_one_side_missing():
    assert merge_with_table(None, None) is None
    assert merge_with_table(None, {"hbm_bandwidth_gbps": 1.0, "peak_flops_tflops": {}})["_source"] == "table"
    assert merge_with_table({"hbm_bandwidth_gbps": 1.0, "peak_flops_tflops": {}, "_source": "nvml"}, None)["_source"] == "nvml"


# ---------------------------------------------------------------------------
# NVML reader with a fake pynvml
# ---------------------------------------------------------------------------

def _fake_pynvml(devices: list[dict], fail_init: bool = False):
    m = types.ModuleType("pynvml")
    m.NVML_CLOCK_SM, m.NVML_CLOCK_MEM = 1, 2
    calls = {"shutdown": 0}

    def _init():
        if fail_init:
            raise RuntimeError("no driver")

    m.nvmlInit = _init
    m.nvmlShutdown = lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1)
    m.nvmlDeviceGetCount = lambda: len(devices)
    m.nvmlDeviceGetHandleByIndex = lambda i: devices[i]
    m.nvmlDeviceGetName = lambda h: h["name"]
    m.nvmlDeviceGetCudaComputeCapability = lambda h: h["cc"]
    m.nvmlDeviceGetNumGpuCores = lambda h: h["cores"]
    m.nvmlDeviceGetMaxClockInfo = lambda h, kind: h["sm"] if kind == 1 else h["mem"]
    m.nvmlDeviceGetMemoryBusWidth = lambda h: h["bus"]
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=h["mem_bytes"])
    m.nvmlDeviceGetPowerManagementLimit = lambda h: h["plimit"]

    def enforced(h):
        raise RuntimeError("unsupported")

    m.nvmlDeviceGetEnforcedPowerLimit = enforced
    return m, calls


def test_read_nvml_devices_with_fake_driver(monkeypatch):
    fake, calls = _fake_pynvml([
        {"name": b"NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "cc": (12, 0), "cores": 24064, "sm": 2617,
         "mem": 14001, "bus": 512, "mem_bytes": 97887 * 1024 * 1024, "plimit": 600000},
    ])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    devs = read_nvml_devices()
    assert len(devs) == 1
    d = devs[0]
    assert d.name.startswith("NVIDIA RTX PRO 6000")  # bytes decoded
    assert d.cuda_cores == 24064 and d.bus_width_bits == 512
    assert d.power_limit_w == 600.0 and d.enforced_power_limit_w is None  # unsupported call -> None
    assert calls["shutdown"] == 1
    spec = gd.derive_local_gpu_spec(0)
    assert spec["peak_flops_tflops"]["bf16"] == pytest.approx(503.8, rel=0.01)


def test_read_nvml_devices_without_driver(monkeypatch):
    fake, _ = _fake_pynvml([], fail_init=True)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert read_nvml_devices() == []
    assert gd.derive_local_gpu_spec(0) is None


def test_read_nvml_devices_without_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert read_nvml_devices() == []


# ---------------------------------------------------------------------------
# resolve_gpu_spec modes
# ---------------------------------------------------------------------------

def test_resolve_gpu_spec_modes(monkeypatch):
    derived = {"hbm_bandwidth_gbps": 1792.0, "peak_flops_tflops": {"bf16": 503.8, "fp8": 1007.6, "fp4": 2015.2, "int8": 1007.6},
               "_source": "nvml", "_matched_key": None, "_device": {"sm_count": 188}, "_notes": []}
    monkeypatch.setattr("vllm_benchmark.core.hardware.gpu_derive.derive_local_gpu_spec", lambda i=0: dict(derived))
    name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    t = mi.resolve_gpu_spec(name, "table")
    assert t["_source"] == "table" and t["_matched_key"] == "RTX PRO 6000 Blackwell"
    n = mi.resolve_gpu_spec(name, "nvml")
    assert n["_source"] == "nvml"
    a = mi.resolve_gpu_spec(name, "auto")
    assert a["peak_flops_tflops"]["fp4"] == 2015.2 and a["_matched_key"] == "RTX PRO 6000 Blackwell"
    # Unknown card + no driver -> None; unknown card + driver -> derived.
    monkeypatch.setattr("vllm_benchmark.core.hardware.gpu_derive.derive_local_gpu_spec", lambda i=0: None)
    assert mi.resolve_gpu_spec("Some Future GPU", "auto") is None


def test_curated_table_pro6000_now_full_rate_with_fp4():
    spec = mi.match_gpu_spec("NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
    assert spec["peak_flops_tflops"]["bf16"] == pytest.approx(503.8)
    assert spec["peak_flops_tflops"]["fp4"] == pytest.approx(2015.2)
    assert mi.match_gpu_spec("NVIDIA GeForce RTX 5090")["peak_flops_tflops"]["fp4"] == pytest.approx(838.0)


# ---------------------------------------------------------------------------
# FP4 plumbing
# ---------------------------------------------------------------------------

def test_fp4_quant_detection_and_bytes():
    assert infer_quantization("nvidia/Llama-3.3-70B-Instruct-NVFP4") == "NVFP4"
    assert infer_quantization("openai/gpt-oss-120b-mxfp4") == "MXFP4"
    assert infer_quantization("x-FP4") == "FP4"
    assert infer_quantization("x-FP8") == "FP8"
    assert mi.bytes_per_param("NVFP4") == 0.5 and mi.bytes_per_param("mxfp4") == 0.5
    assert peak_dtype_for("NVFP4") == "fp4" and peak_dtype_for("fp8_e4m3") == "fp8"
    assert peak_dtype_for("AWQ") == "bf16" and peak_dtype_for(None) == "bf16"


def test_cli_gpu_spec_flag():
    from vllm_benchmark.cli import build_parser
    assert build_parser().parse_args([]).gpu_spec == "auto"
    assert build_parser().parse_args(["--gpu-spec", "nvml"]).gpu_spec == "nvml"
