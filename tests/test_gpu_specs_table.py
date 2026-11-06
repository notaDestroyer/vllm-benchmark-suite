"""The curated GPU table is generated, reproducible and self-consistent.

* The committed JSON equals what ``scripts/gen_gpu_specs.py`` produces.
* Every SKU's real nvidia-smi name resolves to its own key through the
  shared matcher, so adding a card can never silently shadow another.
* Derived entries reproduce the public datasheet peaks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vllm_benchmark.analysis.model_intel import load_gpu_specs, match_gpu_spec
from vllm_benchmark.core.hardware.gpu_names import match_gpu_key

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "src" / "vllm_benchmark" / "data" / "gpu_specs.json"


def _gen():
    """Load the generator script as a module (registered so dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location("gen_gpu_specs", ROOT / "scripts" / "gen_gpu_specs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_gpu_specs"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_committed_table_matches_generator():
    gen = _gen()
    assert SPEC_PATH.read_text() == gen.render(gen.build_table()), "run scripts/gen_gpu_specs.py"


def test_table_covers_ampere_to_blackwell():
    specs = load_gpu_specs()
    archs = {v["architecture"] for v in specs.values()}
    assert {"Ampere (GA100)", "Ampere (GA10x)", "Ada Lovelace", "Hopper", "Blackwell (GB100)", "Blackwell (GB20x)"} <= archs
    assert len(specs) >= 40


def test_every_entry_has_the_roofline_shape():
    for key, v in load_gpu_specs().items():
        assert v["hbm_bandwidth_gbps"] > 0, key
        peaks = v["peak_flops_tflops"]
        assert set(peaks) == {"bf16", "fp8", "int8", "fp4"}, key
        assert peaks["bf16"] > 0, key
        assert v["source"] in ("derived", "datasheet"), key
        assert v["memory_gb"] > 0 and v["sm_count"] > 0, key


def test_every_smi_name_resolves_to_its_own_key():
    specs = load_gpu_specs()
    for key, v in specs.items():
        assert match_gpu_key(v["smi_name"], specs.keys()) == key, f"{v['smi_name']!r} -> {key!r}"
        assert match_gpu_spec(v["smi_name"])["_matched_key"] == key


@pytest.mark.parametrize("name, key", [
    ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
    ("NVIDIA A100 80GB PCIe", "A100 PCIe 80GB"),
    ("NVIDIA H100 80GB HBM3", "H100 SXM"),
    ("NVIDIA H100 NVL", "H100 NVL"),
    ("NVIDIA GeForce RTX 4080", "RTX 4080"),
    ("NVIDIA GeForce RTX 4080 SUPER", "RTX 4080 SUPER"),
    ("NVIDIA GeForce RTX 4070", "RTX 4070"),
    ("NVIDIA RTX 5000 Ada Generation", "RTX 5000 Ada"),
    ("NVIDIA RTX PRO 5000 Blackwell", "RTX PRO 5000 Blackwell"),
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "RTX PRO 6000 Blackwell"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000 Blackwell"),
    ("NVIDIA L40", "L40"),
    ("NVIDIA L40S", "L40S"),
    ("4x NVIDIA H200", "H200"),
])
def test_near_miss_names_resolve_correctly(name, key):
    assert match_gpu_key(name, load_gpu_specs().keys()) == key


@pytest.mark.parametrize("key, bw, bf16, fp8, fp4", [
    ("A100-80GB", 2039, 312, None, None),
    ("A30", 933, 165, None, None),
    ("RTX A6000", 768, 154.8, None, None),
    ("RTX 3090", 936, 71.2, None, None),
    ("L40S", 864, 362.05, 724.1, None),
    ("L40", 864, 181.1, 362.2, None),
    ("L4", 300, 121.2, 242.4, None),
    ("RTX 6000 Ada", 960, 364.2, 728.4, None),
    ("RTX 4090", 1008, 165.2, 330.4, None),
    ("H100 SXM", 3353.6, 989.4, 1978.8, None),
    ("H200", 4800, 989.4, 1978.8, None),
    ("B200", 7997.4, 2250, 4500, 9000),
    ("RTX PRO 6000 Blackwell", 1792, 503.8, 1007.6, 2015.2),
    ("RTX 5090", 1792, 209.5, 419, 838),
    ("RTX 5080", 960, 112.6, 225.2, 450.4),
    ("T4", 320, 65, None, None),
])
def test_entries_match_public_datasheets(key, bw, bf16, fp8, fp4):
    v = load_gpu_specs()[key]
    assert v["hbm_bandwidth_gbps"] == pytest.approx(bw, rel=0.01)
    assert v["peak_flops_tflops"]["bf16"] == pytest.approx(bf16, rel=0.01)
    assert v["peak_flops_tflops"]["fp8"] == (pytest.approx(fp8, rel=0.01) if fp8 else None)
    assert v["peak_flops_tflops"]["fp4"] == (pytest.approx(fp4, rel=0.01) if fp4 else None)


def test_professional_parts_are_not_halved_and_geforce_are():
    specs = load_gpu_specs()
    # Same Ada silicon family: the pro card runs 2x the GeForce per-SM rate.
    expected_ada6000 = 2 * specs["RTX 4090"]["peak_flops_tflops"]["bf16"] * 142 / 128 * 2505 / 2520
    assert specs["RTX 6000 Ada"]["peak_flops_tflops"]["bf16"] == pytest.approx(expected_ada6000, rel=0.01)
    assert specs["RTX PRO 6000 Blackwell"]["halved_fp32_accumulate"] is False
    assert specs["RTX 5090"]["halved_fp32_accumulate"] is True
    assert specs["L40"]["halved_fp32_accumulate"] is True and specs["L40S"]["halved_fp32_accumulate"] is False


def test_json_is_valid_and_readme_present():
    data = json.loads(SPEC_PATH.read_text())
    assert "GENERATED" in data["_README"]
