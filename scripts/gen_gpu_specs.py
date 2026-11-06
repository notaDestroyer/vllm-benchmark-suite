#!/usr/bin/env python3
"""Generate ``src/vllm_benchmark/data/gpu_specs.json`` from per-SKU facts.

Every roofline number in the table is *derived* from four datasheet facts
per card with the same formulas the suite applies to a live device
(``vllm_benchmark.core.hardware.gpu_derive``):

    bandwidth_GBps  = bus_width_bits / 8 * memory_gbps
    bf16_TFLOPS     = SMs * boost_MHz * FLOP/clk/SM(arch, halved) / 1e6
    fp8 / int8 / fp4 = arch multiples of bf16

so a reviewer can check any entry against the public SM count, boost
clock, bus width and memory data rate instead of trusting a copied
number.  Where NVIDIA's own datasheet peak uses a different clock basis
than the advertised boost (Hopper), or the part is cut in a way the
per-SM rule does not capture (H20, A10G, Turing), ``datasheet_bf16`` pins
the BF16 peak and the multiples follow; ``source`` records which it was.

"halved" marks parts whose FP16/BF16 tensor rate with FP32 accumulate is
half the full professional rate: every GeForce card, and the L40 (unlike
the L40S).  Inference kernels accumulate in FP32, so the halved figure is
the one that is achievable.

Run from the repository root::

    python scripts/gen_gpu_specs.py            # rewrite the JSON
    python scripts/gen_gpu_specs.py --check    # exit 1 if the JSON is stale

Author: amit
License: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_benchmark.core.hardware.gpu_derive import ARCH_CONSTANTS  # noqa: E402

OUT = ROOT / "src" / "vllm_benchmark" / "data" / "gpu_specs.json"


@dataclass(frozen=True)
class Sku:
    key: str                 # table key; tokens must match the nvidia-smi name
    smi_name: str            # a real ``nvidia-smi --query-gpu=name`` string
    cc: tuple[int, int]      # compute capability
    sms: int
    boost_mhz: int
    bus_bits: int
    memory_gbps: float       # per-pin data rate (GDDR: "21 Gbps"; HBM: stack rate)
    memory_gb: float
    halved: bool = False     # GeForce-style half-rate FP16 w/ FP32 accumulate
    datasheet_bf16: Optional[float] = None   # pin BF16 peak (TFLOPS) when the formula does not apply
    datasheet_bw: Optional[float] = None     # pin bandwidth (GB/s) when memory_gbps is not public
    note: str = ""


# fmt: off
SKUS: list[Sku] = [
    # ---------------- Ampere, GA100 (cc 8.0) ----------------
    Sku("A100-80GB", "NVIDIA A100-SXM4-80GB", (8, 0), 108, 1410, 5120, 3.19, 80, note="HBM2e; SXM"),
    Sku("A100 PCIe 80GB", "NVIDIA A100 80GB PCIe", (8, 0), 108, 1410, 5120, 3.02, 80, note="HBM2e; 1935 GB/s"),
    Sku("A100-40GB", "NVIDIA A100-SXM4-40GB", (8, 0), 108, 1410, 5120, 2.43, 40, note="HBM2; 1555 GB/s"),
    Sku("A30", "NVIDIA A30", (8, 0), 56, 1440, 3072, 2.43, 24, note="HBM2; 933 GB/s"),
    # ---------------- Ampere, GA10x (cc 8.6) ----------------
    Sku("A40", "NVIDIA A40", (8, 6), 84, 1740, 384, 14.5, 48),
    Sku("A10", "NVIDIA A10", (8, 6), 72, 1695, 384, 12.5, 24),
    Sku("A10G", "NVIDIA A10G", (8, 6), 80, 1710, 384, 12.5, 24, datasheet_bf16=70.0,
        note="AWS part; NVIDIA lists 70 TFLOPS FP16 tensor dense (half rate)"),
    Sku("RTX A6000", "NVIDIA RTX A6000", (8, 6), 84, 1800, 384, 16.0, 48),
    Sku("RTX A5000", "NVIDIA RTX A5000", (8, 6), 64, 1695, 384, 16.0, 24),
    Sku("RTX A4000", "NVIDIA RTX A4000", (8, 6), 48, 1560, 256, 14.0, 16),
    Sku("RTX 3090 Ti", "NVIDIA GeForce RTX 3090 Ti", (8, 6), 84, 1860, 384, 21.0, 24, halved=True),
    Sku("RTX 3090", "NVIDIA GeForce RTX 3090", (8, 6), 82, 1695, 384, 19.5, 24, halved=True),
    Sku("RTX 3080 Ti", "NVIDIA GeForce RTX 3080 Ti", (8, 6), 80, 1665, 384, 19.0, 12, halved=True),
    Sku("RTX 3080", "NVIDIA GeForce RTX 3080", (8, 6), 68, 1710, 320, 19.0, 10, halved=True),
    Sku("RTX 3070 Ti", "NVIDIA GeForce RTX 3070 Ti", (8, 6), 48, 1770, 256, 19.0, 8, halved=True),
    Sku("RTX 3070", "NVIDIA GeForce RTX 3070", (8, 6), 46, 1725, 256, 14.0, 8, halved=True),
    Sku("RTX 3060 Ti", "NVIDIA GeForce RTX 3060 Ti", (8, 6), 38, 1665, 256, 14.0, 8, halved=True),
    Sku("RTX 3060", "NVIDIA GeForce RTX 3060", (8, 6), 28, 1777, 192, 15.0, 12, halved=True),
    # ---------------- Ada Lovelace (cc 8.9) ----------------
    Sku("L40S", "NVIDIA L40S", (8, 9), 142, 2520, 384, 18.0, 48, datasheet_bf16=362.05,
        note="datasheet peak basis 2490 MHz"),
    Sku("L40", "NVIDIA L40", (8, 9), 142, 2490, 384, 18.0, 48, halved=True,
        note="half-rate FP16 w/ FP32 accumulate (181 TFLOPS), unlike the L40S"),
    Sku("L4", "NVIDIA L4", (8, 9), 58, 2040, 192, 12.5, 24),
    Sku("RTX 6000 Ada", "NVIDIA RTX 6000 Ada Generation", (8, 9), 142, 2505, 384, 20.0, 48),
    Sku("RTX 5000 Ada", "NVIDIA RTX 5000 Ada Generation", (8, 9), 100, 2550, 256, 18.0, 32),
    Sku("RTX 4000 Ada", "NVIDIA RTX 4000 Ada Generation", (8, 9), 48, 2175, 160, 18.0, 20),
    Sku("RTX 4090", "NVIDIA GeForce RTX 4090", (8, 9), 128, 2520, 384, 21.0, 24, halved=True),
    Sku("RTX 4080 SUPER", "NVIDIA GeForce RTX 4080 SUPER", (8, 9), 80, 2550, 256, 23.0, 16, halved=True),
    Sku("RTX 4080", "NVIDIA GeForce RTX 4080", (8, 9), 76, 2505, 256, 22.4, 16, halved=True),
    Sku("RTX 4070 Ti SUPER", "NVIDIA GeForce RTX 4070 Ti SUPER", (8, 9), 66, 2610, 256, 21.0, 16, halved=True),
    Sku("RTX 4070 Ti", "NVIDIA GeForce RTX 4070 Ti", (8, 9), 60, 2610, 192, 21.0, 12, halved=True),
    Sku("RTX 4070 SUPER", "NVIDIA GeForce RTX 4070 SUPER", (8, 9), 56, 2475, 192, 21.0, 12, halved=True),
    Sku("RTX 4070", "NVIDIA GeForce RTX 4070", (8, 9), 46, 2475, 192, 21.0, 12, halved=True),
    Sku("RTX 4060 Ti", "NVIDIA GeForce RTX 4060 Ti", (8, 9), 34, 2535, 128, 18.0, 16, halved=True),
    Sku("RTX 4060", "NVIDIA GeForce RTX 4060", (8, 9), 24, 2460, 128, 17.0, 8, halved=True),
    # ---------------- Hopper (cc 9.0): datasheet basis clocks ----------------
    Sku("H100 SXM", "NVIDIA H100 80GB HBM3", (9, 0), 132, 1980, 5120, 5.24, 80, datasheet_bf16=989.4,
        note="HBM3 3352 GB/s; datasheet peak basis 1830 MHz"),
    Sku("H100 PCIe", "NVIDIA H100 PCIe", (9, 0), 114, 1755, 5120, 3.19, 80, datasheet_bf16=756.4,
        note="HBM2e 2039 GB/s; datasheet peak basis 1620 MHz"),
    Sku("H100 NVL", "NVIDIA H100 NVL", (9, 0), 132, 1785, 6144, 5.12, 94, datasheet_bf16=835.0,
        note="HBM3 3938 GB/s"),
    Sku("H200", "NVIDIA H200", (9, 0), 132, 1980, 6144, 6.25, 141, datasheet_bf16=989.4,
        note="HBM3e 4800 GB/s; same compute as H100 SXM"),
    Sku("H20", "NVIDIA H20", (9, 0), 78, 1980, 6144, 5.21, 96, datasheet_bf16=148.0,
        note="HBM3 4000 GB/s; compute-limited part"),
    # ---------------- Blackwell, GB100 (cc 10.0) ----------------
    Sku("B200", "NVIDIA B200", (10, 0), 148, 1965, 8192, 7.81, 180, datasheet_bf16=2250.0,
        note="HBM3e 8000 GB/s; dense FP16 2.25 PF, FP8 4.5 PF, FP4 9 PF"),
    # ---------------- Blackwell, GB20x (cc 12.0) ----------------
    Sku("RTX PRO 6000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0), 188, 2617, 512, 28.0, 96,
        note="also Max-Q (300 W) and Server Edition; same silicon"),
    Sku("RTX PRO 5000 Blackwell", "NVIDIA RTX PRO 5000 Blackwell", (12, 0), 110, 2617, 384, 28.0, 48),
    Sku("RTX PRO 4500 Blackwell", "NVIDIA RTX PRO 4500 Blackwell", (12, 0), 82, 2617, 256, 28.0, 32),
    Sku("RTX 5090", "NVIDIA GeForce RTX 5090", (12, 0), 170, 2407, 512, 28.0, 32, halved=True),
    Sku("RTX 5080", "NVIDIA GeForce RTX 5080", (12, 0), 84, 2617, 256, 30.0, 16, halved=True),
    Sku("RTX 5070 Ti", "NVIDIA GeForce RTX 5070 Ti", (12, 0), 70, 2452, 256, 28.0, 16, halved=True),
    Sku("RTX 5070", "NVIDIA GeForce RTX 5070", (12, 0), 48, 2512, 192, 28.0, 12, halved=True),
    Sku("RTX 5060 Ti", "NVIDIA GeForce RTX 5060 Ti", (12, 0), 36, 2572, 128, 28.0, 16, halved=True),
    Sku("RTX 5060", "NVIDIA GeForce RTX 5060", (12, 0), 30, 2497, 128, 28.0, 8, halved=True),
    # ---------------- Turing (cc 7.5): no per-arch rule, datasheet only ----------------
    Sku("T4", "Tesla T4", (7, 5), 40, 1590, 256, 10.0, 16, datasheet_bf16=65.0,
        note="FP16 tensor 65 TFLOPS; no BF16/FP8 support, value is FP16"),
]
# fmt: on

README = (
    "GPU hardware specifications used for roofline (MBU/MFU) analysis. Units: hbm_bandwidth_gbps is in GB/s "
    "(1 GB = 1e9 bytes). peak_flops_tflops gives DENSE (not 2:4 sparse) tensor-core throughput in TFLOPS per "
    "numeric format; fp4 is the dense FP4 (NVFP4/MXFP4) peak, Blackwell only; null means the format has no "
    "tensor-core support on that part. Figures are FP16/BF16 with FP32 accumulate, which is what inference "
    "kernels run: professional and datacenter parts run that at full rate, GeForce parts (and the L40) at half "
    "rate, and the table carries the achievable (halved) figure for them. GENERATED by scripts/gen_gpu_specs.py "
    "from per-SKU SM count, boost clock, bus width and memory data rate using the same formulas the suite applies "
    "to a live device (core/hardware/gpu_derive.py); entries whose source is 'datasheet' pin the BF16 peak to "
    "NVIDIA's published number where the formula's clock basis differs (Hopper) or the part is cut (H20, A10G, "
    "T4). Do not edit by hand: edit the SKU list in the script and regenerate. On an NVIDIA host the suite derives "
    "the spec from the device and cross-checks this table (see --gpu-spec)."
)


def build_entry(sku: Sku) -> dict:
    arch = ARCH_CONSTANTS.get(sku.cc)
    bw = sku.datasheet_bw if sku.datasheet_bw is not None else round(sku.bus_bits / 8 * sku.memory_gbps, 1)
    if sku.datasheet_bf16 is not None:
        bf16 = float(sku.datasheet_bf16)
        source = "datasheet"
    else:
        if arch is None:
            raise ValueError(f"{sku.key}: no architecture constants for cc {sku.cc} and no datasheet_bf16")
        per_clk = arch.fp16_flop_per_sm_clk // 2 if sku.halved else arch.fp16_flop_per_sm_clk
        bf16 = round(sku.sms * sku.boost_mhz * 1e6 * per_clk / 1e12, 1)
        source = "derived"
    fp8 = round(bf16 * 2, 1) if (arch and arch.fp8) else None
    fp4 = round(bf16 * 4, 1) if (arch and arch.fp4) else None
    int8 = round(bf16 * (arch.int8_multiplier if arch else 2.0), 1)
    return {
        "hbm_bandwidth_gbps": float(bw),
        "peak_flops_tflops": {"bf16": bf16, "fp8": fp8, "int8": int8, "fp4": fp4},
        "memory_gb": sku.memory_gb,
        "architecture": arch.family if arch else "Turing",
        "compute_capability": list(sku.cc),
        "sm_count": sku.sms,
        "boost_clock_mhz": sku.boost_mhz,
        "bus_width_bits": sku.bus_bits,
        "memory_gbps": sku.memory_gbps,
        "halved_fp32_accumulate": sku.halved,
        "source": source,
        "smi_name": sku.smi_name,
        "note": sku.note,
    }


def build_table() -> dict:
    keys = [s.key for s in SKUS]
    assert len(keys) == len(set(keys)), "duplicate SKU keys"
    table: dict = {"_README": README}
    for sku in SKUS:
        table[sku.key] = build_entry(sku)
    return table


def render(table: dict) -> str:
    return json.dumps(table, indent=2) + "\n"


DOCS = ROOT / "docs" / "gpus.md"


def render_docs(table: dict) -> str:
    """Markdown list of every supported card, grouped by architecture."""
    lines = [
        "# Supported GPUs",
        "",
        "Cards in the built-in roofline table (`src/vllm_benchmark/data/gpu_specs.json`), generated by",
        "`scripts/gen_gpu_specs.py`. On an NVIDIA host the suite derives the spec from the device itself and uses",
        "this table only as a cross-check, so a card missing here still gets a correct roofline when the benchmark",
        "runs on the GPU host (`--gpu-spec auto`). Peaks are dense tensor TFLOPS with FP32 accumulate; `fp4` is",
        "Blackwell-only. `source` is `derived` (from SM count x boost clock x per-SM rate) or `datasheet` (pinned to",
        "NVIDIA's published figure where the formula's clock basis differs or the part is cut).",
        "",
        "To add a card: add one `Sku(...)` line to the script and run `python scripts/gen_gpu_specs.py`.",
        "",
    ]
    by_arch: dict[str, list[tuple[str, dict]]] = {}
    for key, v in table.items():
        if key == "_README":
            continue
        by_arch.setdefault(v["architecture"], []).append((key, v))
    order = ["Blackwell (GB100)", "Blackwell (GB20x)", "Hopper", "Ada Lovelace", "Ampere (GA100)", "Ampere (GA10x)", "Turing"]
    for arch in order + [a for a in by_arch if a not in order]:
        if arch not in by_arch:
            continue
        lines += [f"## {arch}", "",
                  "| Key | nvidia-smi name | Memory | GB/s | BF16 | FP8 | FP4 | Source |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, v in by_arch[arch]:
            pk = v["peak_flops_tflops"]
            fmt = lambda x: "-" if x is None else f"{x:,.0f}"  # noqa: E731
            lines.append(f"| {key} | `{v['smi_name']}` | {v['memory_gb']:g} GB | {v['hbm_bandwidth_gbps']:,.0f} | "
                         f"{fmt(pk['bf16'])} | {fmt(pk['fp8'])} | {fmt(pk['fp4'])} | {v['source']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed JSON differs from the generated one")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    table = build_table()
    text = render(table)
    docs = render_docs(table)
    out = Path(args.out)
    if args.check:
        stale = [str(p) for p, want in ((out, text), (DOCS, docs)) if (p.read_text() if p.exists() else "") != want]
        if stale:
            print(f"stale: {', '.join(stale)}; run scripts/gen_gpu_specs.py", file=sys.stderr)
            return 1
        print(f"{out} and {DOCS} are up to date ({len(SKUS)} SKUs)")
        return 0
    out.write_text(text)
    DOCS.write_text(docs)
    print(f"wrote {out} and {DOCS} ({len(SKUS)} SKUs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
