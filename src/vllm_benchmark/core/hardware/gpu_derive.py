"""Derive roofline GPU specs from the device itself via NVML.

Marketing spec sheets mix dense and sparse numbers, quote FP4 "AI TOPS" for
Blackwell, and silently halve FP16 tensor throughput on GeForce parts when
accumulating in FP32 (which every inference kernel does).  A curated table
therefore drifts and is easy to get wrong.  The card, however, reports
everything needed to compute the peaks exactly:

* **Bandwidth** = ``bus_width_bits / 8 * mem_clock_hz * 2``.  The factor 2
  is the DDR data rate and holds for GDDR6, GDDR6X, GDDR7 and HBM2e/HBM3
  as NVML reports their clocks (H100 SXM: 5120 bit x 2619 MHz x 2 =
  3352 GB/s; RTX 4090: 384 bit x 10501 MHz x 2 = 1008 GB/s).
* **Dense FP16/BF16 tensor peak** = ``SMs * boost_clock_hz * FLOP/clk/SM``
  where the last factor is a constant per compute capability
  (:data:`ARCH_CONSTANTS`), halved on GeForce parts.  FP8 is twice that on
  Ada and newer, FP4 four times on Blackwell.
* **SM count** = CUDA cores / cores-per-SM (also per architecture).

Checks against datasheets: A100 SXM (108 SMs x 1410 MHz x 2048) = 312
TFLOPS; RTX 4090 (128 x 2520 x 512) = 165 TFLOPS; L40S (142 x 2520 x 1024)
= 366 TFLOPS; RTX 5090 (170 x 2407 x 512) = 210 TFLOPS; RTX PRO 6000
Blackwell (188 x 2617 x 1024) = 504 TFLOPS.  Hopper is the one family whose
datasheet uses a lower clock basis than NVML's max boost (H100 SXM: 989
TFLOPS at 1830 MHz vs 1070 at the reported 1980 MHz), which is why
:func:`merge_with_table` prefers a curated entry that agrees with the
derivation to within :data:`AGREEMENT_TOLERANCE`.

Everything here is pure and testable without a GPU except
:func:`read_nvml_devices`, which needs ``nvidia-ml-py`` and an NVIDIA
driver and returns ``[]`` otherwise.

Author: amit
License: MIT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

#: Relative disagreement between a derived peak and the curated table above
#: which the derived value wins (and the discrepancy is reported).
AGREEMENT_TOLERANCE = 0.15


@dataclass(frozen=True)
class ArchConstants:
    """Per-compute-capability constants for the roofline derivation."""

    family: str
    cores_per_sm: int
    #: Dense FP16/BF16 tensor FLOP per SM per clock with FP32 accumulate on
    #: professional / datacenter parts.
    fp16_flop_per_sm_clk: int
    #: GeForce parts run FP16/BF16 (and FP8) tensor math at half rate with
    #: FP32 accumulate.
    geforce_halved: bool
    fp8: bool
    fp4: bool
    #: INT8 relative to dense FP16 (2x on Ampere, equal to FP8 rate later).
    int8_multiplier: float


#: Keyed by compute capability (major, minor).
ARCH_CONSTANTS: dict[tuple[int, int], ArchConstants] = {
    # GA100: A100, A30.  64 FP32 cores/SM, 3rd-gen tensor cores.
    (8, 0): ArchConstants("Ampere (GA100)", 64, 2048, False, False, False, 2.0),
    # GA10x: A40, A10, A16, RTX A6000/A5000/A4000, RTX 30xx.
    (8, 6): ArchConstants("Ampere (GA10x)", 128, 1024, True, False, False, 2.0),
    # Orin (GA10B) behaves like GA10x for this purpose.
    (8, 7): ArchConstants("Ampere (Orin)", 128, 1024, True, False, False, 2.0),
    # AD10x: L40S, L40, L4, RTX 6000 Ada, RTX 40xx.
    (8, 9): ArchConstants("Ada Lovelace", 128, 1024, True, True, False, 2.0),
    # GH100: H100, H200, H20.
    (9, 0): ArchConstants("Hopper", 128, 4096, False, True, False, 2.0),
    # GB100/GB200: B200, GB200, B100.
    (10, 0): ArchConstants("Blackwell (GB100)", 128, 8192, False, True, True, 2.0),
    # GB20x: RTX PRO 6000 Blackwell, RTX 50xx.
    (12, 0): ArchConstants("Blackwell (GB20x)", 128, 1024, True, True, True, 2.0),
}

_GEFORCE_MARKERS = ("geforce", "titan")


@dataclass
class NvmlDevice:
    """Raw per-device readings from NVML (``None`` where unsupported)."""

    index: int
    name: str
    compute_capability: Optional[tuple[int, int]]
    cuda_cores: Optional[int]
    max_sm_clock_mhz: Optional[int]
    max_mem_clock_mhz: Optional[int]
    bus_width_bits: Optional[int]
    memory_total_mb: Optional[float]
    power_limit_w: Optional[float]
    enforced_power_limit_w: Optional[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.compute_capability is not None:
            d["compute_capability"] = list(self.compute_capability)
        return d


def is_geforce(name: Optional[str]) -> bool:
    """True for consumer parts whose FP32-accumulate tensor rate is halved."""
    low = (name or "").lower()
    return any(m in low for m in _GEFORCE_MARKERS)


def arch_for(compute_capability: Optional[tuple[int, int]]) -> Optional[ArchConstants]:
    """Constants for a compute capability, falling back within a major version."""
    if compute_capability is None:
        return None
    cc = (int(compute_capability[0]), int(compute_capability[1]))
    if cc in ARCH_CONSTANTS:
        return ARCH_CONSTANTS[cc]
    # Unknown minor: use the closest known minor of the same major.
    same_major = [k for k in ARCH_CONSTANTS if k[0] == cc[0]]
    if not same_major:
        return None
    # Nearest known minor; ties go to the newer (higher) minor.
    nearest = min(same_major, key=lambda k: (abs(k[1] - cc[1]), -k[1]))
    return ARCH_CONSTANTS[nearest]


def derive_bandwidth_gbps(bus_width_bits: Optional[int], mem_clock_mhz: Optional[int]) -> Optional[float]:
    """DDR bandwidth in GB/s (1 GB = 1e9 B) from bus width and NVML memory clock."""
    if not bus_width_bits or not mem_clock_mhz:
        return None
    return round(bus_width_bits / 8 * mem_clock_mhz * 1e6 * 2 / 1e9, 1)


def derive_peak_flops(
    name: Optional[str],
    compute_capability: Optional[tuple[int, int]],
    cuda_cores: Optional[int],
    max_sm_clock_mhz: Optional[int],
) -> Optional[dict[str, Optional[float]]]:
    """Dense tensor peaks in TFLOPS keyed ``bf16`` / ``fp8`` / ``fp4`` / ``int8``.

    Returns ``None`` when the architecture is unknown or inputs are missing.
    Formats the architecture does not support are ``None``.
    """
    arch = arch_for(compute_capability)
    if arch is None or not cuda_cores or not max_sm_clock_mhz:
        return None
    sms = cuda_cores / arch.cores_per_sm
    per_clk = arch.fp16_flop_per_sm_clk
    if arch.geforce_halved and is_geforce(name):
        per_clk //= 2
    fp16 = sms * max_sm_clock_mhz * 1e6 * per_clk / 1e12
    return {
        "bf16": round(fp16, 1),
        "fp8": round(fp16 * 2, 1) if arch.fp8 else None,
        "fp4": round(fp16 * 4, 1) if arch.fp4 else None,
        "int8": round(fp16 * arch.int8_multiplier, 1),
    }


def derive_spec(device: NvmlDevice) -> Optional[dict[str, Any]]:
    """Build a ``gpu_specs.json``-shaped spec dict from NVML readings.

    Returns ``None`` when neither bandwidth nor FLOPS can be derived.
    """
    bw = derive_bandwidth_gbps(device.bus_width_bits, device.max_mem_clock_mhz)
    peaks = derive_peak_flops(device.name, device.compute_capability, device.cuda_cores, device.max_sm_clock_mhz)
    if bw is None and peaks is None:
        return None
    arch = arch_for(device.compute_capability)
    sms = (device.cuda_cores / arch.cores_per_sm) if (arch and device.cuda_cores) else None
    spec: dict[str, Any] = {
        "hbm_bandwidth_gbps": bw,
        "peak_flops_tflops": peaks or {},
        "_source": "nvml",
        "_matched_key": None,
        "_device": {
            "name": device.name,
            "architecture": arch.family if arch else None,
            "compute_capability": list(device.compute_capability) if device.compute_capability else None,
            "sm_count": int(sms) if sms else None,
            "cuda_cores": device.cuda_cores,
            "max_sm_clock_mhz": device.max_sm_clock_mhz,
            "max_mem_clock_mhz": device.max_mem_clock_mhz,
            "bus_width_bits": device.bus_width_bits,
            "memory_total_mb": device.memory_total_mb,
            "power_limit_w": device.power_limit_w,
            "enforced_power_limit_w": device.enforced_power_limit_w,
            "geforce_halved_rate": bool(arch and arch.geforce_halved and is_geforce(device.name)),
        },
        "_notes": [],
    }
    return spec


def _rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or not b:
        return None
    return abs(a - b) / abs(b)


def merge_with_table(derived: Optional[dict], table: Optional[dict]) -> Optional[dict]:
    """Combine an NVML-derived spec with a curated table entry.

    Rules, per field:

    * Only one side present: use it.
    * Both present and agreeing within :data:`AGREEMENT_TOLERANCE`: keep
      the curated (datasheet) value and mark it NVML-confirmed.  This keeps
      Hopper's datasheet basis.
    * Both present and disagreeing: the derived value wins and the
      discrepancy is recorded in ``_notes`` so the report can say the table
      was overridden (the RTX PRO 6000 Blackwell case).
    * Formats only the derivation knows about (FP4 on Blackwell) are added.
    """
    if derived is None and table is None:
        return None
    if derived is None:
        out = dict(table)
        out.setdefault("_source", "table")
        return out
    if table is None:
        return dict(derived)

    out: dict[str, Any] = dict(table)
    out["_device"] = derived.get("_device")
    notes: list[str] = list(table.get("_notes") or [])
    confirmed: list[str] = []
    overridden: list[str] = []

    d_bw, t_bw = derived.get("hbm_bandwidth_gbps"), table.get("hbm_bandwidth_gbps")
    diff = _rel_diff(d_bw, t_bw)
    if t_bw is None and d_bw is not None:
        out["hbm_bandwidth_gbps"] = d_bw
    elif diff is not None and diff > AGREEMENT_TOLERANCE:
        out["hbm_bandwidth_gbps"] = d_bw
        overridden.append(f"bandwidth table={t_bw} nvml={d_bw}")
    elif diff is not None:
        confirmed.append("bandwidth")

    peaks: dict[str, Optional[float]] = dict(table.get("peak_flops_tflops") or {})
    for fmt, d_val in (derived.get("peak_flops_tflops") or {}).items():
        t_val = peaks.get(fmt)
        if d_val is None:
            continue
        if t_val is None:
            peaks[fmt] = d_val
            continue
        diff = _rel_diff(d_val, t_val)
        if diff is not None and diff > AGREEMENT_TOLERANCE:
            peaks[fmt] = d_val
            overridden.append(f"{fmt} table={t_val} nvml={d_val}")
        else:
            confirmed.append(fmt)
    out["peak_flops_tflops"] = peaks

    if overridden:
        out["_source"] = "nvml (table overridden)"
        notes.append("Curated table disagreed with the device by >15%: " + "; ".join(overridden) + ".")
    else:
        out["_source"] = "table (nvml-confirmed)" if confirmed else "table"
    out["_notes"] = notes
    return out


# ---------------------------------------------------------------------------
# NVML access (the only impure part)
# ---------------------------------------------------------------------------

def _as_str(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def read_nvml_devices() -> list[NvmlDevice]:
    """Read every NVIDIA device via NVML.  Returns ``[]`` without a driver."""
    try:
        import pynvml  # nvidia-ml-py
    except Exception:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    devices: list[NvmlDevice] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            def _get(fn, *args):
                try:
                    return fn(h, *args)
                except Exception:
                    return None

            cc = _get(pynvml.nvmlDeviceGetCudaComputeCapability)
            mem = _get(pynvml.nvmlDeviceGetMemoryInfo)
            plimit = _get(pynvml.nvmlDeviceGetPowerManagementLimit)
            enforced = _get(pynvml.nvmlDeviceGetEnforcedPowerLimit)
            devices.append(NvmlDevice(
                index=i,
                name=_as_str(_get(pynvml.nvmlDeviceGetName) or "unknown"),
                compute_capability=(int(cc[0]), int(cc[1])) if cc else None,
                cuda_cores=_get(pynvml.nvmlDeviceGetNumGpuCores),
                max_sm_clock_mhz=_get(pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_SM),
                max_mem_clock_mhz=_get(pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_MEM),
                bus_width_bits=_get(pynvml.nvmlDeviceGetMemoryBusWidth),
                memory_total_mb=(mem.total / (1024 * 1024)) if mem is not None else None,
                power_limit_w=(plimit / 1000.0) if plimit is not None else None,
                enforced_power_limit_w=(enforced / 1000.0) if enforced is not None else None,
            ))
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return devices


def derive_local_gpu_spec(index: int = 0) -> Optional[dict[str, Any]]:
    """Derive the spec of the local GPU ``index`` from NVML, or ``None``."""
    devices = read_nvml_devices()
    if index >= len(devices):
        return None
    return derive_spec(devices[index])
