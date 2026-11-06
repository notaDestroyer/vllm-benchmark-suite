"""Tests for SystemInfo multi-GPU nvidia-smi parsing (GitHub issue #2).

``nvidia-smi --query-gpu=...`` prints one line per GPU.  Earlier versions
called ``float()`` on the whole stdout, which raised on the embedded newline
and silently returned ``None`` on every multi-GPU host.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from vllm_benchmark.core.server import SystemInfo


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*_a, **_k):
        return SimpleNamespace(stdout=stdout, returncode=returncode)
    return _run


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------

def test_parse_gpu_names_single():
    assert SystemInfo.parse_gpu_names(["NVIDIA H100 80GB HBM3"]) == "NVIDIA H100 80GB HBM3"


def test_parse_gpu_names_homogeneous_collapses_with_count():
    assert SystemInfo.parse_gpu_names(["NVIDIA H100 80GB HBM3"] * 4) == "4x NVIDIA H100 80GB HBM3"


def test_parse_gpu_names_heterogeneous_joined():
    out = SystemInfo.parse_gpu_names(["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB", "NVIDIA L4"])
    assert out == "2x NVIDIA A100-SXM4-80GB + NVIDIA L4"


def test_parse_gpu_names_empty():
    assert SystemInfo.parse_gpu_names([]) is None
    assert SystemInfo.parse_gpu_names(["", "  "]) is None


def test_parse_total_vram_mb_sums_all_gpus():
    assert SystemInfo.parse_total_vram_mb(["81559", "81559"]) == 163118.0


def test_parse_total_vram_mb_skips_unparseable():
    assert SystemInfo.parse_total_vram_mb(["81559", "[N/A]"]) == 81559.0
    assert SystemInfo.parse_total_vram_mb(["[N/A]"]) is None
    assert SystemInfo.parse_total_vram_mb([]) is None


def test_parse_total_vram_mb_tolerates_units_suffix():
    # csv without 'nounits' prints e.g. "81559 MiB"
    assert SystemInfo.parse_total_vram_mb(["81559 MiB", "81559 MiB"]) == 163118.0


# ---------------------------------------------------------------------------
# Through subprocess (monkeypatched)
# ---------------------------------------------------------------------------

def test_get_total_vram_multi_gpu_newline_output(monkeypatch):
    """Regression for issue #2: newline-joined nvidia-smi output must not fail."""
    monkeypatch.setattr(subprocess, "run", _fake_run("81559\n81559\n"))
    vram = SystemInfo.get_total_vram()
    assert vram is not None
    assert abs(vram - 163118 / 1024) < 1e-6


def test_get_gpu_name_multi_gpu(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("NVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3\n"))
    assert SystemInfo.get_gpu_name() == "2x NVIDIA H100 80GB HBM3"
    assert SystemInfo.get_gpu_count() == 2


def test_get_driver_version_multi_gpu_returns_single_value(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("550.54.15\n550.54.15\n"))
    assert SystemInfo.get_driver_version() == "550.54.15"


def test_nvidia_smi_failure_returns_none(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=1))
    assert SystemInfo.get_gpu_name() is None
    assert SystemInfo.get_total_vram() is None
    assert SystemInfo.get_gpu_count() == 0


def test_nvidia_smi_missing_returns_none(monkeypatch):
    def _raise(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", _raise)
    info = SystemInfo.get_system_info()
    assert info["gpu_name"] is None
    assert info["total_vram_gb"] is None
    assert info["gpu_count"] == 0
