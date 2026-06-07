"""Tests for cross-quant comparison (``analysis/quant_compare.py``).

Synthetic runs exercise: the throughput ranking, per-metric significance,
Holm correction across the comparison family, and honest ties when the
confidence intervals overlap / the difference is not real.

Author: amit
"""

from __future__ import annotations

from vllm_benchmark.analysis.quant_compare import compare_quant_runs


def _run(model: str, quant: str, tps: list[float], mem: list[float],
         quality: float | None = None) -> dict:
    """Build a synthetic run with one cell per provided value."""
    results = []
    for i, (t, m) in enumerate(zip(tps, mem)):
        results.append({
            "context_length": 32000,
            "concurrent_users": 2 ** i,  # distinct cells so they match across runs
            "prompt_type": "classic",
            "tokens_per_second": t,
            "max_mem_used": m,
        })
    meta: dict = {"server_info": {"model_name": model, "quantization": quant}}
    if quality is not None:
        meta["quality"] = {"score": quality}
    return {"metadata": meta, "results": results}


def test_ranking_by_throughput() -> None:
    fast = _run("m", "fp8", [2000, 2100, 2200, 2300], [40000, 40000, 40000, 40000])
    slow = _run("m", "int4", [800, 850, 900, 950], [20000, 20000, 20000, 20000])
    out = compare_quant_runs([slow, fast])
    assert out["ranking"][0] == "m [fp8]"
    assert out["ranking"][1] == "m [int4]"


def test_real_difference_winner_on_throughput() -> None:
    # Large, well-separated throughput samples -> a real difference winner.
    fast = _run("m", "fp8", [2000, 2050, 2100, 2150, 2200],
                [40000, 40100, 40050, 40080, 40020])
    slow = _run("m", "int4", [800, 820, 840, 860, 880],
                [20000, 20100, 20050, 20080, 20020])
    out = compare_quant_runs([slow, fast])
    pair = next(iter(out["metrics"]["tokens_per_second"].values()))
    assert pair["tie"] is False
    assert pair["winner"] == "m [fp8]"
    # peak_mem: lower-is-better, so int4 should win on VRAM.
    mem_pair = next(iter(out["metrics"]["peak_mem"].values()))
    assert mem_pair["winner"] == "m [int4]"


def test_tie_when_cis_overlap() -> None:
    # Nearly identical, overlapping samples -> a tie, never a fabricated winner.
    a = _run("m", "fp8", [1000, 1010, 1005, 1008, 1002], [40000] * 5)
    b = _run("m", "int4", [1001, 1009, 1004, 1007, 1003], [40010] * 5)
    out = compare_quant_runs([a, b])
    pair = next(iter(out["metrics"]["tokens_per_second"].values()))
    assert pair["tie"] is True
    assert pair["winner"] is None


def test_holm_correction_applied() -> None:
    fast = _run("m", "fp8", [2000, 2050, 2100, 2150, 2200], [40000] * 5)
    slow = _run("m", "int4", [800, 820, 840, 860, 880], [20000] * 5)
    out = compare_quant_runs([slow, fast])
    # Every comparison must carry an adjusted_p >= its raw welch_p (Holm
    # never shrinks p-values) and the family size is recorded.
    assert out["n_comparisons"] >= 1
    for metric_pairs in out["metrics"].values():
        for pair in metric_pairs.values():
            assert pair["adjusted_p"] >= pair["welch_p"] - 1e-9
            assert 0.0 <= pair["adjusted_p"] <= 1.0


def test_three_runs_pairwise_family() -> None:
    a = _run("m", "fp8", [2000, 2050, 2100, 2150], [40000] * 4)
    b = _run("m", "int8", [1500, 1520, 1540, 1560], [30000] * 4)
    c = _run("m", "int4", [800, 820, 840, 860], [20000] * 4)
    out = compare_quant_runs([a, b, c])
    # 3 runs -> 3 pairs per metric, 4 metrics -> 12 comparisons.
    assert out["n_comparisons"] == 12
    assert len(out["metrics"]["tokens_per_second"]) == 3
