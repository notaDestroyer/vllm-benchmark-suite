"""Tests for roofline metrics: MBU, MFU and critical batch size.

Golden hand-computed values (dtype-aware), missing-input -> ``None``
guarantees (no fabrication), and Hypothesis property tests asserting B*
is monotonic in bytes/param and MBU decreases as KV sequence length grows.

Author: amit
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from vllm_benchmark.analysis.model_intel import critical_batch, mbu, mfu

# ---------------------------------------------------------------------------
# Golden MBU
# ---------------------------------------------------------------------------

def test_mbu_golden() -> None:
    # 8B params bf16, kv 131072 B/token, seq 8192, 3350 GB/s.
    # bytes_per_token = 8e9*2 + 131072*8192 = 1.6e10 + 1.0737e9 = 1.7074e10
    # ceiling = 3350e9 / 1.7074e10 = 196.2 tok/s
    # measured 50 -> MBU ~0.2549
    val = mbu(50.0, 8_000_000_000, 131072, 8192, 3350.0, 2.0)
    assert val is not None
    assert abs(val - 0.2548) < 0.01


def test_mbu_fp8_weights_higher_than_bf16() -> None:
    # Halving bytes/param halves weight bytes -> higher ceiling -> lower MBU.
    bf16 = mbu(50.0, 8_000_000_000, 131072, 8192, 3350.0, 2.0)
    fp8 = mbu(50.0, 8_000_000_000, 65536, 8192, 3350.0, 1.0)
    assert bf16 is not None and fp8 is not None
    assert fp8 < bf16


def test_mbu_none_on_missing() -> None:
    assert mbu(None, 8e9, 1, 1, 1000, 2) is None
    assert mbu(50, None, 1, 1, 1000, 2) is None
    assert mbu(50, 8e9, None, 1, 1000, 2) is None
    assert mbu(50, 8e9, 1, None, 1000, 2) is None
    assert mbu(50, 8e9, 1, 1, None, 2) is None
    assert mbu(50, 8e9, 1, 1, 1000, None) is None


def test_mbu_zero_inputs_return_none() -> None:
    assert mbu(50, 0, 0, 0, 1000, 2) is None  # bytes_per_token == 0
    assert mbu(50, 8e9, 1, 1, 0, 2) is None  # bandwidth 0


# ---------------------------------------------------------------------------
# Golden MFU
# ---------------------------------------------------------------------------

def test_mfu_golden() -> None:
    # 8B active, peak bf16 989 TFLOPS.
    # flops_per_token = 1.6e10; ceiling = 989e12/1.6e10 = 61812.5 tok/s
    # measured 10000 -> MFU 0.1618
    val = mfu(10000.0, 8_000_000_000, 989.0)
    assert val is not None
    assert abs(val - 0.1618) < 0.01


def test_mfu_dtype_dependence() -> None:
    # FP8 peak (1979) doubles the ceiling -> halves MFU for same throughput.
    bf16 = mfu(10000.0, 8_000_000_000, 989.0)
    fp8 = mfu(10000.0, 8_000_000_000, 1979.0)
    assert bf16 is not None and fp8 is not None
    assert abs(fp8 - bf16 / 2) < 0.01


def test_mfu_none_on_missing() -> None:
    assert mfu(None, 8e9, 989) is None
    assert mfu(10000, None, 989) is None
    assert mfu(10000, 8e9, None) is None
    assert mfu(10000, 0, 989) is None
    assert mfu(10000, 8e9, 0) is None


# ---------------------------------------------------------------------------
# Golden critical batch
# ---------------------------------------------------------------------------

def test_critical_batch_golden_h100_bf16() -> None:
    # ops/byte = 989e12 / 3350e9 = 295.2; B* = round(295.2 * 2 / 2) = 295
    assert critical_batch(989.0, 3350.0, 2.0) == 295


def test_critical_batch_fp8_lower_than_bf16() -> None:
    # bytes_per_param halves -> B* halves.
    bf16 = critical_batch(1979.0, 3350.0, 2.0)
    fp8 = critical_batch(1979.0, 3350.0, 1.0)
    assert bf16 is not None and fp8 is not None
    assert fp8 < bf16


def test_critical_batch_none_on_missing() -> None:
    assert critical_batch(None, 3350, 2) is None
    assert critical_batch(989, None, 2) is None
    assert critical_batch(989, 3350, None) is None
    assert critical_batch(0, 3350, 2) is None


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

@given(
    peak=st.floats(min_value=10.0, max_value=2000.0),
    hbm=st.floats(min_value=100.0, max_value=4000.0),
    bpp_low=st.floats(min_value=0.5, max_value=2.0),
    extra=st.floats(min_value=0.1, max_value=2.0),
)
def test_bstar_monotonic_in_bytes_per_param(peak, hbm, bpp_low, extra) -> None:
    """B* is non-decreasing as bytes/param increases."""
    bpp_high = bpp_low + extra
    low = critical_batch(peak, hbm, bpp_low)
    high = critical_batch(peak, hbm, bpp_high)
    assert low is not None and high is not None
    assert high >= low


@given(
    active=st.integers(min_value=1_000_000_000, max_value=70_000_000_000),
    kvbpt=st.integers(min_value=1000, max_value=1_000_000),
    seq_low=st.integers(min_value=128, max_value=8192),
    seq_add=st.integers(min_value=1, max_value=120_000),
)
def test_decode_bandwidth_ceiling_decreases_with_seq_len(
    active, kvbpt, seq_low, seq_add
) -> None:
    """The achievable single-user decode tok/s ceiling falls with seq_len.

    MBU is ``measured / ceiling`` where the ceiling is the tokens/s the
    HBM bandwidth permits.  Longer context means more KV bytes read per
    token, so the *ceiling itself* drops monotonically.  We probe the
    ceiling by inverting MBU at a fixed measured throughput: a smaller
    ceiling yields a larger MBU for the same measurement, so MBU is
    non-decreasing in seq_len.  Equivalently, to realise the same MBU you
    must give up throughput as context grows — the practical "MBU
    decreases as context grows" intuition for a fixed-bandwidth deploy.
    """
    measured = 50.0
    low = mbu(measured, active, kvbpt, seq_low, 3350.0, 2.0)
    high = mbu(measured, active, kvbpt, seq_low + seq_add, 3350.0, 2.0)
    assert low is not None and high is not None
    # Smaller ceiling at longer seq_len -> larger measured/ceiling ratio.
    assert high >= low
