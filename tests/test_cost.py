"""Tests for :func:`autoverse.cost.estimate`.

The first block pins the Tier-0 pure-roofline semantics (``use_l2=False``):
  - compute_ms = flops / peak_compute_flops_per_s
  - memory_ms  = bytes / peak_bw_bytes_per_s
  - effective_ms = max(compute_ms, memory_ms) + per_op_overhead

The second block pins the Tier-2 L2 hit-rate heuristic.
"""

from __future__ import annotations

import math
from dataclasses import replace

from autoverse import H100_SXM
from autoverse.cost import estimate, l2_hit_rate
from autoverse.ops import AttentionDecode, Embedding, MatMul

# ---------- Tier-0 ablation (use_l2=False) ----------


def test_compute_bound_gemm_uses_flops() -> None:
    # Big square prefill GEMM: heavily compute-bound on H100.
    op = MatMul(m=2048, k=2048, n=2048, dtype="bf16")
    timing = estimate(op, H100_SXM, use_l2=False)

    expected_compute_ms = op.flops() / (H100_SXM.peak_bf16_tflops * 1e12) * 1e3
    assert math.isclose(timing.compute_ms, expected_compute_ms, rel_tol=1e-9)
    assert timing.compute_ms > timing.memory_ms
    assert math.isclose(timing.effective_ms, timing.compute_ms, rel_tol=1e-9)


def test_memory_bound_attention_decode_uses_bytes() -> None:
    # Decode attention at modest context — very HBM-bound.
    op = AttentionDecode(batch=1, ctx_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    timing = estimate(op, H100_SXM, use_l2=False)

    bytes_moved = op.bytes_read() + op.bytes_written()
    expected_memory_ms = bytes_moved / (H100_SXM.hbm_gbps * 1e9) * 1e3
    assert math.isclose(timing.memory_ms, expected_memory_ms, rel_tol=1e-9)
    assert timing.memory_ms > timing.compute_ms
    assert math.isclose(timing.effective_ms, timing.memory_ms, rel_tol=1e-9)


def test_zero_flop_op_is_memory_bound() -> None:
    op = Embedding(n_tokens=1024, d_model=2048)
    timing = estimate(op, H100_SXM, use_l2=False)
    assert timing.compute_ms == 0.0
    assert timing.memory_ms > 0
    assert math.isclose(timing.effective_ms, timing.memory_ms, rel_tol=1e-9)


def test_per_op_overhead_is_additive() -> None:
    # When calibration pushes overhead > 0, it should add to every op.
    spec_with_overhead = replace(H100_SXM, per_op_overhead_us=2.5)
    op = Embedding(n_tokens=16, d_model=2048)

    base = estimate(op, H100_SXM, use_l2=False)
    with_overhead = estimate(op, spec_with_overhead, use_l2=False)

    delta = with_overhead.effective_ms - base.effective_ms
    assert math.isclose(delta, 2.5e-3, rel_tol=1e-9)
    # Compute and memory components are unaffected.
    assert math.isclose(with_overhead.compute_ms, base.compute_ms, rel_tol=1e-9)
    assert math.isclose(with_overhead.memory_ms, base.memory_ms, rel_tol=1e-9)


def test_effective_is_max_of_compute_and_memory() -> None:
    for op in [
        MatMul(m=2048, k=2048, n=2048),
        MatMul(m=1, k=2048, n=2048),  # decode-shape GEMM (memory-bound)
        Embedding(n_tokens=1, d_model=2048),
    ]:
        timing = estimate(op, H100_SXM, use_l2=False)
        assert math.isclose(
            timing.effective_ms,
            max(timing.compute_ms, timing.memory_ms),
            rel_tol=1e-9,
        )


# ---------- Tier-2 L2 heuristic (default behaviour) ----------


def test_l2_hit_rate_one_when_inputs_fit() -> None:
    """Op whose input bytes fit comfortably in 50 MB L2 ⇒ hit_rate=1."""
    # Embedding for 16 tokens × 2048 dims × bf16 = 64 KB read ≪ 50 MB.
    op = Embedding(n_tokens=16, d_model=2048)
    assert l2_hit_rate(op.bytes_read(), H100_SXM.l2_mb) == 1.0


def test_l2_hit_rate_below_one_when_inputs_exceed_l2() -> None:
    # 4096^3 bf16 matmul: inputs (M·K + K·N)·2 ≈ 67 MB > 50 MB L2 ⇒ hit_rate < 1.
    op = MatMul(m=4096, k=4096, n=4096)
    hr = l2_hit_rate(op.bytes_read(), H100_SXM.l2_mb)
    assert 0.0 < hr < 1.0
    assert math.isclose(hr, (H100_SXM.l2_mb * 1024 * 1024) / op.bytes_read(), rel_tol=1e-12)


def test_l2_disabled_returns_zero_hit_rate() -> None:
    # use_l2=False (or l2_mb=0) ⇒ heuristic ablated, full bytes count.
    assert l2_hit_rate(123, l2_mb=0) == 0.0


def test_l2_full_hit_keeps_output_writes_in_memory_term() -> None:
    """Inputs fully cached, but output writes always stream to HBM."""
    op = Embedding(n_tokens=16, d_model=2048)  # input 64 KB ≪ L2; output 64 KB
    t = estimate(op, H100_SXM)
    assert t.l2_hit_rate == 1.0
    # Output bytes still drive a non-zero memory term.
    expected_memory_ms = op.bytes_written() / (H100_SXM.hbm_gbps * 1e9) * 1e3
    assert math.isclose(t.memory_ms, expected_memory_ms, rel_tol=1e-9)


def test_l2_partial_hit_only_scales_input_bytes() -> None:
    """Inputs partially cached ⇒ effective_bytes = output + (1-hit)·input."""
    op = MatMul(m=4096, k=4096, n=4096)
    bytes_read = op.bytes_read()
    bytes_written = op.bytes_written()

    cold = estimate(op, H100_SXM, use_l2=False)
    warm = estimate(op, H100_SXM, use_l2=True)
    # Re-derive the expected memory from the formula.
    expected_eff_bytes = bytes_written + bytes_read * (1.0 - warm.l2_hit_rate)
    expected_memory_ms = expected_eff_bytes / (H100_SXM.hbm_gbps * 1e9) * 1e3
    assert math.isclose(warm.memory_ms, expected_memory_ms, rel_tol=1e-9)
    # Cold uses full bytes_read + bytes_written.
    cold_eff = bytes_read + bytes_written
    cold_expected_ms = cold_eff / (H100_SXM.hbm_gbps * 1e9) * 1e3
    assert math.isclose(cold.memory_ms, cold_expected_ms, rel_tol=1e-9)
    # Compute term is unaffected.
    assert math.isclose(warm.compute_ms, cold.compute_ms, rel_tol=1e-9)
