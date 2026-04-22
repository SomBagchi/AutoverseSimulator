"""Tests for :func:`autoverse.cost.estimate`.

Pins the Tier-0 pure-roofline semantics:
  - compute_ms = flops / peak_compute_flops_per_s
  - memory_ms  = bytes / peak_bw_bytes_per_s
  - effective_ms = max(compute_ms, memory_ms) + per_op_overhead
"""

from __future__ import annotations

import math
from dataclasses import replace

from autoverse import H100_SXM
from autoverse.cost import estimate
from autoverse.ops import AttentionDecode, Embedding, MatMul


def test_compute_bound_gemm_uses_flops() -> None:
    # Big square prefill GEMM: heavily compute-bound on H100.
    op = MatMul(m=2048, k=2048, n=2048, dtype="bf16")
    timing = estimate(op, H100_SXM)

    expected_compute_ms = op.flops() / (H100_SXM.peak_bf16_tflops * 1e12) * 1e3
    assert math.isclose(timing.compute_ms, expected_compute_ms, rel_tol=1e-9)
    assert timing.compute_ms > timing.memory_ms
    assert math.isclose(timing.effective_ms, timing.compute_ms, rel_tol=1e-9)


def test_memory_bound_attention_decode_uses_bytes() -> None:
    # Decode attention at modest context — very HBM-bound.
    op = AttentionDecode(batch=1, ctx_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    timing = estimate(op, H100_SXM)

    bytes_moved = op.bytes_read() + op.bytes_written()
    expected_memory_ms = bytes_moved / (H100_SXM.hbm_gbps * 1e9) * 1e3
    assert math.isclose(timing.memory_ms, expected_memory_ms, rel_tol=1e-9)
    assert timing.memory_ms > timing.compute_ms
    assert math.isclose(timing.effective_ms, timing.memory_ms, rel_tol=1e-9)


def test_zero_flop_op_is_memory_bound() -> None:
    op = Embedding(n_tokens=1024, d_model=2048)
    timing = estimate(op, H100_SXM)
    assert timing.compute_ms == 0.0
    assert timing.memory_ms > 0
    assert math.isclose(timing.effective_ms, timing.memory_ms, rel_tol=1e-9)


def test_per_op_overhead_is_additive() -> None:
    # When calibration pushes overhead > 0, it should add to every op.
    spec_with_overhead = replace(H100_SXM, per_op_overhead_us=2.5)
    op = Embedding(n_tokens=16, d_model=2048)

    base = estimate(op, H100_SXM)
    with_overhead = estimate(op, spec_with_overhead)

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
        timing = estimate(op, H100_SXM)
        assert math.isclose(
            timing.effective_ms,
            max(timing.compute_ms, timing.memory_ms),
            rel_tol=1e-9,
        )
