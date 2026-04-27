"""End-to-end tests for :func:`autoverse.simulator.simulate`.

These exercise the full prediction path (op graph → per-op estimates →
summed latency) for Llama-1B on H100-SXM. Specific latency numbers are
uncalibrated; what we pin here is *structure and sanity*:
  - per-op breakdown has the same length as the op graph,
  - total_ms is the sum of effective_ms,
  - prefill latency scales with seq_len,
  - decode latency scales linearly with ctx_len (dominated by KV-cache reads).
"""

from __future__ import annotations

import math

from autoverse import H100_SXM, LLAMA_1B
from autoverse.cost import estimate
from autoverse.model import build_op_graph
from autoverse.ops import MatMul
from autoverse.simulator import simulate


def test_simulate_empty_graph() -> None:
    result = simulate([], H100_SXM)
    assert result.total_ms == 0.0
    assert result.per_op == []


def test_simulate_single_op_matches_estimate() -> None:
    op = MatMul(m=2048, k=2048, n=2048, name="probe")
    result = simulate([op], H100_SXM)
    timing = estimate(op, H100_SXM)

    assert len(result.per_op) == 1
    name, recorded = result.per_op[0]
    assert name == "probe"
    assert math.isclose(recorded.effective_ms, timing.effective_ms, rel_tol=1e-9)
    assert math.isclose(result.total_ms, timing.effective_ms, rel_tol=1e-9)


def test_simulate_total_is_sum_of_effective() -> None:
    ops = build_op_graph(LLAMA_1B, seq_len=512, mode="prefill")
    result = simulate(ops, H100_SXM)
    assert len(result.per_op) == len(ops)
    expected = sum(t.effective_ms for _, t in result.per_op)
    assert math.isclose(result.total_ms, expected, rel_tol=1e-12)


def test_prefill_latency_grows_with_seq_len() -> None:
    """Attention dominates prefill at long L, so doubling L should roughly quadruple
    the attention contribution. End-to-end latency grows monotonically."""
    short = simulate(build_op_graph(LLAMA_1B, seq_len=512, mode="prefill"), H100_SXM)
    long_ = simulate(build_op_graph(LLAMA_1B, seq_len=2048, mode="prefill"), H100_SXM)
    assert long_.total_ms > short.total_ms


def test_decode_latency_grows_with_ctx_len() -> None:
    """Decode is HBM-bound by KV cache ∝ ctx_len. Total should grow roughly linearly."""
    short = simulate(build_op_graph(LLAMA_1B, seq_len=1024, mode="decode"), H100_SXM)
    long_ = simulate(build_op_graph(LLAMA_1B, seq_len=4096, mode="decode"), H100_SXM)
    assert long_.total_ms > short.total_ms


def test_llama_1b_decode_latency_is_in_plausible_range() -> None:
    """Sanity bound: an uncalibrated roofline for Llama-1B one-token decode at
    ctx_len=1024 on H100 should sit somewhere between 0.1ms and 100ms. Far
    outside that range means we've miscounted bytes or FLOPs by >1000x."""
    result = simulate(build_op_graph(LLAMA_1B, seq_len=1024, mode="decode"), H100_SXM)
    assert 0.1 < result.total_ms < 100.0, f"decode total_ms out of range: {result.total_ms}"
