"""CPU smoke tests for :mod:`autoverse.measure`.

These tests don't assert accuracy — just structure and that the harness runs
end-to-end on whatever device is available (CPU/MPS on dev, CUDA in prod).
Real calibration numbers come from running on an H100 via
``scripts/collect_measurements.py``.

Torch is an optional dep; the whole module skips when torch is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autoverse.measure import (  # noqa: E402
    MeasuredTiming,
    measure_attention_decode,
    measure_attention_prefill,
    measure_embedding,
    measure_graph,
    measure_matmul,
    measure_op,
    measure_residual,
    measure_rmsnorm,
    measure_rope,
    measure_silu_gate,
    time_callable,
)
from autoverse.model import LLAMA_1B, build_op_graph  # noqa: E402
from autoverse.ops import MatMul, Op, RMSNorm  # noqa: E402

# Smoke-test iteration counts: tiny so CPU runs stay under a second.
N_WARMUP = 2
N_ITERS = 5
# Tiny shapes — we're testing plumbing, not throughput.
DEVICE = "cpu"


def _assert_valid_timing(t: MeasuredTiming, n_iters: int) -> None:
    assert isinstance(t, MeasuredTiming)
    assert t.n_iters == n_iters
    assert t.median_ms >= 0.0
    assert t.mean_ms >= 0.0
    assert t.std_ms >= 0.0
    assert t.p10_ms <= t.median_ms <= t.p90_ms


def test_time_callable_returns_valid_structure() -> None:
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1

    t = time_callable(fn, device=DEVICE, n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)
    assert calls == N_WARMUP + N_ITERS


def test_measure_matmul_smoke() -> None:
    t = measure_matmul(m=4, k=8, n=16, dtype="fp32", device=DEVICE,
                       n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_attention_prefill_smoke() -> None:
    t = measure_attention_prefill(batch=1, seq_len=16, n_heads=4, n_kv_heads=2, d_head=8,
                                   dtype="fp32", device=DEVICE,
                                   n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_attention_decode_smoke() -> None:
    t = measure_attention_decode(batch=1, ctx_len=32, n_heads=4, n_kv_heads=2, d_head=8,
                                  dtype="fp32", device=DEVICE,
                                  n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_rmsnorm_smoke() -> None:
    t = measure_rmsnorm(n_tokens=4, d_model=32, dtype="fp32", device=DEVICE,
                        n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_silu_gate_smoke() -> None:
    t = measure_silu_gate(n_tokens=4, d_ffn=32, dtype="fp32", device=DEVICE,
                          n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_residual_smoke() -> None:
    t = measure_residual(n_tokens=4, d_model=32, dtype="fp32", device=DEVICE,
                         n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_embedding_smoke() -> None:
    t = measure_embedding(n_tokens=4, d_model=32, vocab_size=128, dtype="fp32", device=DEVICE,
                          n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_rope_smoke() -> None:
    t = measure_rope(n_tokens=4, n_heads=4, n_kv_heads=2, d_head=8, dtype="fp32", device=DEVICE,
                     n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_op_dispatches_matmul() -> None:
    op = MatMul(m=4, k=8, n=16, dtype="fp32")
    t = measure_op(op, device=DEVICE, n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_op_dispatches_rmsnorm() -> None:
    op = RMSNorm(n_tokens=4, d_model=32, dtype="fp32")
    t = measure_op(op, device=DEVICE, n_warmup=N_WARMUP, n_iters=N_ITERS)
    _assert_valid_timing(t, N_ITERS)


def test_measure_graph_covers_every_op() -> None:
    """Walking a tiny op graph through the dispatcher touches every op type."""
    # Use a reduced config so the graph stays cheap but still has all op kinds.
    ops = build_op_graph(LLAMA_1B, seq_len=8, mode="prefill")
    # 1 embedding + 14 * 16 + 2 = 227 ops in Llama-1B. That's too many for CPU smoke.
    # Measure a small slice that still hits every op family.
    by_type: dict[type, Op] = {}
    for op in ops:
        by_type.setdefault(type(op), op)
    # At minimum: Embedding, RMSNorm, MatMul, RoPE, AttentionPrefill, SiLUGate, Residual.
    sampled = list(by_type.values())
    results = measure_graph(sampled, device=DEVICE, vocab_size=128,
                             n_warmup=N_WARMUP, n_iters=N_ITERS)
    assert len(results) == len(sampled)
    for op, timing in results:
        _assert_valid_timing(timing, N_ITERS)
        assert timing.median_ms >= 0.0
        assert op is not None


def test_measure_op_rejects_unknown_type() -> None:
    class FakeOp:
        name = "fake"
        dtype = "bf16"

        def flops(self) -> int:
            return 0

        def bytes_read(self) -> int:
            return 0

        def bytes_written(self) -> int:
            return 0

    with pytest.raises(TypeError, match="unknown op type"):
        measure_op(FakeOp(), device=DEVICE, n_iters=1, n_warmup=0)  # type: ignore[arg-type]
