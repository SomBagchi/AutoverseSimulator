"""Real-hardware measurement harness.

Times each primitive operator using ``torch.cuda.Event`` pairs (or
``time.perf_counter`` on CPU/MPS) with warm-up and high iteration counts,
returning median + percentile latencies.

This module is the **only** one that imports ``torch``. Torch is an optional
dependency (``uv sync --extra measure``) — the rest of the library is
CPU-only and CI doesn't install it. Tests for this module skip gracefully
when torch is absent.

Measurement methodology
-----------------------
- **Warm-up** (default 10 iters) absorbs JIT/cuBLAS algorithm selection costs.
- **CUDA events** (``Event(enable_timing=True)``): each iter is bracketed by a
  pair of events, all dispatched into the same stream; ``synchronize()`` once
  at the end; then ``elapsed_time`` reads. This avoids per-iter
  synchronisation overhead.
- **CPU/MPS fallback**: ``time.perf_counter`` delta per iter. Not accurate for
  real kernel timing (MPS is async, CPU is host-timed), but enough for smoke
  tests of structure/plumbing.
- Returned timings are in **milliseconds**.

Per-op dispatch (:func:`measure_op`) takes an :class:`~autoverse.ops.Op` and
routes to the right primitive function, so calibration can walk a graph built
by :func:`~autoverse.model.build_op_graph` and produce
``(op, MeasuredTiming)`` pairs directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from autoverse.ops import (
    AttentionDecode,
    AttentionPrefill,
    Embedding,
    MatMul,
    Op,
    Residual,
    RMSNorm,
    RoPE,
    SiLUGate,
)


@dataclass(frozen=True)
class MeasuredTiming:
    """Aggregated timing summary across ``n_iters`` measured runs.

    All times are in **milliseconds**. ``median_ms`` is the preferred
    single-number latency — robust to outliers from noise (e.g.,
    GPU clock jitter, preemption). ``p10/p90`` gives a noise band.
    """

    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    std_ms: float
    n_iters: int


_DTYPE_NAMES = ("bf16", "fp16", "fp32")


def _torch() -> Any:
    """Lazy import of ``torch`` — kept out of module scope so mypy / CI without
    the ``measure`` extra do not choke importing this file."""
    import torch

    return torch


def _torch_dtype(name: str) -> Any:
    torch = _torch()
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype for measurement: {name!r}")
    return mapping[name]


def _resolve_device(device: str) -> str:
    """Accept ``'cuda'`` / ``'cpu'`` / ``'mps'`` / ``'auto'``; fall back as needed."""
    torch = _torch()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def time_callable(
    fn: Callable[[], None],
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """Time a zero-argument callable on the given device.

    Uses CUDA events when ``device='cuda'`` and CUDA is available; otherwise
    wall-clock via ``time.perf_counter``. Warm-up runs are not timed.
    """
    torch = _torch()
    use_cuda_events = device == "cuda" and torch.cuda.is_available()

    for _ in range(n_warmup):
        fn()
    if use_cuda_events:
        torch.cuda.synchronize()

    if use_cuda_events:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        for i in range(n_iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times_ms = [float(starts[i].elapsed_time(ends[i])) for i in range(n_iters)]
    else:
        times_ms = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(times_ms, dtype=np.float64)
    return MeasuredTiming(
        median_ms=float(np.median(arr)),
        p10_ms=float(np.percentile(arr, 10)),
        p90_ms=float(np.percentile(arr, 90)),
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        n_iters=n_iters,
    )


# ---------- Primitive measurements ----------


def measure_matmul(
    m: int,
    k: int,
    n: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """Time ``x @ w`` where ``x: (m, k)``, ``w: (k, n)``."""
    torch = _torch()
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    x = torch.randn(m, k, dtype=td, device=dev)
    w = torch.randn(k, n, dtype=td, device=dev)

    def fn() -> None:
        _ = x @ w

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_attention_prefill(
    batch: int,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """Causal self-attention over ``seq_len`` tokens (flash-style via SDPA)."""
    torch = _torch()
    f = torch.nn.functional
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    q = torch.randn(batch, n_heads, seq_len, d_head, dtype=td, device=dev)
    k_ = torch.randn(batch, n_kv_heads, seq_len, d_head, dtype=td, device=dev)
    v = torch.randn(batch, n_kv_heads, seq_len, d_head, dtype=td, device=dev)
    gqa = n_heads != n_kv_heads

    def fn() -> None:
        _ = f.scaled_dot_product_attention(q, k_, v, is_causal=True, enable_gqa=gqa)

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_attention_decode(
    batch: int,
    ctx_len: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """One-token-Q against length-``ctx_len`` KV cache. No causal mask needed
    (Q is a single new token; it attends to the entire cache)."""
    torch = _torch()
    f = torch.nn.functional
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    q = torch.randn(batch, n_heads, 1, d_head, dtype=td, device=dev)
    k_ = torch.randn(batch, n_kv_heads, ctx_len, d_head, dtype=td, device=dev)
    v = torch.randn(batch, n_kv_heads, ctx_len, d_head, dtype=td, device=dev)
    gqa = n_heads != n_kv_heads

    def fn() -> None:
        _ = f.scaled_dot_product_attention(q, k_, v, is_causal=False, enable_gqa=gqa)

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_rmsnorm(
    n_tokens: int,
    d_model: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """RMSNorm with a learned gain vector (no bias)."""
    torch = _torch()
    f = torch.nn.functional
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    x = torch.randn(n_tokens, d_model, dtype=td, device=dev)
    gain = torch.randn(d_model, dtype=td, device=dev)

    def fn() -> None:
        _ = f.rms_norm(x, (d_model,), gain, eps=1e-5)

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_silu_gate(
    n_tokens: int,
    d_ffn: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """SwiGLU elementwise: ``silu(gate) * up``, both shape ``(n_tokens, d_ffn)``."""
    torch = _torch()
    f = torch.nn.functional
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    gate = torch.randn(n_tokens, d_ffn, dtype=td, device=dev)
    up = torch.randn(n_tokens, d_ffn, dtype=td, device=dev)

    def fn() -> None:
        _ = f.silu(gate) * up

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_residual(
    n_tokens: int,
    d_model: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """Elementwise add: ``a + b``."""
    torch = _torch()
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    a = torch.randn(n_tokens, d_model, dtype=td, device=dev)
    b = torch.randn(n_tokens, d_model, dtype=td, device=dev)

    def fn() -> None:
        _ = a + b

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_embedding(
    n_tokens: int,
    d_model: int,
    vocab_size: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """Gather ``n_tokens`` rows out of a ``(vocab_size, d_model)`` table."""
    torch = _torch()
    f = torch.nn.functional
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    weight = torch.randn(vocab_size, d_model, dtype=td, device=dev)
    idx = torch.randint(0, vocab_size, (n_tokens,), device=dev)

    def fn() -> None:
        _ = f.embedding(idx, weight)

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


def measure_rope(
    n_tokens: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    dtype: str = "bf16",
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """RoPE applied to Q and K stacked along the head axis. Implemented as the
    standard pair-rotation: split into even/odd halves, rotate with cos/sin."""
    torch = _torch()
    dev = _resolve_device(device)
    td = _torch_dtype(dtype)
    total_heads = n_heads + n_kv_heads
    x = torch.randn(n_tokens, total_heads, d_head, dtype=td, device=dev)
    half = d_head // 2
    cos = torch.randn(n_tokens, 1, half, dtype=td, device=dev)
    sin = torch.randn(n_tokens, 1, half, dtype=td, device=dev)

    def fn() -> None:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        _ = torch.stack([y1, y2], dim=-1).flatten(-2)

    return time_callable(fn, device=dev, n_warmup=n_warmup, n_iters=n_iters)


# ---------- Dispatcher ----------


def measure_op(
    op: Op,
    device: str = "cuda",
    vocab_size: int = 128_256,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> MeasuredTiming:
    """Dispatch an :class:`~autoverse.ops.Op` to the matching measurement primitive.

    ``vocab_size`` is only used by :class:`~autoverse.ops.Embedding` (which
    doesn't carry it as a field, since it doesn't affect FLOPs/bytes in the
    cost model). Defaults to Llama-1B's vocab.
    """
    d = device
    w = n_warmup
    i = n_iters
    if isinstance(op, MatMul):
        return measure_matmul(op.m, op.k, op.n, dtype=op.dtype, device=d, n_warmup=w, n_iters=i)
    if isinstance(op, AttentionPrefill):
        return measure_attention_prefill(
            op.batch, op.seq_len, op.n_heads, op.n_kv_heads, op.d_head,
            dtype=op.dtype, device=d, n_warmup=w, n_iters=i,
        )
    if isinstance(op, AttentionDecode):
        return measure_attention_decode(
            op.batch, op.ctx_len, op.n_heads, op.n_kv_heads, op.d_head,
            dtype=op.dtype, device=d, n_warmup=w, n_iters=i,
        )
    if isinstance(op, RMSNorm):
        return measure_rmsnorm(op.n_tokens, op.d_model, dtype=op.dtype,
                               device=d, n_warmup=w, n_iters=i)
    if isinstance(op, SiLUGate):
        return measure_silu_gate(op.n_tokens, op.d_ffn, dtype=op.dtype,
                                 device=d, n_warmup=w, n_iters=i)
    if isinstance(op, Residual):
        return measure_residual(op.n_tokens, op.d_model, dtype=op.dtype,
                                device=d, n_warmup=w, n_iters=i)
    if isinstance(op, Embedding):
        return measure_embedding(op.n_tokens, op.d_model, vocab_size,
                                 dtype=op.dtype, device=d, n_warmup=w, n_iters=i)
    if isinstance(op, RoPE):
        return measure_rope(op.n_tokens, op.n_heads, op.n_kv_heads, op.d_head,
                            dtype=op.dtype, device=d, n_warmup=w, n_iters=i)
    raise TypeError(f"unknown op type: {type(op).__name__}")


def measure_graph(
    ops: list[Op],
    device: str = "cuda",
    vocab_size: int = 128_256,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> list[tuple[Op, MeasuredTiming]]:
    """Measure every op in an execution-ordered graph, in isolation.

    Returns a list of ``(op, MeasuredTiming)`` pairs. Summing the median
    latencies gives a **compositional** E2E latency — it ignores ordering
    effects, cache state carrying between ops, and graph-level fusions, but
    matches what our roofline predictor also assumes.
    """
    return [(op, measure_op(op, device=device, vocab_size=vocab_size,
                             n_warmup=n_warmup, n_iters=n_iters)) for op in ops]
