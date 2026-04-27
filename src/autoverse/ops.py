"""Operator representations — shape + FLOP/byte accounting.

Each concrete op is a frozen dataclass that knows its own FLOP count and the
bytes it reads and writes. The analytical cost model (:mod:`autoverse.cost`)
uses these to produce latency estimates.

Conventions
-----------
- **1 multiply + 1 add = 2 FLOPs.** Canonical for GEMM accounting.
- **Bytes are HBM round-trips only.** No intermediate materialisation; we
  model flash-style attention. L2 hit-rate effects are layered on top in
  :func:`autoverse.cost.estimate`, not baked into the byte counts here.
- Shape shorthand: ``M = n_tokens = batch * seq_len`` for the matmul view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Byte widths per supported dtype. Add entries here when a new precision is modelled.
DTYPE_BYTES: dict[str, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp8": 1,
}


@runtime_checkable
class Op(Protocol):
    """An operator contributes FLOPs and bytes-moved to a workload.

    Concrete ops below are frozen dataclasses that conform structurally.
    ``name`` and ``dtype`` are declared as read-only properties so frozen
    dataclass fields satisfy the protocol (mutable-attribute protocols reject
    read-only-attribute implementations under ``mypy --strict``).
    """

    @property
    def name(self) -> str: ...

    @property
    def dtype(self) -> str: ...

    def flops(self) -> int:
        """Total multiply-add-equivalent FLOPs for this op."""
        ...

    def bytes_read(self) -> int:
        """Bytes read from HBM (the cost model layers L2 hit-rate on top)."""
        ...

    def bytes_written(self) -> int:
        """Bytes written to HBM."""
        ...


def dtype_bytes(dtype: str) -> int:
    """Return the byte width of a named dtype. Raises :class:`KeyError` if unknown."""
    return DTYPE_BYTES[dtype]


@dataclass(frozen=True)
class Embedding:
    """Token-embedding lookup (gather).

    FLOPs: 0 — pure memory op.

    Bytes read:    ``n_tokens * d_model * dtype_bytes`` (only the *selected* rows).
    Bytes written: ``n_tokens * d_model * dtype_bytes``.
    """

    n_tokens: int
    d_model: int
    name: str = "embedding"
    dtype: str = "bf16"

    def flops(self) -> int:
        return 0

    def bytes_read(self) -> int:
        return self.n_tokens * self.d_model * dtype_bytes(self.dtype)

    def bytes_written(self) -> int:
        return self.n_tokens * self.d_model * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class RMSNorm:
    """RMSNorm: ``y = x * rsqrt(mean(x**2) + eps) * gain``.

    FLOPs: ~``4 * n_tokens * d_model``
      - sum-of-squares:      ~2 * d_model per token (mul + add).
      - scale + gain mul:    ~2 * d_model per token.
      - rsqrt/div/eps:       O(1) per token — negligible.

    Bytes read:    ``(n_tokens + 1) * d_model * dtype_bytes`` (x + gain vector).
    Bytes written: ``n_tokens * d_model * dtype_bytes``.
    """

    n_tokens: int
    d_model: int
    name: str = "rmsnorm"
    dtype: str = "bf16"

    def flops(self) -> int:
        return 4 * self.n_tokens * self.d_model

    def bytes_read(self) -> int:
        return (self.n_tokens + 1) * self.d_model * dtype_bytes(self.dtype)

    def bytes_written(self) -> int:
        return self.n_tokens * self.d_model * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class MatMul:
    """Generic GEMM: ``(M, K) @ (K, N) -> (M, N)``.

    Covers Q/K/V/Out projections, MLP gate/up/down, and the LM head — they
    differ only in the (M, K, N) triple.

    FLOPs: ``2 * M * N * K`` (each output element: K muls + K adds ≈ 2K ops).

    Bytes read:    ``(M*K + K*N) * dtype_bytes`` (activation + weight).
    Bytes written: ``M*N * dtype_bytes`` (output).
    """

    m: int
    k: int
    n: int
    name: str = "matmul"
    dtype: str = "bf16"

    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def bytes_read(self) -> int:
        return (self.m * self.k + self.k * self.n) * dtype_bytes(self.dtype)

    def bytes_written(self) -> int:
        return self.m * self.n * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class RoPE:
    """Rotary position embedding applied to Q and K.

    Per (even, odd) element pair: ``y_e = x_e*cos - x_o*sin``;
    ``y_o = x_e*sin + x_o*cos`` — 4 muls + 2 adds = 6 FLOPs per pair ⇒ 3 FLOPs
    per scalar element.

    FLOPs: ``3 * n_tokens * (n_heads + n_kv_heads) * d_head``.

    Bytes read:    ``n_tokens * (n_heads + n_kv_heads) * d_head * dtype_bytes``
                   (Q and K activations; cos/sin table is tiny, ignored).
    Bytes written: same size as bytes_read (in-place-style).
    """

    n_tokens: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    name: str = "rope"
    dtype: str = "bf16"

    def _elems(self) -> int:
        return self.n_tokens * (self.n_heads + self.n_kv_heads) * self.d_head

    def flops(self) -> int:
        return 3 * self._elems()

    def bytes_read(self) -> int:
        return self._elems() * dtype_bytes(self.dtype)

    def bytes_written(self) -> int:
        return self._elems() * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class AttentionPrefill:
    """Causal self-attention over the full sequence, flash-style bytes.

    With GQA, each query head attends over K/V shared across its group, so
    FLOPs use ``n_heads`` while KV-cache bytes scale with ``n_kv_heads``.

    FLOPs: ~``4 * batch * n_heads * seq_len^2 * d_head``
      - ``QK^T``:   ``2 * B * h * L^2 * d_h``
      - softmax:    ~``3 * B * h * L^2`` (negligible vs. matmuls).
      - ``AV``:     ``2 * B * h * L^2 * d_h``

    Bytes read:    ``B * L * (n_heads + 2 * n_kv_heads) * d_head * dtype_bytes``
                   (Q + K + V; scores not materialized — flash-style).
    Bytes written: ``B * L * n_heads * d_head * dtype_bytes`` (output).
    """

    batch: int
    seq_len: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    name: str = "attn_prefill"
    dtype: str = "bf16"

    def flops(self) -> int:
        b, el, h, d_h = self.batch, self.seq_len, self.n_heads, self.d_head
        return 4 * b * h * el * el * d_h

    def bytes_read(self) -> int:
        ib = dtype_bytes(self.dtype)
        b, el, h, h_kv, d_h = (
            self.batch,
            self.seq_len,
            self.n_heads,
            self.n_kv_heads,
            self.d_head,
        )
        return b * el * (h + 2 * h_kv) * d_h * ib

    def bytes_written(self) -> int:
        ib = dtype_bytes(self.dtype)
        b, el, h, d_h = self.batch, self.seq_len, self.n_heads, self.d_head
        return b * el * h * d_h * ib


@dataclass(frozen=True)
class AttentionDecode:
    """Decode attention: one new-token Q against a length-``ctx_len`` KV cache.

    FLOPs: ~``4 * batch * n_heads * ctx_len * d_head``.

    Bytes read:    ``2 * batch * n_kv_heads * ctx_len * d_head * dtype_bytes``
                   (K and V cache — dominant; Q is tiny and ignored).
    Bytes written: ``batch * n_heads * d_head * dtype_bytes`` (output, tiny).

    Arithmetic intensity
    --------------------
    ``AI = 2 * n_heads / (n_kv_heads * dtype_bytes)``.
    For Llama-1B BF16: ``AI = 2*32/(8*2) = 4 ops/byte`` vs. H100 balance
    ~295 ops/byte ⇒ hard HBM-bound. Setting ``n_kv_heads = n_heads`` recovers
    the MHA case: ``AI = 2/ι = 1 op/byte`` in BF16. GQA (4 Q heads per KV head
    in Llama-1B) raises AI 4× over MHA but still lands nowhere near the
    compute roof — decode is memory-bound under any reasonable attention
    variant on this hardware.
    """

    batch: int
    ctx_len: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    name: str = "attn_decode"
    dtype: str = "bf16"

    def flops(self) -> int:
        b, lc, h, d_h = self.batch, self.ctx_len, self.n_heads, self.d_head
        return 4 * b * h * lc * d_h

    def bytes_read(self) -> int:
        ib = dtype_bytes(self.dtype)
        b, lc, h_kv, d_h = self.batch, self.ctx_len, self.n_kv_heads, self.d_head
        return 2 * b * h_kv * lc * d_h * ib

    def bytes_written(self) -> int:
        ib = dtype_bytes(self.dtype)
        b, h, d_h = self.batch, self.n_heads, self.d_head
        return b * h * d_h * ib


@dataclass(frozen=True)
class SiLUGate:
    """SwiGLU elementwise: ``y = SiLU(gate) * up``, both shape ``(n_tokens, d_ffn)``.

    FLOPs: ~``6 * n_tokens * d_ffn``
      - ``SiLU(x) = x * sigmoid(x)`` ≈ 5 FLOPs/elem (exp, add, div, mul).
      - gate-up elementwise mul: 1 FLOP/elem.

    Bytes read:    ``2 * n_tokens * d_ffn * dtype_bytes`` (gate + up).
    Bytes written: ``n_tokens * d_ffn * dtype_bytes``.
    """

    n_tokens: int
    d_ffn: int
    name: str = "silu_gate"
    dtype: str = "bf16"

    def flops(self) -> int:
        return 6 * self.n_tokens * self.d_ffn

    def bytes_read(self) -> int:
        return 2 * self.n_tokens * self.d_ffn * dtype_bytes(self.dtype)

    def bytes_written(self) -> int:
        return self.n_tokens * self.d_ffn * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class Residual:
    """Residual add: ``y = a + b``, both shape ``(n_tokens, d_model)``.

    FLOPs: ``n_tokens * d_model``.

    Bytes read:    ``2 * n_tokens * d_model * dtype_bytes``.
    Bytes written: ``n_tokens * d_model * dtype_bytes``.
    """

    n_tokens: int
    d_model: int
    name: str = "residual"
    dtype: str = "bf16"

    def flops(self) -> int:
        return self.n_tokens * self.d_model

    def bytes_read(self) -> int:
        return 2 * self.n_tokens * self.d_model * dtype_bytes(self.dtype)

    def bytes_written(self) -> int:
        return self.n_tokens * self.d_model * dtype_bytes(self.dtype)
