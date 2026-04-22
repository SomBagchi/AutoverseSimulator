"""Unit tests pinning FLOP and byte formulas for every concrete op.

If these numbers move, something in the modelling changed — the fix is either
to update the op implementation or, very deliberately, to update this test.
"""

from __future__ import annotations

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

# ---------- MatMul ----------


def test_matmul_flops_square_gemm() -> None:
    op = MatMul(m=1024, k=2048, n=2048)
    assert op.flops() == 2 * 1024 * 2048 * 2048


def test_matmul_flops_rectangular_gemm() -> None:
    op = MatMul(m=1, k=2048, n=128_256)  # decode-mode LM head, Llama-1B
    assert op.flops() == 2 * 1 * 128_256 * 2048


def test_matmul_bytes_bf16() -> None:
    op = MatMul(m=1024, k=2048, n=2048, dtype="bf16")
    assert op.bytes_read() == (1024 * 2048 + 2048 * 2048) * 2
    assert op.bytes_written() == 1024 * 2048 * 2


def test_matmul_bytes_fp32_uses_dtype_bytes() -> None:
    op = MatMul(m=1, k=128, n=256, dtype="fp32")
    assert op.bytes_read() == (1 * 128 + 128 * 256) * 4
    assert op.bytes_written() == 1 * 256 * 4


# ---------- Embedding ----------


def test_embedding_is_pure_gather() -> None:
    op = Embedding(n_tokens=16, d_model=2048)
    assert op.flops() == 0
    assert op.bytes_read() == 16 * 2048 * 2
    assert op.bytes_written() == 16 * 2048 * 2


# ---------- RMSNorm ----------


def test_rmsnorm_flops_and_bytes() -> None:
    op = RMSNorm(n_tokens=32, d_model=2048)
    assert op.flops() == 4 * 32 * 2048
    # +1 row to cover the gain vector.
    assert op.bytes_read() == (32 + 1) * 2048 * 2
    assert op.bytes_written() == 32 * 2048 * 2


# ---------- RoPE ----------


def test_rope_three_flops_per_element() -> None:
    op = RoPE(n_tokens=1, n_heads=32, n_kv_heads=8, d_head=64)
    elems = 1 * (32 + 8) * 64
    assert op.flops() == 3 * elems
    assert op.bytes_read() == elems * 2
    assert op.bytes_written() == elems * 2


# ---------- Attention (prefill) ----------


def test_attention_prefill_flops_quadratic_in_L() -> None:
    small = AttentionPrefill(batch=1, seq_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    big = AttentionPrefill(batch=1, seq_len=2048, n_heads=32, n_kv_heads=8, d_head=64)
    assert big.flops() == 4 * small.flops()


def test_attention_prefill_gqa_bytes() -> None:
    op = AttentionPrefill(batch=1, seq_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    # Bytes in = Q (h=32) + K (h_kv=8) + V (h_kv=8), all per-token.
    assert op.bytes_read() == 1 * 1024 * (32 + 2 * 8) * 64 * 2
    assert op.bytes_written() == 1 * 1024 * 32 * 64 * 2


# ---------- Attention (decode) ----------


def test_attention_decode_flops_formula() -> None:
    op = AttentionDecode(batch=1, ctx_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    assert op.flops() == 4 * 1 * 32 * 1024 * 64


def test_attention_decode_bytes_dominated_by_kv_cache() -> None:
    op = AttentionDecode(batch=1, ctx_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    assert op.bytes_read() == 2 * 1 * 8 * 1024 * 64 * 2
    assert op.bytes_written() == 1 * 32 * 64 * 2


def test_attention_decode_arithmetic_intensity_matches_gqa_math() -> None:
    """AI = 2 * h / (h_kv * dtype_bytes). Llama-1B bf16 ⇒ 4 ops/byte."""
    op = AttentionDecode(batch=1, ctx_len=1024, n_heads=32, n_kv_heads=8, d_head=64)
    ai = op.flops() / op.bytes_read()
    assert abs(ai - 4.0) < 1e-9


# ---------- SiLUGate ----------


def test_silu_gate_flops_and_bytes() -> None:
    op = SiLUGate(n_tokens=32, d_ffn=8192)
    assert op.flops() == 6 * 32 * 8192
    assert op.bytes_read() == 2 * 32 * 8192 * 2
    assert op.bytes_written() == 32 * 8192 * 2


# ---------- Residual ----------


def test_residual_flops_and_bytes() -> None:
    op = Residual(n_tokens=32, d_model=2048)
    assert op.flops() == 32 * 2048
    assert op.bytes_read() == 2 * 32 * 2048 * 2
    assert op.bytes_written() == 32 * 2048 * 2


# ---------- Protocol conformance ----------


def test_all_ops_conform_to_op_protocol() -> None:
    """Every concrete op is structurally an ``Op`` (runtime_checkable Protocol)."""
    ops: list[Op] = [
        Embedding(n_tokens=1, d_model=2048),
        RMSNorm(n_tokens=1, d_model=2048),
        MatMul(m=1, k=2048, n=2048),
        RoPE(n_tokens=1, n_heads=32, n_kv_heads=8, d_head=64),
        AttentionPrefill(batch=1, seq_len=16, n_heads=32, n_kv_heads=8, d_head=64),
        AttentionDecode(batch=1, ctx_len=16, n_heads=32, n_kv_heads=8, d_head=64),
        SiLUGate(n_tokens=1, d_ffn=8192),
        Residual(n_tokens=1, d_model=2048),
    ]
    for op in ops:
        assert isinstance(op, Op)
        assert op.flops() >= 0
        assert op.bytes_read() >= 0
        assert op.bytes_written() >= 0
        assert isinstance(op.name, str) and op.name
