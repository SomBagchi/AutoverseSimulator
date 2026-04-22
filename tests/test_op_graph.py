"""Tests for :func:`autoverse.model.build_op_graph`.

Verifies the shape of the lowered graph (counts, first/last ops, per-layer
structure, mode-dependent differences) without rechecking individual op formulas
— those live in :mod:`tests.test_ops_flops`.
"""

from __future__ import annotations

import pytest

from autoverse import LLAMA_1B
from autoverse.model import build_op_graph
from autoverse.ops import (
    AttentionDecode,
    AttentionPrefill,
    Embedding,
    MatMul,
    Residual,
    RMSNorm,
    RoPE,
    SiLUGate,
)


def test_graph_total_op_count_llama_1b() -> None:
    # 1 embedding + 16 layers * 14 ops_per_layer + 1 final RMSNorm + 1 LM head = 227.
    ops = build_op_graph(LLAMA_1B, seq_len=1024, mode="prefill")
    assert len(ops) == 1 + LLAMA_1B.n_layers * 14 + 2
    assert len(ops) == 227


def test_graph_starts_with_embedding_ends_with_lm_head() -> None:
    ops = build_op_graph(LLAMA_1B, seq_len=1024, mode="prefill")
    assert isinstance(ops[0], Embedding)
    assert isinstance(ops[-1], MatMul) and ops[-1].name == "lm_head"


def test_prefill_uses_attention_prefill_and_seq_len_tokens() -> None:
    seq_len = 1024
    ops = build_op_graph(LLAMA_1B, seq_len=seq_len, mode="prefill")
    attn_ops = [op for op in ops if isinstance(op, AttentionPrefill)]
    assert len(attn_ops) == LLAMA_1B.n_layers
    for attn in attn_ops:
        assert attn.seq_len == seq_len
        assert attn.n_heads == LLAMA_1B.n_heads
        assert attn.n_kv_heads == LLAMA_1B.n_kv_heads

    # MatMul's M axis = n_tokens = seq_len in prefill.
    q_projs = [op for op in ops if isinstance(op, MatMul) and op.name.startswith("q_proj")]
    assert all(op.m == seq_len for op in q_projs)


def test_decode_uses_attention_decode_and_single_token() -> None:
    ctx_len = 2048
    ops = build_op_graph(LLAMA_1B, seq_len=ctx_len, mode="decode")
    attn_ops = [op for op in ops if isinstance(op, AttentionDecode)]
    assert len(attn_ops) == LLAMA_1B.n_layers
    for attn in attn_ops:
        assert attn.ctx_len == ctx_len

    # n_tokens = 1 in decode mode for all per-token ops.
    q_projs = [op for op in ops if isinstance(op, MatMul) and op.name.startswith("q_proj")]
    assert all(op.m == 1 for op in q_projs)

    # No prefill-attention ops leak into decode.
    assert not any(isinstance(op, AttentionPrefill) for op in ops)


def test_graph_has_expected_per_layer_structure() -> None:
    """Each layer must contain: 2 RMSNorm, 4 proj MatMul (Q/K/V/Out), RoPE, 1 attn,
    3 MLP MatMul (gate/up/down), 1 SiLUGate, 2 Residual."""
    ops = build_op_graph(LLAMA_1B, seq_len=128, mode="prefill")
    counts = {
        "RMSNorm": sum(isinstance(op, RMSNorm) for op in ops),
        "MatMul": sum(isinstance(op, MatMul) for op in ops),
        "RoPE": sum(isinstance(op, RoPE) for op in ops),
        "Attention": sum(isinstance(op, (AttentionPrefill, AttentionDecode)) for op in ops),
        "SiLUGate": sum(isinstance(op, SiLUGate) for op in ops),
        "Residual": sum(isinstance(op, Residual) for op in ops),
        "Embedding": sum(isinstance(op, Embedding) for op in ops),
    }
    nl = LLAMA_1B.n_layers
    assert counts["Embedding"] == 1
    assert counts["RMSNorm"] == 2 * nl + 1  # pre-attn, pre-mlp, final
    assert counts["RoPE"] == nl
    assert counts["Attention"] == nl
    assert counts["SiLUGate"] == nl
    assert counts["Residual"] == 2 * nl
    # 4 projections per layer + 3 MLP matmuls + LM head.
    assert counts["MatMul"] == 4 * nl + 3 * nl + 1


def test_graph_op_names_unique() -> None:
    ops = build_op_graph(LLAMA_1B, seq_len=256, mode="prefill")
    names = [op.name for op in ops]
    # Embedding / final RMSNorm / LM head are global; per-layer ops are suffixed by layer index.
    assert len(names) == len(set(names))


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        # mypy would catch this; runtime guard still exists for robustness.
        build_op_graph(LLAMA_1B, seq_len=128, mode="train")  # type: ignore[arg-type]
