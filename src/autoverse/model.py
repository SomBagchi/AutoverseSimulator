"""Transformer configuration and op-graph construction.

Day 0: :class:`TransformerConfig` dataclass + the reference ``LLAMA_1B`` constant.
Day 1: :func:`build_op_graph` — lowers a config to a flat, execution-ordered op list.

At Tier 0 we only handle Llama-3.2-1B (decoder-only, GQA, SwiGLU, RoPE). The op
graph is deliberately linearised: no control-flow ops, no autotuning hooks.
Generalisation to other families waits until Tier 3 actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

#: Forward-pass mode. ``"prefill"`` processes ``seq_len`` tokens at once;
#: ``"decode"`` processes one token given a KV cache of length ``seq_len``.
Mode = Literal["prefill", "decode"]


@dataclass(frozen=True)
class TransformerConfig:
    """Dimensional parameters of a decoder-only transformer.

    Attributes:
        name: Short identifier.
        d_model: Residual-stream width.
        n_layers: Number of transformer blocks.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads (equal to ``n_heads`` for MHA; fewer for GQA).
        d_head: Per-head embedding dimension. ``d_model == n_heads * d_head``.
        d_ffn: MLP inner dimension.
        vocab_size: Vocabulary size (shared between embedding and LM head).
        dtype: Activation/weight dtype for the simulated run.
    """

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ffn: int
    vocab_size: int
    dtype: str = "bf16"


LLAMA_1B = TransformerConfig(
    name="llama-3.2-1b",
    d_model=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,  # GQA: 4 Q-heads per KV-head.
    d_head=64,
    d_ffn=8192,
    vocab_size=128_256,
    dtype="bf16",
)


def build_op_graph(cfg: TransformerConfig, seq_len: int, mode: Mode) -> list[Op]:
    """Build a flat, execution-ordered op graph for one forward pass.

    Arguments:
        cfg: Transformer dimensions.
        seq_len: In ``"prefill"``, the sequence length processed. In ``"decode"``,
            the length of the prior KV cache (one new token is processed).
        mode: ``"prefill"`` (parallel over all tokens) or ``"decode"`` (one token).

    Shape semantics:
        - **prefill:** ``n_tokens = seq_len`` tokens flow through every op.
          Attention sees ``seq_len x seq_len`` scores.
        - **decode:**  ``n_tokens = 1`` for all per-token ops; attention reads
          a length-``seq_len`` KV cache.

    The graph is linear; the simulator consumes it sequentially.
    """
    if mode == "prefill":
        n_tokens = seq_len
    elif mode == "decode":
        n_tokens = 1
    else:  # pragma: no cover — Literal exhaustively covers valid values.
        raise ValueError(f"unknown mode: {mode!r}")

    d = cfg.d_model
    h = cfg.n_heads
    h_kv = cfg.n_kv_heads
    d_h = cfg.d_head
    ffn = cfg.d_ffn
    vocab = cfg.vocab_size
    dtype = cfg.dtype

    ops: list[Op] = []

    ops.append(Embedding(n_tokens=n_tokens, d_model=d, dtype=dtype))

    for layer in range(cfg.n_layers):
        ops.append(
            RMSNorm(
                n_tokens=n_tokens,
                d_model=d,
                name=f"rmsnorm_pre_attn_{layer}",
                dtype=dtype,
            )
        )
        ops.append(MatMul(m=n_tokens, k=d, n=h * d_h, name=f"q_proj_{layer}", dtype=dtype))
        ops.append(MatMul(m=n_tokens, k=d, n=h_kv * d_h, name=f"k_proj_{layer}", dtype=dtype))
        ops.append(MatMul(m=n_tokens, k=d, n=h_kv * d_h, name=f"v_proj_{layer}", dtype=dtype))
        ops.append(
            RoPE(
                n_tokens=n_tokens,
                n_heads=h,
                n_kv_heads=h_kv,
                d_head=d_h,
                name=f"rope_{layer}",
                dtype=dtype,
            )
        )
        if mode == "prefill":
            ops.append(
                AttentionPrefill(
                    batch=1,
                    seq_len=seq_len,
                    n_heads=h,
                    n_kv_heads=h_kv,
                    d_head=d_h,
                    name=f"attn_prefill_{layer}",
                    dtype=dtype,
                )
            )
        else:
            ops.append(
                AttentionDecode(
                    batch=1,
                    ctx_len=seq_len,
                    n_heads=h,
                    n_kv_heads=h_kv,
                    d_head=d_h,
                    name=f"attn_decode_{layer}",
                    dtype=dtype,
                )
            )
        ops.append(MatMul(m=n_tokens, k=h * d_h, n=d, name=f"out_proj_{layer}", dtype=dtype))
        ops.append(
            Residual(
                n_tokens=n_tokens,
                d_model=d,
                name=f"residual_attn_{layer}",
                dtype=dtype,
            )
        )

        ops.append(
            RMSNorm(
                n_tokens=n_tokens,
                d_model=d,
                name=f"rmsnorm_pre_mlp_{layer}",
                dtype=dtype,
            )
        )
        ops.append(MatMul(m=n_tokens, k=d, n=ffn, name=f"mlp_gate_{layer}", dtype=dtype))
        ops.append(MatMul(m=n_tokens, k=d, n=ffn, name=f"mlp_up_{layer}", dtype=dtype))
        ops.append(
            SiLUGate(
                n_tokens=n_tokens,
                d_ffn=ffn,
                name=f"silu_gate_{layer}",
                dtype=dtype,
            )
        )
        ops.append(MatMul(m=n_tokens, k=ffn, n=d, name=f"mlp_down_{layer}", dtype=dtype))
        ops.append(
            Residual(
                n_tokens=n_tokens,
                d_model=d,
                name=f"residual_mlp_{layer}",
                dtype=dtype,
            )
        )

    ops.append(RMSNorm(n_tokens=n_tokens, d_model=d, name="rmsnorm_final", dtype=dtype))
    ops.append(MatMul(m=n_tokens, k=d, n=vocab, name="lm_head", dtype=dtype))

    return ops
