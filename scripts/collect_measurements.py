"""Sweep real-hardware measurements for calibration (checkpoint 2B).

Typical use on an H100 pod:

    uv sync --extra measure
    uv run python scripts/collect_measurements.py \\
        --device cuda \\
        --out measurements/h100_sxm/run_$(date +%Y%m%d_%H%M%S).json

The CPU path (``--device cpu``, add ``--quick``) runs in a few seconds locally
for dry-run / pipeline testing.

Sweep coverage (matches ``04_autoverse_implementation_plan.md`` §Day 2.2):

- MatMul: full Cartesian product over ``M x K x N`` in {256, 1024, 2048, 4096}
  (64 shapes), bf16 (or fp32 with ``--dtype fp32``).
- AttentionPrefill: seq_len in {128, 512, 1024, 2048}, Llama-1B head config.
- AttentionDecode: ctx_len in {128, 1024, 4096}, Llama-1B head config.
- RMSNorm, SiLUGate, Residual, Embedding, RoPE: representative Llama-1B shapes.
- E2E Llama-1B: full op-graph walk, prefill seq_len=1024 + decode at ctx_len=1024.

Output JSON schema::

    {
        "device": "cuda",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "torch_version": "2.11.0",
        "timestamp_utc": "2026-04-23T10:12:00Z",
        "n_warmup": 10,
        "n_iters": 100,
        "dtype": "bf16",
        "measurements": [
            {
                "op_type": "MatMul",
                "op_name": "matmul",      // from Op.name, layer index preserved
                "params": {"m": 256, "k": 256, "n": 256, "dtype": "bf16"},
                "median_ms": 0.05, "p10_ms": ..., "p90_ms": ..., "mean_ms": ..., "std_ms": ...,
                "n_iters": 100
            },
            ...
        ]
    }

Consumed by :mod:`autoverse.calibrate` (checkpoint 2C).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from autoverse import LLAMA_1B
from autoverse.model import build_op_graph
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

MATMUL_AXES_FULL = (256, 1024, 2048, 4096)
MATMUL_AXES_QUICK = (128, 256)

PREFILL_SEQ_LENS_FULL = (128, 512, 1024, 2048)
PREFILL_SEQ_LENS_QUICK = (32, 128)

DECODE_CTX_LENS_FULL = (128, 1024, 4096)
DECODE_CTX_LENS_QUICK = (32, 128)


def _gpu_name(device: str) -> str:
    """Best-effort GPU product name for the provenance header."""
    if device != "cuda":
        return f"(no-cuda: {device})"
    import torch

    if not torch.cuda.is_available():
        return "(cuda-requested-but-unavailable)"
    return str(torch.cuda.get_device_name(0))


def _op_to_record(op: Op) -> dict[str, Any]:
    """Capture the shape-carrying fields of an op (everything except 'name')."""
    fields = dataclasses.fields(op)  # type: ignore[arg-type]
    return {
        "op_type": type(op).__name__,
        "op_name": op.name,
        "params": {f.name: getattr(op, f.name) for f in fields if f.name != "name"},
    }


def _timing_to_record(t: Any) -> dict[str, float | int]:
    return {
        "median_ms": t.median_ms,
        "p10_ms": t.p10_ms,
        "p90_ms": t.p90_ms,
        "mean_ms": t.mean_ms,
        "std_ms": t.std_ms,
        "n_iters": t.n_iters,
    }


def _build_sweep(dtype: str, quick: bool) -> list[Op]:
    """Return the list of ops we will measure, in the order we'll measure them."""
    axes = MATMUL_AXES_QUICK if quick else MATMUL_AXES_FULL
    prefill_lens = PREFILL_SEQ_LENS_QUICK if quick else PREFILL_SEQ_LENS_FULL
    decode_lens = DECODE_CTX_LENS_QUICK if quick else DECODE_CTX_LENS_FULL

    ops: list[Op] = []

    for m, k, n in itertools.product(axes, repeat=3):
        ops.append(MatMul(m=m, k=k, n=n, name=f"matmul_{m}x{k}x{n}", dtype=dtype))

    h = LLAMA_1B.n_heads
    h_kv = LLAMA_1B.n_kv_heads
    d_h = LLAMA_1B.d_head

    for sl in prefill_lens:
        ops.append(AttentionPrefill(
            batch=1, seq_len=sl, n_heads=h, n_kv_heads=h_kv, d_head=d_h,
            name=f"attn_prefill_L{sl}", dtype=dtype,
        ))

    for cl in decode_lens:
        ops.append(AttentionDecode(
            batch=1, ctx_len=cl, n_heads=h, n_kv_heads=h_kv, d_head=d_h,
            name=f"attn_decode_C{cl}", dtype=dtype,
        ))

    ops.append(RMSNorm(n_tokens=1024, d_model=LLAMA_1B.d_model,
                       name="rmsnorm_repr", dtype=dtype))
    ops.append(SiLUGate(n_tokens=1024, d_ffn=LLAMA_1B.d_ffn,
                        name="silu_gate_repr", dtype=dtype))
    ops.append(Residual(n_tokens=1024, d_model=LLAMA_1B.d_model,
                        name="residual_repr", dtype=dtype))
    ops.append(Embedding(n_tokens=1024, d_model=LLAMA_1B.d_model,
                         name="embedding_repr", dtype=dtype))
    ops.append(RoPE(n_tokens=1024, n_heads=h, n_kv_heads=h_kv, d_head=d_h,
                    name="rope_repr", dtype=dtype))

    return ops


def _build_e2e(dtype: str, quick: bool) -> list[Op]:
    """Full Llama-1B op graph: prefill then decode, tagged by mode."""
    if quick:
        prefill_len = 128
        decode_ctx = 128
    else:
        prefill_len = 1024
        decode_ctx = 1024

    prefill = build_op_graph(LLAMA_1B, seq_len=prefill_len, mode="prefill")
    decode = build_op_graph(LLAMA_1B, seq_len=decode_ctx, mode="decode")
    # Tag names so the two passes don't collide in the JSON.
    return [*prefill, *decode]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", type=Path, required=True, help="JSON output path")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--quick", action="store_true",
                   help="Shrink sweeps for dry-run on CPU.")
    p.add_argument("--skip-e2e", action="store_true",
                   help="Skip the full Llama-1B op-graph walk (227 ops x 2 modes).")
    args = p.parse_args()

    # Import torch lazily — this script may be imported in --help on a box
    # without torch, and we want that to succeed.
    import torch  # type: ignore[import-not-found]

    from autoverse.measure import measure_op  # noqa: F401

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_ops = _build_sweep(args.dtype, args.quick)
    if not args.skip_e2e:
        sweep_ops.extend(_build_e2e(args.dtype, args.quick))

    print(f"[collect] device={device} dtype={args.dtype} "
          f"ops={len(sweep_ops)} n_iters={args.n_iters}", file=sys.stderr)

    from autoverse.measure import measure_op as _measure_op

    records: list[dict[str, Any]] = []
    for i, op in enumerate(sweep_ops, 1):
        timing = _measure_op(op, device=device,
                             n_warmup=args.n_warmup, n_iters=args.n_iters)
        rec = {**_op_to_record(op), **_timing_to_record(timing)}
        records.append(rec)
        if i % 10 == 0 or i == len(sweep_ops):
            print(f"[collect] {i}/{len(sweep_ops)}  "
                  f"last: {op.name} median={timing.median_ms:.4f}ms",
                  file=sys.stderr)

    payload: dict[str, Any] = {
        "device": device,
        "gpu_name": _gpu_name(device),
        "torch_version": torch.__version__,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_warmup": args.n_warmup,
        "n_iters": args.n_iters,
        "dtype": args.dtype,
        "quick": args.quick,
        "measurements": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[collect] wrote {len(records)} measurements to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
