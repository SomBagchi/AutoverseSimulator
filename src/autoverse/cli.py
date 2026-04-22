"""Command-line interface for Autoverse.

Usage::

    python -m autoverse simulate --model llama1b --mode decode --seq-len 1024
    python -m autoverse simulate --model llama1b --mode prefill --seq-len 1024 --breakdown

At Tier 0 the output is an uncalibrated roofline estimate — useful for
ratios and scaling, not for absolute latency. Calibration lands at Tier 1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from autoverse.hardware import H100_SXM
from autoverse.model import LLAMA_1B, build_op_graph
from autoverse.simulator import SimResult, simulate

_MODELS = {"llama1b": LLAMA_1B}


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser. Separated for testability."""
    parser = argparse.ArgumentParser(prog="autoverse")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim_p = sub.add_parser("simulate", help="simulate a transformer forward pass")
    sim_p.add_argument("--model", default="llama1b", choices=sorted(_MODELS))
    sim_p.add_argument("--mode", default="decode", choices=["prefill", "decode"])
    sim_p.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="prefill: sequence length; decode: prior-KV-cache length.",
    )
    sim_p.add_argument(
        "--breakdown",
        action="store_true",
        help="print per-op-family latency breakdown (top contributors first).",
    )

    return parser


def _op_family(name: str) -> str:
    """Collapse per-layer op names ('q_proj_0', 'q_proj_1', ...) to a family ('q_proj')."""
    head, _, tail = name.rpartition("_")
    return head if head and tail.isdigit() else name


def _print_breakdown(result: SimResult) -> None:
    """Print per-op-family total ms, sorted descending. Uses ``effective_ms``."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for name, timing in result.per_op:
        family = _op_family(name)
        totals[family] += timing.effective_ms
        counts[family] += 1

    print("  per-op-family breakdown (effective_ms):")
    for family, ms in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * ms / result.total_ms if result.total_ms > 0 else 0.0
        print(f"    {family:<24s} {ms:10.3f} ms  ({pct:5.1f}%, {counts[family]} ops)")


def _print_header(result: SimResult, model: str, mode: str, seq_len: int) -> None:
    print(f"[autoverse] simulate: model={model} mode={mode} seq_len={seq_len}")
    print(f"  ops simulated: {len(result.per_op)}")
    print(f"  total latency: {result.total_ms:.3f} ms  (uncalibrated Tier-0 roofline)")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a shell exit code."""
    args = build_parser().parse_args(argv)

    if args.cmd == "simulate":
        cfg = _MODELS[args.model]
        ops = build_op_graph(cfg, args.seq_len, args.mode)
        result = simulate(ops, H100_SXM)

        _print_header(result, args.model, args.mode, args.seq_len)
        if args.breakdown:
            _print_breakdown(result)
        return 0

    return 1


__all__ = ["build_parser", "main"]
