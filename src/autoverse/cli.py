"""Command-line interface for Autoverse.

Usage (Tier 0+):
    python -m autoverse simulate --model llama1b --mode decode --seq-len 1024

Subcommands for later tiers (validate, whatif) will be added at Tier 1 / Tier 3.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser. Separated for testability."""
    parser = argparse.ArgumentParser(prog="autoverse")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim_p = sub.add_parser("simulate", help="simulate a transformer forward pass")
    sim_p.add_argument("--model", default="llama1b", choices=["llama1b"])
    sim_p.add_argument("--mode", default="decode", choices=["prefill", "decode"])
    sim_p.add_argument("--seq-len", type=int, default=1024)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a shell exit code."""
    args = build_parser().parse_args(argv)

    if args.cmd == "simulate":
        print(
            f"[autoverse] simulate: model={args.model} "
            f"mode={args.mode} seq_len={args.seq_len}"
        )
        print("  (Tier 0 not yet implemented — see CLAUDE.md Day 1 checkpoint 1E)")
        return 0

    return 1
