"""What-if experiments — predict counterfactual hardware scenarios.

Reads the calibration fit (``reports/calibration_fit.json``) and a
Llama-1B op graph, then asks: what does end-to-end latency look like if we
twiddle one of HBM bandwidth, compute throughput, or L2 capacity? And how
does decode scale with context length?

The point is not perfect prediction — the calibrated model has MAPE 10.1 %,
so absolute numbers come with that error bar. The point is that **deltas**
under counterfactual changes are largely roofline-determined and survive
calibration error: doubling B halves memory time for memory-bound ops
regardless of whether the absolute numbers were 10 % off.

Usage::

    uv run python scripts/whatif.py --out reports/03_whatif.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from autoverse import LLAMA_1B
from autoverse.calibrate import predict_ms
from autoverse.hardware import H100_SXM, HardwareSpec
from autoverse.model import Mode, build_op_graph
from autoverse.ops import Op


def _baseline_spec_from_fit(fit_path: Path) -> tuple[HardwareSpec, dict[str, float]]:
    """Apply the calibration fit to the H100 baseline spec.

    Returns ``(calibrated_spec, overhead_by_family)``. Overhead-by-family is
    not stored on HardwareSpec; the what-if scenarios pass it as a sidecar
    when calling :func:`predict_ms`.
    """
    fit = json.loads(fit_path.read_text())["fit"]
    calibrated = replace(
        H100_SXM,
        peak_bf16_tflops=fit["fitted_peak_bf16_tflops"],
        hbm_gbps=fit["fitted_hbm_gbps"],
        per_op_overhead_us=fit["fitted_per_op_overhead_us"],
    )
    overhead_by_family: dict[str, float] = fit.get("fitted_overhead_by_family") or {}
    return calibrated, overhead_by_family


def _total_ms(
    ops: list[Op], spec: HardwareSpec, overhead_by_family: dict[str, float]
) -> float:
    """Sum predict_ms across an op graph with the calibrated settings (L2 + per-family)."""
    # n_sm=0 disables wave quant. The committed fit was produced with wave
    # quant off (it strictly worsened MAPE — see reports/02_refinements.md);
    # using it here too keeps the prediction formula matching the fit.
    return sum(
        predict_ms(
            op,
            spec.peak_bf16_tflops,
            spec.hbm_gbps,
            spec.per_op_overhead_us,
            l2_mb=spec.l2_mb,
            overhead_by_family=overhead_by_family,
            n_sm=0,
        )
        for op in ops
    )


def _decompose_ms(
    ops: list[Op], spec: HardwareSpec, overhead_by_family: dict[str, float]
) -> dict[str, float]:
    """Bucket total predicted latency by op family. Sums to _total_ms."""
    out: dict[str, float] = {}
    for op in ops:
        t = predict_ms(
            op, spec.peak_bf16_tflops, spec.hbm_gbps,
            spec.per_op_overhead_us, l2_mb=spec.l2_mb,
            overhead_by_family=overhead_by_family,
            n_sm=0,
        )
        out[type(op).__name__] = out.get(type(op).__name__, 0.0) + t
    return out


# ---------- Experiments ----------


def whatif_hbm_doubled(
    spec: HardwareSpec, oh: dict[str, float],
    seq_len: int = 1024, mode: Mode = "decode",
) -> dict[str, float]:
    """E1: double HBM bandwidth. Memory-bound ops shrink ~2×."""
    ops = build_op_graph(LLAMA_1B, seq_len=seq_len, mode=mode)
    base = _total_ms(ops, spec, oh)
    counter = _total_ms(ops, replace(spec, hbm_gbps=spec.hbm_gbps * 2), oh)
    return {"baseline_ms": base, "counterfactual_ms": counter,
            "delta_ms": counter - base, "speedup_x": base / counter}


def whatif_compute_doubled(
    spec: HardwareSpec, oh: dict[str, float],
    seq_len: int = 1024, mode: Mode = "prefill",
) -> dict[str, float]:
    """E2: double tensor-core throughput. Compute-bound ops shrink ~2×."""
    ops = build_op_graph(LLAMA_1B, seq_len=seq_len, mode=mode)
    base = _total_ms(ops, spec, oh)
    counter = _total_ms(ops, replace(spec, peak_bf16_tflops=spec.peak_bf16_tflops * 2), oh)
    return {"baseline_ms": base, "counterfactual_ms": counter,
            "delta_ms": counter - base, "speedup_x": base / counter}


def whatif_l2_doubled(
    spec: HardwareSpec, oh: dict[str, float],
    seq_len: int = 1024, mode: Mode = "decode",
) -> dict[str, float]:
    """E3: double L2 capacity (50 → 100 MB). More ops fit ⇒ lower effective bytes."""
    ops = build_op_graph(LLAMA_1B, seq_len=seq_len, mode=mode)
    base = _total_ms(ops, spec, oh)
    counter = _total_ms(ops, replace(spec, l2_mb=spec.l2_mb * 2), oh)
    return {"baseline_ms": base, "counterfactual_ms": counter,
            "delta_ms": counter - base, "speedup_x": base / counter}


def sweep_decode_vs_context(
    spec: HardwareSpec, oh: dict[str, float],
    ctx_lens: tuple[int, ...] = (128, 512, 1024, 2048, 4096, 8192, 16384, 32768),
) -> list[dict[str, float]]:
    """E4: decode latency vs context length. Eventually KV-cache reads dominate."""
    out: list[dict[str, float]] = []
    for cl in ctx_lens:
        ops = build_op_graph(LLAMA_1B, seq_len=cl, mode="decode")
        total = _total_ms(ops, spec, oh)
        breakdown = _decompose_ms(ops, spec, oh)
        out.append({
            "ctx_len": cl,
            "total_ms": total,
            "attn_ms": breakdown.get("AttentionDecode", 0.0),
            "matmul_ms": breakdown.get("MatMul", 0.0),
            "attn_share": breakdown.get("AttentionDecode", 0.0) / total,
        })
    return out


def prefill_vs_decode_per_token(
    spec: HardwareSpec, oh: dict[str, float],
    prefill_seq_len: int = 1024,
) -> dict[str, float]:
    """E5: tokens/sec advantage of prefill over single-token decode."""
    prefill_ops = build_op_graph(LLAMA_1B, seq_len=prefill_seq_len, mode="prefill")
    prefill_total = _total_ms(prefill_ops, spec, oh)
    prefill_per_token = prefill_total / prefill_seq_len

    decode_ops = build_op_graph(LLAMA_1B, seq_len=prefill_seq_len, mode="decode")
    decode_per_token = _total_ms(decode_ops, spec, oh)

    return {
        "prefill_total_ms": prefill_total,
        "prefill_per_token_ms": prefill_per_token,
        "decode_per_token_ms": decode_per_token,
        "prefill_advantage_x": decode_per_token / prefill_per_token,
        "tokens_per_sec_prefill": 1000.0 / prefill_per_token,
        "tokens_per_sec_decode": 1000.0 / decode_per_token,
    }


# ---------- Markdown report ----------


def _md(fit_path: Path) -> str:
    spec, oh = _baseline_spec_from_fit(fit_path)

    e1 = whatif_hbm_doubled(spec, oh, seq_len=1024, mode="decode")
    e2 = whatif_compute_doubled(spec, oh, seq_len=1024, mode="prefill")
    e3 = whatif_l2_doubled(spec, oh, seq_len=1024, mode="decode")
    e4 = sweep_decode_vs_context(spec, oh)
    e5 = prefill_vs_decode_per_token(spec, oh)

    # Decompose decode total into overhead vs (compute+memory).
    decode_ops = build_op_graph(LLAMA_1B, seq_len=1024, mode="decode")
    decode_overhead_ms = sum(
        oh.get(type(op).__name__, spec.per_op_overhead_us) * 1e-3 for op in decode_ops
    )
    decode_total_ms = e1["baseline_ms"]
    decode_overhead_share = decode_overhead_ms / decode_total_ms

    lines: list[str] = [
        "# What-if experiments — counterfactual hardware on the calibrated model",
        "",
        "> Five counterfactual hardware questions answered using the calibrated",
        "> H100 model from `reports/calibration_fit.json`. Baseline is the",
        "> calibrated H100; each experiment changes one parameter and reports",
        "> the delta.",
        "",
        f"Baseline calibrated `HardwareSpec`: "
        f"`F={spec.peak_bf16_tflops:.0f} TFLOPs`, "
        f"`B={spec.hbm_gbps:.0f} GB/s`, "
        f"`L2={spec.l2_mb} MB`, "
        f"per-family overhead from the calibration fit.",
        "",
        "## Headline finding: Llama-1B decode is overhead-bound, not memory-bound",
        "",
        f"Decoding one token through 227 ops on the calibrated H100 model "
        f"predicts **{decode_total_ms:.2f} ms**, of which "
        f"**{decode_overhead_ms:.2f} ms ({decode_overhead_share*100:.0f}%)** is "
        f"per-op kernel-launch overhead. Compute + HBM together account for the "
        f"remaining ~{(1-decode_overhead_share)*100:.0f}%. This is the dominant "
        f"surprise of the what-if work and reframes most of the experiments "
        f"below: hardware throughput knobs (HBM, compute, L2) move the small "
        f"non-overhead slice; the big lever for small-model single-stream decode "
        f"is collapsing the launch count itself (CUDA Graphs, kernel fusion, "
        f"larger batches).",
        "",
        "## E1 — What if HBM bandwidth doubled? (decode, ctx=1024)",
        "",
        "| | latency | comment |",
        "|---|---|---|",
        f"| Baseline (B = {spec.hbm_gbps:.0f} GB/s) |"
        f" **{e1['baseline_ms']:.3f} ms** | Calibrated H100. |",
        f"| Counterfactual (B × 2) |"
        f" {e1['counterfactual_ms']:.3f} ms |"
        f" Speedup **{e1['speedup_x']:.2f} ×** "
        f"(saves {(-e1['delta_ms'])*1000:.0f} µs/token). |",
        "",
        "**Reading.** I expected ≈ 2 × — \"decode is memory-bound, double the "
        "bandwidth, halve the time.\" The model says otherwise. With 95 % of "
        "total time in launch overhead, doubling HBM only acts on the 5 % "
        "memory slice. **For Llama-1B decode on H100, an HBM upgrade barely "
        "moves the needle.** The roofline still holds — within memory-bound ops "
        "the time roughly halves — but those ops are not what you're paying for.",
        "",
        "Implication: large-model decode (where weight-read traffic per layer is "
        "much larger and the per-op compute/memory time grows past the launch "
        "overhead) **would** see the expected ≈ 2 × from this experiment. "
        "Llama-1B is small enough that it lives in a different regime.",
        "",
        "## E2 — What if BF16 compute doubled? (prefill, seq_len=1024)",
        "",
        "| | latency | comment |",
        "|---|---|---|",
        f"| Baseline (F = {spec.peak_bf16_tflops:.0f} TFLOPs) |"
        f" **{e2['baseline_ms']:.3f} ms** | Calibrated H100. |",
        f"| Counterfactual (F × 2) |"
        f" {e2['counterfactual_ms']:.3f} ms |"
        f" Speedup **{e2['speedup_x']:.2f} ×**. |",
        "",
        "**Reading.** Prefill at seq_len=1024 *is* substantially compute-bound. "
        "The 1.18 × falls short of 2 × because ~4.4 ms of the 6.9 ms baseline "
        "is the same launch-overhead floor as decode (227 ops, ~14–52 µs each). "
        "Halving compute time on the remaining ~2.5 ms gets you the 1.18 × seen. "
        "**This is the same story as E1 from the opposite direction**: per-op "
        "overhead caps the upside of any single throughput knob.",
        "",
        "## E3 — What if L2 doubled? (decode, ctx=1024)",
        "",
        "| | latency | comment |",
        "|---|---|---|",
        f"| Baseline (L2 = {spec.l2_mb} MB) |"
        f" **{e3['baseline_ms']:.3f} ms** | Calibrated H100. |",
        f"| Counterfactual (L2 × 2 = {spec.l2_mb*2} MB) |"
        f" {e3['counterfactual_ms']:.3f} ms |"
        f" Speedup **{e3['speedup_x']:.2f} ×**. |",
        "",
        "**Reading.** Effectively no help. Decode-shape MLP matmuls "
        "(`m=1`, `k=2048`, `n=8192`) already fit in 50 MB L2 (working set ~33 MB), "
        "so doubling L2 doesn't change their hit-rate. The LM head (`n=128 256`, "
        "525 MB working set) is way too big for any plausible L2; doubling 50→100 "
        "shifts hit_rate from 9.5 % to 19 %, saving a few µs. **Llama-1B decode "
        "L2 is already \"big enough\" by a comfortable margin** — the 50 MB H100 "
        "L2 was a substantial uplift over A100's 40 MB and pays off here.",
        "",
        "## E4 — Decode latency vs context length (sweep)",
        "",
        "| ctx_len | total ms | attn ms | matmul ms | attn share |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in e4:
        lines.append(
            f"| {row['ctx_len']:>7} | {row['total_ms']:.4f} |"
            f" {row['attn_ms']:.4f} | {row['matmul_ms']:.4f} |"
            f" {row['attn_share']*100:5.1f} % |"
        )
    lines.extend([
        "",
        "**Reading.** Llama-1B's 8 KV heads × 64 d_head = 512 floats per token "
        "of KV cache, BF16 = 1024 bytes/token/layer. Even at ctx=32 768 the per-"
        "layer KV cache is 32 MB — fits comfortably in L2. Combined with the "
        "fact that AttentionDecode has a single-digit-µs roofline contribution "
        "(memory-bound but small) plus its 22 µs per-family launch overhead, the "
        "attention share stays ≤ 10 % of total decode latency at every context "
        "length we tested.",
        "",
        "**Generalisation watch-out.** This is an artefact of Llama-1B's *very* "
        "aggressive GQA (4 Q-heads per KV-head). Llama-3.1-70B with 8 KV-heads "
        "× 128 d_head sees 4 × the KV-cache size per token, and a `d_model=8192` "
        "MLP that's bigger but not 4 × larger — the crossover where attention "
        "dominates moves much closer in.",
        "",
        f"## E5 — Prefill vs decode per-token (prefill seq_len={1024})",
        "",
        "| | per-token latency | tokens/sec |",
        "|---|---:|---:|",
        f"| Prefill (batch all {1024} tokens at once) |"
        f" **{e5['prefill_per_token_ms']*1000:.1f} µs** |"
        f" {e5['tokens_per_sec_prefill']:>8,.0f} |",
        f"| Decode (one token at a time) |"
        f" **{e5['decode_per_token_ms']*1000:.0f} µs** |"
        f" {e5['tokens_per_sec_decode']:>8,.0f} |",
        f"| **Prefill advantage** |"
        f" **{e5['prefill_advantage_x']:.0f} ×** faster per token | |",
        "",
        "**Reading.** Decode pays the full 227-op launch overhead *per token*. "
        "Prefill pays it once for the whole batch and then amortises across all "
        f"{1024} tokens — the per-op overhead per token drops by ~{1024}×. Plus, "
        "prefill MatMuls are big enough to actually engage the compute roof "
        "instead of bottoming out at the launch floor. Result: a "
        f"~{e5['prefill_advantage_x']:.0f} × per-token latency advantage. "
        "**This number is the strongest argument for continuous batching.** "
        "Every additional decode step you can convert to a prefill-shaped step "
        "(by stacking concurrent users) gets close to that advantage.",
        "",
        "## What to take from these five",
        "",
        "1. **Per-op launch overhead is the dominant Llama-1B-on-H100 single-",
        "   stream-decode bottleneck.** Not HBM, not L2, not compute. Engineering "
        "   hours on this model class go furthest into kernel fusion, CUDA Graphs, "
        "   or batching — anything that collapses the 227-launch count.",
        "2. **Prefill is on a different shore of the roofline** — partially "
        "   compute-bound, where doubling compute does buy you 1.2 ×. Compute "
        "   upgrades pay off in prefill time-to-first-token (TTFT), much less "
        "   in inter-token latency.",
        "3. **L2 is already over-provisioned for Llama-1B** at H100's 50 MB. "
        "   This is a real architectural feature of the H100 generation paying "
        "   off; doubling it again wouldn't.",
        "4. **Context-length scaling is benign for Llama-1B** because of GQA. "
        "   This conclusion does *not* generalise to bigger Llama variants — "
        "   the simulator can answer that question once those configs are added "
        "   to `model.py`.",
        "5. **Continuous batching is the headline serving lever.** The per-token "
        "   ratio between batched-prefill and isolated-decode is the upper bound "
        "   on what fancy batching can buy you.",
        "",
        "## Caveats",
        "",
        "- The calibrated model has held-out MAPE 10.1 %. Absolute numbers "
        "carry that error bar.",
        "- **Speedup ratios** in E1–E3 are mostly roofline-determined and "
        "survive the absolute calibration error — if 5 % of time is memory and "
        "you halve memory, you save 2.5 % regardless of whether your time was "
        "9 % off.",
        "- The model has no concept of multi-GPU collectives, paged attention, "
        "speculative decoding, or batched continuous serving — explicitly out "
        "of scope per `03_autoverse_end_product.md` §2.",
        "- Per-op overhead is fitted from a measurement methodology that runs "
        "the same op 100 × in a tight loop. CUDA Graphs / kernel-launch APIs "
        "in production may produce lower effective overhead than this fit "
        "captures; that would shrink the overhead floor and shift these "
        "conclusions toward more conventional roofline regimes.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make whatif        # regenerates this report from the calibration fit",
        "```",
        "",
        "Or for a one-off: `uv run python scripts/whatif.py --out -` prints to stdout.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, default=Path("reports/calibration_fit.json"),
                   help="Calibrated fit JSON (output of `make calibrate`).")
    p.add_argument("--out", type=Path, default=Path("reports/03_whatif.md"),
                   help="Markdown report path. Pass '-' to print to stdout.")
    args = p.parse_args()

    if not args.fit.exists():
        print(f"[whatif] missing {args.fit} — run `make calibrate` first", file=sys.stderr)
        return 2

    md = _md(args.fit)
    if str(args.out) == "-":
        sys.stdout.write(md)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"[whatif] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
