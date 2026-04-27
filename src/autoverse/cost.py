"""Per-op analytical cost model.

Tier 0 (Day 1): pure roofline — the effective time of an op is
``max(flops / peak_compute, bytes / peak_bw)``, with an additive per-op launch
overhead. The ``max`` encodes **perfect compute-memory overlap within a single
op** — ALU and HBM run fully in parallel; the op finishes when the slower
phase finishes. This is a lower bound on real per-op latency; reality sits
above it because overlap is imperfect, nominal peaks are unachievable, and
small kernels under-utilise the machine.

Tier 2 (Day 3) refinement, **shipped**:

- **L2 hit-rate heuristic.** ``hit_rate = min(1, L2_capacity / working_set)``,
  applied to the bytes term: ``effective_bytes = bytes * (1 - hit_rate)``. Justified
  empirically: in our tight measurement loop (same input tensors reused across 100
  iters), inputs that fit in the 50 MB H100 L2 stay resident after iter 1, so per-iter
  HBM traffic shrinks. Without this, the calibrated B was forced 1.6× above vendor
  to absorb the L2 effect; with it, B drops back into physical range. See
  ``reports/02_tier2.md`` for before/after numbers.

Tier 2 refinements deferred (would help if MAPE remained the bottleneck):
  - wave quantisation for small GEMMs,
  - a calibratable overlap model ``t = max(t_c, t_m) + (1 - alpha) * min(t_c, t_m)``,
  - per-op-family overhead (currently a single global O).

See ``../03_autoverse_end_product.md`` §8 for the pinned modelling decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoverse.hardware import HardwareSpec
from autoverse.ops import Op


@dataclass(frozen=True)
class OpTiming:
    """Result of costing a single op.

    Attributes:
        compute_ms: Time if purely compute-bound (``flops / peak_compute``).
        memory_ms: Time if purely memory-bound (``effective_bytes / peak_bw``)
            after L2 hit-rate adjustment.
        effective_ms: Combined roofline output: ``max(t_c, t_m)`` plus per-op
            launch overhead.
        l2_hit_rate: Fraction of working-set bytes assumed to be served from L2
            (0 = cold, 1 = fully cached). 0 if ``l2_mb`` is None.
    """

    compute_ms: float
    memory_ms: float
    effective_ms: float
    l2_hit_rate: float = 0.0


def _peak_tflops_for_dtype(spec: HardwareSpec, dtype: str) -> float:
    """Pick the relevant peak compute throughput for the op's dtype.

    Tier 0 is coarse: BF16/FP16 use the BF16 tensor-core peak; FP32 uses the
    non-tensor-core peak; other dtypes (FP8, future precisions) fall back to
    the BF16 peak and get refined later.
    """
    if dtype in ("bf16", "fp16"):
        return spec.peak_bf16_tflops
    if dtype == "fp32":
        return spec.peak_fp32_tflops
    return spec.peak_bf16_tflops


def l2_hit_rate(input_bytes: int, l2_mb: float) -> float:
    """Tier-2 heuristic: ``min(1, L2_capacity / input_bytes)``.

    Heuristic-by-design: caches reduce *re-reads*, not first-time writes. We
    therefore apply the hit rate to ``bytes_read`` only — output writes
    always stream to HBM at full bandwidth. This is a one-line refinement
    over the naive "hit rate on working set" formulation, but materially
    improves accuracy on ops like SwiGLU whose output is a substantial
    fraction of total traffic.

    A linear blend between "all inputs in L2" (input_bytes ≤ L2_cap ⇒
    hit_rate=1) and "spilling out" (input_bytes ≫ L2_cap ⇒ hit_rate→0).
    Matches the asymptotic warm-cache regime of our measurement methodology:
    iter 1 of 100 populates L2; iters 2-100 hit it for inputs.

    Pass ``l2_mb=0`` (or negative) to disable the heuristic (returns 0).
    """
    if l2_mb <= 0 or input_bytes <= 0:
        return 0.0
    l2_bytes = l2_mb * 1024 * 1024
    return min(1.0, l2_bytes / input_bytes)


def estimate(op: Op, spec: HardwareSpec, *, use_l2: bool = True) -> OpTiming:
    """Estimate the effective latency of ``op`` on ``spec``.

    Roofline with launch overhead and the Tier-2 L2 hit-rate heuristic:

        t_c = flops / peak_compute
        hit  = min(1, L2_cap / bytes_read)                # 0 if use_l2=False
        eff_bytes = bytes_written + bytes_read · (1 - hit)
        t_m = eff_bytes / peak_bandwidth
        t_eff = max(t_c, t_m) + per_op_overhead

    Returns times in **milliseconds**. ``use_l2=False`` recovers the Tier-0
    behaviour (kept for ablations and for the synthetic-recovery calibration
    test).
    """
    peak_flops_per_s = _peak_tflops_for_dtype(spec, op.dtype) * 1e12
    peak_bytes_per_s = spec.hbm_gbps * 1e9
    bytes_read = op.bytes_read()
    bytes_written = op.bytes_written()

    hit = l2_hit_rate(bytes_read, spec.l2_mb) if use_l2 else 0.0
    effective_bytes = bytes_written + bytes_read * (1.0 - hit)

    compute_s = op.flops() / peak_flops_per_s if peak_flops_per_s > 0 else 0.0
    memory_s = effective_bytes / peak_bytes_per_s if peak_bytes_per_s > 0 else 0.0

    compute_ms = compute_s * 1e3
    memory_ms = memory_s * 1e3
    effective_ms = max(compute_ms, memory_ms) + spec.per_op_overhead_us / 1e3

    return OpTiming(
        compute_ms=compute_ms,
        memory_ms=memory_ms,
        effective_ms=effective_ms,
        l2_hit_rate=hit,
    )
