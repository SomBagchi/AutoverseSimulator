"""Fit :class:`HardwareSpec` parameters to real measurements.

Given a set of ``(op, measured_median_ms)`` pairs, we fit the parameters the
roofline cost model exposes:

- ``F`` — effective BF16 tensor-core throughput, in **TFLOPs/s**.
- ``B`` — effective HBM bandwidth, in **GB/s**.
- one ``O`` per op family — launch overhead, in **microseconds**.

Predicted latency for op ``i`` is the same formula as in :mod:`autoverse.cost`::

    p_i = max(flops_i / (F · 1e12), eff_bytes_i / (B · 1e9)) · 1e3 + O_family · 1e-3      # ms

with ``eff_bytes`` carrying the L2 hit-rate adjustment.

We minimise log-space residuals via SciPy ``least_squares`` (Trust-Region
Reflective, with positivity bounds). Log-space is the natural loss because
latencies span 3+ decades (≈ 10 µs to ≈ 10 ms) and we care about relative
error (MAPE), not absolute ms.

The fit/held-out split defaults to a deterministic 70/30 shuffle with a seed,
so the report numbers are reproducible across runs.

Three calibration entry points:

- :func:`calibrate` — joint fit of ``(F, B, single global O)``. The bare
  baseline; kept for ablations.
- :func:`calibrate_per_family` — joint fit of ``(F, B, per-family O)``.
- :func:`calibrate_two_stage` — fit F on compute-bound subset only, then
  freeze it and fit B + per-family O on the full dataset. **The default**;
  required to keep F in physically meaningful ranges. See
  ``reports/01_methodology.md`` for the derivation.

Calibration is **dtype-scoped**: for a BF16 baseline, we filter measurements
to BF16 ops. Mixed-precision calibration would need a per-dtype peak.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from autoverse.hardware import HardwareSpec
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

# Default initial guesses and bounds for (F_tflops, B_gbps, O_us).
# Starting guesses are H100-SXM vendor nominals; bounds keep the optimiser
# inside physically plausible territory (no negative throughput / overhead,
# and not more than ~4x vendor nominal).
_X0 = (989.0, 3350.0, 1.0)
_LB = (1e-3, 1e-3, 0.0)
_UB = (4000.0, 12000.0, 1000.0)


@dataclass(frozen=True)
class CalibrationResult:
    """Output of a calibration run.

    Attributes:
        fitted_peak_bf16_tflops: Effective BF16 peak, post-fit.
        fitted_hbm_gbps: Effective HBM bandwidth, post-fit.
        fitted_per_op_overhead_us: Per-op overhead constant, post-fit.
            For per-family fits, this is the un-weighted mean of the
            per-family overheads (a summary number, not what's used at
            prediction time).
        fitted_overhead_by_family: Per-op-family overhead in microseconds.
            Empty dict for the global-overhead fit; populated by the per-family
            variant. The prediction path uses this when non-empty.
        mape_fit: Mean absolute percentage error on the fitted subset.
        mape_held_out: MAPE on the held-out subset. ``None`` if not split.
        n_fit: Number of measurements used for fitting.
        n_held_out: Number of measurements held out.
        per_op_mape_fit: MAPE broken down by op-type name on the fit set.
        per_op_mape_held_out: Same for held-out.
        residual_cost: Final least_squares cost (half the sum of squared log-residuals).
    """

    fitted_peak_bf16_tflops: float
    fitted_hbm_gbps: float
    fitted_per_op_overhead_us: float
    mape_fit: float
    mape_held_out: float | None
    n_fit: int
    n_held_out: int
    per_op_mape_fit: dict[str, float] = field(default_factory=dict)
    per_op_mape_held_out: dict[str, float] = field(default_factory=dict)
    residual_cost: float = 0.0
    fitted_overhead_by_family: dict[str, float] = field(default_factory=dict)


def predict_ms(
    op: Op,
    peak_tflops: float,
    hbm_gbps: float,
    overhead_us: float,
    l2_mb: float = 0.0,
    overhead_by_family: dict[str, float] | None = None,
    n_sm: int = 0,
) -> float:
    """Roofline prediction with optional L2 heuristic, per-family overhead,
    and wave quantisation (MatMul only). In **ms**.

    Kept as a free function (not a method of HardwareSpec) so calibration can
    call it in a hot loop without constructing frozen dataclasses per iter.

    - ``l2_mb=0`` (default) disables the L2 hit-rate heuristic.
      Set to ``spec.l2_mb`` (50 for H100) to enable.
    - ``overhead_by_family``: when given, looks up a per-family overhead by
      ``type(op).__name__``; falls back to ``overhead_us`` for unknown families.
      ``None`` uses ``overhead_us`` everywhere.
    - ``n_sm=0`` (default) disables wave quantisation. Set to ``spec.n_sm``
      (132 for H100) to enable. Only affects MatMul ops.
    """
    flops = op.flops()
    if n_sm > 0 and isinstance(op, MatMul):
        # Mirror cost.wave_quant_factor inline to avoid the import cycle.
        tiles_m = -(-op.m // 128)
        tiles_n = -(-op.n // 128)
        tiles = tiles_m * tiles_n
        if tiles > 0:
            waves = -(-tiles // n_sm)
            flops = int(flops * (waves * n_sm) / tiles)
    compute_s = flops / (peak_tflops * 1e12) if peak_tflops > 0 else 0.0
    bytes_read = op.bytes_read()
    bytes_written = op.bytes_written()
    if l2_mb > 0 and bytes_read > 0:
        hit = min(1.0, (l2_mb * 1024 * 1024) / bytes_read)
        effective_bytes = bytes_written + bytes_read * (1.0 - hit)
    else:
        effective_bytes = bytes_read + bytes_written
    memory_s = effective_bytes / (hbm_gbps * 1e9) if hbm_gbps > 0 else 0.0
    if overhead_by_family:
        ov_us = overhead_by_family.get(type(op).__name__, overhead_us)
    else:
        ov_us = overhead_us
    return max(compute_s, memory_s) * 1e3 + ov_us * 1e-3


def _residuals(
    params: np.ndarray,
    ops: list[Op],
    measured_ms: np.ndarray,
    l2_mb: float,
    n_sm: int,
    eps: float = 1e-9,
) -> np.ndarray:
    """Log-latency residuals: ``log(predicted) - log(measured)``."""
    peak_tflops, hbm_gbps, overhead_us = params
    pred = np.asarray(
        [predict_ms(op, peak_tflops, hbm_gbps, overhead_us, l2_mb, n_sm=n_sm)
         for op in ops],
        dtype=np.float64,
    )
    # Clip to avoid log(0) in pathological cases (shouldn't trigger with O >= 0).
    pred = np.maximum(pred, eps)
    m = np.maximum(measured_ms, eps)
    out: np.ndarray = np.log(pred) - np.log(m)
    return out


def _mape(preds: np.ndarray, measured: np.ndarray) -> float:
    """Mean absolute percentage error, as a fraction (0.22 = 22%)."""
    return float(np.mean(np.abs(preds - measured) / np.maximum(measured, 1e-9)))


def _mape_by_op_type(
    ops: list[Op], preds: np.ndarray, measured: np.ndarray
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for op, p, m in zip(ops, preds, measured, strict=True):
        err = abs(p - m) / max(m, 1e-9)
        buckets.setdefault(type(op).__name__, []).append(err)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def split_fit_held_out(
    ops: list[Op], measured_ms: list[float], *, fit_frac: float = 0.7, seed: int = 0
) -> tuple[list[Op], list[float], list[Op], list[float]]:
    """Deterministic shuffled split of (ops, measured_ms) into fit / held-out."""
    rng = random.Random(seed)
    idx = list(range(len(ops)))
    rng.shuffle(idx)
    n_fit = int(round(fit_frac * len(ops)))
    fit_idx, ho_idx = idx[:n_fit], idx[n_fit:]
    return (
        [ops[i] for i in fit_idx],
        [measured_ms[i] for i in fit_idx],
        [ops[i] for i in ho_idx],
        [measured_ms[i] for i in ho_idx],
    )


def calibrate(
    ops: list[Op],
    measured_ms: list[float],
    *,
    x0: tuple[float, float, float] = _X0,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (_LB, _UB),
    fit_frac: float = 0.7,
    seed: int = 0,
    verbose: bool = False,
    l2_mb: float = 0.0,
    n_sm: int = 0,
) -> CalibrationResult:
    """Fit (F, B, O) to match measured latencies on ``ops``.

    Arguments:
        ops, measured_ms: parallel lists, same length.
        x0: initial (F_tflops, B_gbps, O_us). Defaults to H100 vendor nominals.
        bounds: ((lb), (ub)) for the three scalars.
        fit_frac: fraction of measurements used for fitting; remainder held out.
        seed: RNG seed for the fit/held-out split.
        verbose: pass through to scipy.
        l2_mb: L2 cache capacity in MB used by the L2 hit-rate heuristic.
            ``0`` (default) disables; ``50`` = H100.
        n_sm: SM count used by the wave-quantisation heuristic on MatMul ops.
            ``0`` (default) disables; ``132`` = H100.
    """
    if len(ops) != len(measured_ms):
        raise ValueError(f"ops/measured_ms length mismatch: {len(ops)} vs {len(measured_ms)}")
    if len(ops) < 4:
        raise ValueError(f"need >=4 measurements to fit 3 params; got {len(ops)}")

    fit_ops, fit_ms, ho_ops, ho_ms = split_fit_held_out(ops, measured_ms,
                                                        fit_frac=fit_frac, seed=seed)
    fit_ms_arr = np.asarray(fit_ms, dtype=np.float64)
    ho_ms_arr = np.asarray(ho_ms, dtype=np.float64)

    result = least_squares(
        _residuals,
        x0=np.asarray(x0, dtype=np.float64),
        args=(fit_ops, fit_ms_arr, l2_mb, n_sm),
        bounds=bounds,
        method="trf",
        verbose=2 if verbose else 0,
    )

    peak_tflops, hbm_gbps, overhead_us = result.x
    fit_preds = np.asarray(
        [predict_ms(op, peak_tflops, hbm_gbps, overhead_us, l2_mb, n_sm=n_sm)
         for op in fit_ops]
    )
    ho_preds = (
        np.asarray([predict_ms(op, peak_tflops, hbm_gbps, overhead_us, l2_mb,
                                n_sm=n_sm)
                    for op in ho_ops])
        if ho_ops else None
    )

    mape_fit_val = _mape(fit_preds, fit_ms_arr)
    mape_ho_val = _mape(ho_preds, ho_ms_arr) if ho_preds is not None and len(ho_preds) else None

    return CalibrationResult(
        fitted_peak_bf16_tflops=float(peak_tflops),
        fitted_hbm_gbps=float(hbm_gbps),
        fitted_per_op_overhead_us=float(overhead_us),
        mape_fit=mape_fit_val,
        mape_held_out=mape_ho_val,
        n_fit=len(fit_ops),
        n_held_out=len(ho_ops),
        per_op_mape_fit=_mape_by_op_type(fit_ops, fit_preds, fit_ms_arr),
        per_op_mape_held_out=(
            _mape_by_op_type(ho_ops, ho_preds, ho_ms_arr)
            if ho_preds is not None and len(ho_preds) else {}
        ),
        residual_cost=float(result.cost),
    )


def _residuals_per_family(
    params: np.ndarray,
    ops: list[Op],
    measured_ms: np.ndarray,
    families: list[str],
    l2_mb: float,
    n_sm: int,
    eps: float = 1e-9,
) -> np.ndarray:
    """Like _residuals, but with one overhead scalar per op family.

    ``params`` layout: ``[F, B, O_family_0, O_family_1, ...]`` in the same
    order as ``families``.
    """
    peak_tflops, hbm_gbps = params[0], params[1]
    family_index = {f: i for i, f in enumerate(families)}
    pred_list: list[float] = []
    for op in ops:
        fam = type(op).__name__
        ov_us = float(params[2 + family_index[fam]]) if fam in family_index else 0.0
        pred_list.append(predict_ms(op, peak_tflops, hbm_gbps, ov_us, l2_mb,
                                     n_sm=n_sm))
    pred = np.maximum(np.asarray(pred_list, dtype=np.float64), eps)
    m = np.maximum(measured_ms, eps)
    out: np.ndarray = np.log(pred) - np.log(m)
    return out


def calibrate_per_family(
    ops: list[Op],
    measured_ms: list[float],
    *,
    x0_FB: tuple[float, float] = (989.0, 3350.0),
    x0_overhead_us: float = 5.0,
    bound_overhead_us: tuple[float, float] = (0.0, 200.0),
    bound_F: tuple[float, float] = (1e-3, 4000.0),
    bound_B: tuple[float, float] = (1e-3, 12000.0),
    fit_frac: float = 0.7,
    seed: int = 0,
    l2_mb: float = 0.0,
    n_sm: int = 0,
) -> CalibrationResult:
    """Fit (F, B, O_per_family) — one launch overhead per op family.

    Higher-leverage than fitting a single global ``O`` when the dataset has a
    mix of overhead-dominated op families: a single global ``O`` is pulled
    high by RoPE-like ops with real ~50 µs launch costs, hurting the fit on
    genuinely-cheap residual / RMSNorm ops at ~12 µs.

    Number of free params = 2 + N_families (typically 10 for a Llama-1B sweep).
    Still well-determined with ~500 measurements.
    """
    if len(ops) != len(measured_ms):
        raise ValueError(f"ops/measured_ms length mismatch: {len(ops)} vs {len(measured_ms)}")
    if len(ops) < 4:
        raise ValueError(f"need >=4 measurements; got {len(ops)}")

    fit_ops, fit_ms, ho_ops, ho_ms = split_fit_held_out(ops, measured_ms,
                                                        fit_frac=fit_frac, seed=seed)
    families = sorted({type(op).__name__ for op in fit_ops})
    n_fam = len(families)

    x0 = np.array([x0_FB[0], x0_FB[1], *([x0_overhead_us] * n_fam)], dtype=np.float64)
    lb = np.array([bound_F[0], bound_B[0], *([bound_overhead_us[0]] * n_fam)])
    ub = np.array([bound_F[1], bound_B[1], *([bound_overhead_us[1]] * n_fam)])

    fit_ms_arr = np.asarray(fit_ms, dtype=np.float64)
    ho_ms_arr = np.asarray(ho_ms, dtype=np.float64)

    result = least_squares(
        _residuals_per_family,
        x0=x0,
        args=(fit_ops, fit_ms_arr, families, l2_mb, n_sm),
        bounds=(lb, ub),
        method="trf",
    )

    F = float(result.x[0])
    B = float(result.x[1])
    overhead_by_family = {fam: float(result.x[2 + i]) for i, fam in enumerate(families)}

    def _pred_ms(op: Op) -> float:
        return predict_ms(op, F, B, x0_overhead_us, l2_mb, overhead_by_family,
                          n_sm=n_sm)

    fit_preds = np.asarray([_pred_ms(op) for op in fit_ops])
    ho_preds = np.asarray([_pred_ms(op) for op in ho_ops]) if ho_ops else None

    mape_fit_val = _mape(fit_preds, fit_ms_arr)
    mape_ho_val = _mape(ho_preds, ho_ms_arr) if ho_preds is not None and len(ho_preds) else None

    return CalibrationResult(
        fitted_peak_bf16_tflops=F,
        fitted_hbm_gbps=B,
        fitted_per_op_overhead_us=float(np.mean(list(overhead_by_family.values()))),
        fitted_overhead_by_family=overhead_by_family,
        mape_fit=mape_fit_val,
        mape_held_out=mape_ho_val,
        n_fit=len(fit_ops),
        n_held_out=len(ho_ops),
        per_op_mape_fit=_mape_by_op_type(fit_ops, fit_preds, fit_ms_arr),
        per_op_mape_held_out=(
            _mape_by_op_type(ho_ops, ho_preds, ho_ms_arr)
            if ho_preds is not None and len(ho_preds) else {}
        ),
        residual_cost=float(result.cost),
    )


def calibrate_two_stage(
    ops: list[Op],
    measured_ms: list[float],
    *,
    ridge_flops_per_byte: float = 295.0,
    l2_mb: float = 0.0,
    n_sm: int = 0,
    seed: int = 0,
    fit_frac: float = 0.7,
    x0_overhead_us: float = 5.0,
    bound_overhead_us: tuple[float, float] = (0.0, 200.0),
    bound_F: tuple[float, float] = (1e-3, 4000.0),
    bound_B: tuple[float, float] = (1e-3, 12000.0),
) -> CalibrationResult:
    """Two-stage calibration: F from compute-bound ops; B + per-family O from all.

    Why this exists: in the joint per-family fit, memory-bound ops put weak
    gradient on F (their predicted time barely depends on F — it's bottlenecked
    by ``bytes/B`` or by per-op overhead). With ~75% of the Llama-1B sweep
    being memory- or overhead-bound, the optimiser drifts F toward unphysical
    values (e.g. 1172 TFLOPs vs vendor 989) without strong counter-pressure.

    The fix splits responsibility between two fits:

    - **Stage 1 — compute-bound subset only.** Filter ops by arithmetic
      intensity ``flops / bytes > ridge_flops_per_byte`` (default 295,
      H100 vendor ridge). On this subset, ``t_compute`` dominates ``t_memory``,
      so the fit on F is well-conditioned. We fit ``(F, B, per-family O)`` and
      keep only F.
    - **Stage 2 — full dataset, F frozen at the stage-1 value.** Fit B and
      per-family overhead. With F fixed at a physically-sensible value,
      memory-bound ops are no longer at war with compute-bound ops over F,
      and each scalar settles where its data actually constrains it.

    Returns a :class:`CalibrationResult` with F from stage 1, everything else
    from stage 2. ``mape_fit`` / ``mape_held_out`` are reported on the stage-2
    full-dataset fit (same denominator as the joint per-family fit, so the
    headline MAPE comparison is apples-to-apples).
    """
    if len(ops) != len(measured_ms):
        raise ValueError(f"ops/measured_ms length mismatch: {len(ops)} vs {len(measured_ms)}")
    if len(ops) < 8:
        raise ValueError(f"need >=8 measurements for two-stage fit; got {len(ops)}")

    cb_ops: list[Op] = []
    cb_ms: list[float] = []
    for op, t in zip(ops, measured_ms, strict=True):
        bytes_total = op.bytes_read() + op.bytes_written()
        if bytes_total <= 0:
            continue
        if op.flops() / bytes_total > ridge_flops_per_byte:
            cb_ops.append(op)
            cb_ms.append(t)

    if len(cb_ops) < 4:
        raise ValueError(
            f"need >=4 compute-bound ops (AI > {ridge_flops_per_byte}) for stage 1; "
            f"got {len(cb_ops)}. Lower ridge_flops_per_byte or add bigger GEMMs."
        )

    # Stage 1: compute-bound subset, fit F (and aux per-family overheads).
    r1 = calibrate_per_family(
        cb_ops, cb_ms,
        l2_mb=l2_mb, n_sm=n_sm, seed=seed, fit_frac=fit_frac,
        x0_overhead_us=x0_overhead_us, bound_overhead_us=bound_overhead_us,
        bound_F=bound_F, bound_B=bound_B,
    )
    F_stage1 = r1.fitted_peak_bf16_tflops

    # Stage 2: full dataset, F effectively frozen via tight bounds.
    eps_rel = 1e-6
    r2 = calibrate_per_family(
        ops, measured_ms,
        x0_FB=(F_stage1, r1.fitted_hbm_gbps),
        bound_F=(F_stage1 * (1 - eps_rel), F_stage1 * (1 + eps_rel)),
        bound_B=bound_B,
        l2_mb=l2_mb, n_sm=n_sm, seed=seed, fit_frac=fit_frac,
        x0_overhead_us=x0_overhead_us, bound_overhead_us=bound_overhead_us,
    )

    # Pin F exactly to the stage-1 value (tight bound is a numerical
    # approximation; users want F=F_stage1 reported exactly).
    from dataclasses import replace as _replace
    return _replace(r2, fitted_peak_bf16_tflops=F_stage1)


def apply(spec: HardwareSpec, r: CalibrationResult) -> HardwareSpec:
    """Return a new HardwareSpec with the calibrated scalars swapped in."""
    from dataclasses import replace

    return replace(
        spec,
        peak_bf16_tflops=r.fitted_peak_bf16_tflops,
        hbm_gbps=r.fitted_hbm_gbps,
        per_op_overhead_us=r.fitted_per_op_overhead_us,
    )


# ---------- JSON loader (consumes output of scripts/collect_measurements.py) ----------

_OP_CLASSES: dict[str, type[Op]] = {
    "MatMul": MatMul,
    "AttentionPrefill": AttentionPrefill,
    "AttentionDecode": AttentionDecode,
    "RMSNorm": RMSNorm,
    "SiLUGate": SiLUGate,
    "Residual": Residual,
    "Embedding": Embedding,
    "RoPE": RoPE,
}


def _record_to_op(rec: dict[str, Any]) -> Op:
    """Reconstruct an Op from a ``collect_measurements.py`` record."""
    cls = _OP_CLASSES[rec["op_type"]]
    params = dict(rec["params"])
    name = rec.get("op_name")
    if name is not None:
        params["name"] = name
    return cls(**params)


def load_measurements(
    path: Path, *, dtype_filter: str | None = "bf16",
) -> tuple[list[Op], list[float], dict[str, Any]]:
    """Load a measurements JSON and return ``(ops, median_ms, provenance)``.

    ``dtype_filter``: if set, drop measurements whose op dtype != this string.
    Pass ``None`` to keep everything.
    """
    payload = json.loads(Path(path).read_text())
    ops: list[Op] = []
    measured: list[float] = []
    for rec in payload["measurements"]:
        if dtype_filter is not None and rec["params"].get("dtype") != dtype_filter:
            continue
        ops.append(_record_to_op(rec))
        measured.append(float(rec["median_ms"]))
    provenance = {k: v for k, v in payload.items() if k != "measurements"}
    return ops, measured, provenance
