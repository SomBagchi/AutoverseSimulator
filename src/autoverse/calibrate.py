"""Fit :class:`HardwareSpec` parameters to real measurements (checkpoint 2C).

Given a set of ``(op, measured_median_ms)`` pairs, we fit three scalars that
the Tier-0 roofline exposes:

- ``F`` — effective BF16 tensor-core throughput, in **TFLOPs/s**.
- ``B`` — effective HBM bandwidth, in **GB/s**.
- ``O`` — per-op overhead, in **microseconds**.

Predicted latency (Tier-0 roofline + launch overhead) for op ``i``::

    p_i = max(flops_i / (F · 1e12), bytes_i / (B · 1e9)) · 1e3 + O · 1e-3      # ms

We minimise log-space residuals via SciPy ``least_squares`` (Trust-Region
Reflective, with positivity bounds). Log-space is the natural loss because
latencies span 3+ decades (≈ 10 µs to ≈ 10 ms) and we care about relative
error (MAPE), not absolute ms.

The fit/held-out split defaults to a deterministic 70/30 shuffle with a seed,
so the report numbers are reproducible across runs.

Calibration is **dtype-scoped**: for a BF16 baseline, we filter measurements
to BF16 ops. Mixed-precision calibration is a Tier-2+ concern; not worth the
per-dtype-peak fitting complexity at this tier.
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


def predict_ms(op: Op, peak_tflops: float, hbm_gbps: float, overhead_us: float) -> float:
    """Tier-0 roofline prediction, parameterised by ``(F, B, O)``. In **ms**.

    Kept as a free function (not a method of HardwareSpec) so calibration can
    call it in a hot loop without constructing frozen dataclasses per iter.
    """
    compute_s = op.flops() / (peak_tflops * 1e12) if peak_tflops > 0 else 0.0
    bytes_moved = op.bytes_read() + op.bytes_written()
    memory_s = bytes_moved / (hbm_gbps * 1e9) if hbm_gbps > 0 else 0.0
    return max(compute_s, memory_s) * 1e3 + overhead_us * 1e-3


def _residuals(
    params: np.ndarray,
    ops: list[Op],
    measured_ms: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Log-latency residuals: ``log(predicted) - log(measured)``."""
    peak_tflops, hbm_gbps, overhead_us = params
    pred = np.asarray(
        [predict_ms(op, peak_tflops, hbm_gbps, overhead_us) for op in ops],
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
) -> CalibrationResult:
    """Fit (F, B, O) to match measured latencies on ``ops``.

    Arguments:
        ops, measured_ms: parallel lists, same length.
        x0: initial (F_tflops, B_gbps, O_us). Defaults to H100 vendor nominals.
        bounds: ((lb), (ub)) for the three scalars.
        fit_frac: fraction of measurements used for fitting; remainder held out.
        seed: RNG seed for the fit/held-out split.
        verbose: pass through to scipy.
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
        args=(fit_ops, fit_ms_arr),
        bounds=bounds,
        method="trf",
        verbose=2 if verbose else 0,
    )

    peak_tflops, hbm_gbps, overhead_us = result.x
    fit_preds = np.asarray(
        [predict_ms(op, peak_tflops, hbm_gbps, overhead_us) for op in fit_ops]
    )
    ho_preds = (
        np.asarray([predict_ms(op, peak_tflops, hbm_gbps, overhead_us) for op in ho_ops])
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
