"""Tests for :mod:`autoverse.calibrate`.

Key test: synthetic-recovery. Generate measurements from :func:`estimate` with
known ``(F, B, Ov)``; calibration should recover them to within 1%.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoverse import LLAMA_1B
from autoverse.calibrate import (
    CalibrationResult,
    apply,
    calibrate,
    load_measurements,
    predict_ms,
    split_fit_held_out,
)
from autoverse.hardware import H100_SXM, HardwareSpec
from autoverse.model import build_op_graph
from autoverse.ops import MatMul, RMSNorm


def _sample_ops() -> list:
    """A shape-varied sample that covers compute-bound, memory-bound, and
    overhead-dominated regimes so the three params are identifiable."""
    return [
        # Big GEMMs: compute-bound, pin F.
        MatMul(m=4096, k=4096, n=4096),
        MatMul(m=2048, k=4096, n=4096),
        MatMul(m=1024, k=4096, n=4096),
        # Thin GEMMs: memory-bound-ish, pin B.
        MatMul(m=1, k=4096, n=128_256),
        MatMul(m=1, k=4096, n=4096),
        # Large elementwise RMSNorm: memory-bound, pin B.
        RMSNorm(n_tokens=4096, d_model=4096),
        RMSNorm(n_tokens=2048, d_model=4096),
        # Tiny ops: overhead-dominated, pin O.
        RMSNorm(n_tokens=1, d_model=128),
        MatMul(m=8, k=16, n=16),
        MatMul(m=16, k=16, n=16),
    ]


def _synth_measure(ops: list, F: float, B: float, Ov: float) -> list[float]:
    """Generate noise-free measurements from the Tier-0 roofline itself."""
    return [predict_ms(op, F, B, Ov) for op in ops]


# ---------- Synthetic recovery ----------


def test_calibration_recovers_synthetic_params_to_within_1pct() -> None:
    F_true, B_true, Ov_true = 500.0, 2000.0, 5.0
    ops = _sample_ops()
    measured = _synth_measure(ops, F_true, B_true, Ov_true)

    r = calibrate(ops, measured, x0=(989.0, 3350.0, 1.0), seed=42)

    assert abs(r.fitted_peak_bf16_tflops - F_true) / F_true < 0.01
    assert abs(r.fitted_hbm_gbps - B_true) / B_true < 0.01
    assert abs(r.fitted_per_op_overhead_us - Ov_true) / Ov_true < 0.01


def test_calibration_mape_is_near_zero_on_noise_free_synthetic() -> None:
    F, B, Ov = 600.0, 2500.0, 2.0
    ops = _sample_ops()
    measured = _synth_measure(ops, F, B, Ov)
    r = calibrate(ops, measured, seed=0)
    # Noise-free data generated from the model itself ⇒ MAPE ≈ 0.
    assert r.mape_fit < 1e-3
    assert r.mape_held_out is not None
    assert r.mape_held_out < 1e-3


def test_calibration_is_robust_to_small_noise() -> None:
    """Add +-5% multiplicative noise; recovered params still within 5%."""
    import numpy as np

    F, B, Ov = 700.0, 2800.0, 3.0
    ops = _sample_ops()
    clean = _synth_measure(ops, F, B, Ov)
    rng = np.random.default_rng(123)
    noise = rng.uniform(0.95, 1.05, size=len(clean))
    noisy = [c * n for c, n in zip(clean, noise, strict=True)]

    r = calibrate(ops, noisy, seed=0)
    assert abs(r.fitted_peak_bf16_tflops - F) / F < 0.05
    assert abs(r.fitted_hbm_gbps - B) / B < 0.05
    # MAPE is bounded by the noise scale.
    assert r.mape_fit < 0.08


# ---------- Structural checks ----------


def test_calibration_respects_bounds() -> None:
    """Even if data nominates infeasible params, fit stays in the bounding box."""
    F, B, Ov = 989.0, 3350.0, 1.0
    ops = _sample_ops()
    measured = _synth_measure(ops, F, B, Ov)
    r = calibrate(ops, measured,
                  bounds=((100.0, 100.0, 0.0), (1500.0, 5000.0, 100.0)))
    assert 100.0 <= r.fitted_peak_bf16_tflops <= 1500.0
    assert 100.0 <= r.fitted_hbm_gbps <= 5000.0
    assert 0.0 <= r.fitted_per_op_overhead_us <= 100.0


def test_calibration_rejects_length_mismatch() -> None:
    ops = _sample_ops()
    with pytest.raises(ValueError, match="length mismatch"):
        calibrate(ops, [1.0, 2.0])


def test_calibration_rejects_too_few_points() -> None:
    ops = _sample_ops()[:2]
    measured = [1.0, 2.0]
    with pytest.raises(ValueError, match=">=4"):
        calibrate(ops, measured)


def test_split_is_deterministic_with_same_seed() -> None:
    ops = _sample_ops()
    measured = [float(i) for i in range(len(ops))]
    a1, b1, c1, d1 = split_fit_held_out(ops, measured, seed=7)
    a2, b2, c2, d2 = split_fit_held_out(ops, measured, seed=7)
    assert [op.name for op in a1] == [op.name for op in a2]
    assert b1 == b2
    assert [op.name for op in c1] == [op.name for op in c2]
    assert d1 == d2


def test_split_respects_fit_frac() -> None:
    ops = _sample_ops()  # 10 ops
    measured = [float(i) for i in range(len(ops))]
    fit_ops, fit_ms, ho_ops, ho_ms = split_fit_held_out(ops, measured, fit_frac=0.7, seed=0)
    assert len(fit_ops) == 7
    assert len(ho_ops) == 3
    assert len(fit_ops) == len(fit_ms)
    assert len(ho_ops) == len(ho_ms)


def test_apply_produces_updated_hardware_spec() -> None:
    r = CalibrationResult(
        fitted_peak_bf16_tflops=500.0,
        fitted_hbm_gbps=2000.0,
        fitted_per_op_overhead_us=3.0,
        mape_fit=0.0,
        mape_held_out=0.0,
        n_fit=7, n_held_out=3,
    )
    spec = apply(H100_SXM, r)
    assert isinstance(spec, HardwareSpec)
    assert spec.peak_bf16_tflops == 500.0
    assert spec.hbm_gbps == 2000.0
    assert spec.per_op_overhead_us == 3.0
    # Untouched fields preserved.
    assert spec.n_sm == H100_SXM.n_sm
    assert spec.l2_mb == H100_SXM.l2_mb


def test_predict_ms_matches_cost_estimate() -> None:
    """predict_ms is the same roofline as cost.estimate, parameterised differently."""
    from autoverse.cost import estimate

    op = MatMul(m=1024, k=2048, n=2048, dtype="bf16")
    # Use H100 nominals so both paths produce the same number.
    spec = H100_SXM
    t_from_cost = estimate(op, spec).effective_ms
    t_from_calibrate = predict_ms(op, spec.peak_bf16_tflops, spec.hbm_gbps,
                                   spec.per_op_overhead_us)
    assert abs(t_from_cost - t_from_calibrate) < 1e-9


# ---------- Per-op MAPE breakdown ----------


def test_per_op_mape_covers_each_op_type_present() -> None:
    F, B, Ov = 500.0, 2000.0, 2.0
    ops = _sample_ops()
    measured = _synth_measure(ops, F, B, Ov)
    r = calibrate(ops, measured, seed=0)
    # Keys on the fit set are a subset of present op types.
    present = {type(op).__name__ for op in ops}
    assert set(r.per_op_mape_fit).issubset(present)
    # Noise-free synthetic data ⇒ every bucket near zero.
    for v in r.per_op_mape_fit.values():
        assert v < 1e-3


# ---------- JSON round-trip ----------


def test_load_measurements_reconstructs_ops_and_filters_dtype(tmp_path: Path) -> None:
    import dataclasses as _dc

    ops = _sample_ops() + [MatMul(m=1, k=1, n=1, dtype="fp32")]  # mix dtype
    measured = _synth_measure(_sample_ops(), 500.0, 2000.0, 2.0) + [0.001]
    records = []
    for op, m in zip(ops, measured, strict=True):
        fields = _dc.fields(op)  # type: ignore[arg-type]
        records.append({
            "op_type": type(op).__name__,
            "op_name": op.name,
            "params": {f.name: getattr(op, f.name) for f in fields if f.name != "name"},
            "median_ms": m,
            "p10_ms": m, "p90_ms": m, "mean_ms": m, "std_ms": 0.0, "n_iters": 10,
        })
    payload = {
        "device": "cuda", "gpu_name": "synthetic", "torch_version": "x",
        "timestamp_utc": "2026-04-23T00:00:00+00:00",
        "n_warmup": 1, "n_iters": 10, "dtype": "bf16", "quick": False,
        "measurements": records,
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))

    loaded_ops, loaded_ms, prov = load_measurements(path, dtype_filter="bf16")
    # The fp32 MatMul is filtered out.
    assert len(loaded_ops) == len(ops) - 1
    assert all(op.dtype == "bf16" for op in loaded_ops)
    assert prov["device"] == "cuda"

    # No filter ⇒ keep everything.
    all_ops, _, _ = load_measurements(path, dtype_filter=None)
    assert len(all_ops) == len(ops)


def test_calibration_on_llama_graph_is_stable() -> None:
    """Smoke: a full Llama-1B prefill graph calibrates without diverging.

    Prefill at seq_len=2048 mixes compute-bound big GEMMs (MLP gate/up/down at
    M=2048, K=2048, N=8192 sits well above the H100 ridge) with memory-bound
    elementwise ops. Decode would leave F unconstrained — every matmul has
    M=1, so nothing reaches the compute roof — which is itself worth knowing:
    real calibration must include prefill shapes (or standalone big GEMMs).
    """
    F, B, Ov = 800.0, 3000.0, 2.0
    ops = build_op_graph(LLAMA_1B, seq_len=2048, mode="prefill")
    measured = _synth_measure(ops, F, B, Ov)
    r = calibrate(ops, measured, seed=0)
    assert abs(r.fitted_peak_bf16_tflops - F) / F < 0.02
    assert abs(r.fitted_hbm_gbps - B) / B < 0.02
    assert abs(r.fitted_per_op_overhead_us - Ov) / Ov < 0.05


def test_calibration_of_decode_only_graph_underdetermines_peak_compute() -> None:
    """Decode-only measurements cannot pin F: every matmul has M=1 ⇒ all ops
    are memory-bound or overhead-dominated, F is free. The optimiser should
    exit near the initial guess for F but still fit B and O cleanly.
    """
    F_true, B_true, Ov_true = 800.0, 3000.0, 2.0
    ops = build_op_graph(LLAMA_1B, seq_len=256, mode="decode")
    measured = _synth_measure(ops, F_true, B_true, Ov_true)
    r = calibrate(ops, measured, x0=(989.0, 3350.0, 1.0), seed=0)
    assert abs(r.fitted_hbm_gbps - B_true) / B_true < 0.02
    assert abs(r.fitted_per_op_overhead_us - Ov_true) / Ov_true < 0.05
    # MAPE is still near zero — F doesn't matter when nothing hits the compute roof.
    assert r.mape_fit < 1e-3
