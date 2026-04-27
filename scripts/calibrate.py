"""CLI wrapper around :mod:`autoverse.calibrate`.

Loads a measurements JSON produced by ``scripts/collect_measurements.py``,
fits ``(peak_bf16_tflops, hbm_gbps, per_op_overhead_us)``, and prints a
human-readable report. Optionally writes the fit + diagnostics to a JSON
sidecar so downstream tools (the validation report, ``make validate``) can
consume it.

Typical use::

    uv run python scripts/calibrate.py \\
        measurements/h100_sxm/run_20260426_233235.json \\
        --out reports/calibration_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from autoverse.calibrate import (
    calibrate,
    calibrate_per_family,
    calibrate_two_stage,
    load_measurements,
    predict_ms,
)
from autoverse.hardware import H100_SXM


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("measurements", type=Path, help="JSON from collect_measurements.py")
    p.add_argument("--dtype-filter", default="bf16",
                   help="Restrict fit to ops of this dtype (default bf16; '' = all).")
    p.add_argument("--seed", type=int, default=0, help="Fit/held-out split seed.")
    p.add_argument("--fit-frac", type=float, default=0.7)
    p.add_argument("--l2-mb", type=float, default=H100_SXM.l2_mb,
                   help="L2 capacity (MB) for the L2 hit-rate heuristic. "
                        "Set 0 to ablate.")
    p.add_argument("--n-sm", type=int, default=0,
                   help=f"SM count for the wave-quantisation heuristic on MatMul. "
                        f"Default 0 = OFF (wave quant strictly worsens MAPE on "
                        f"this dataset; see reports/02_tier2.md). Pass "
                        f"{H100_SXM.n_sm} to enable on H100 for ablation.")
    p.add_argument("--global-overhead", action="store_true",
                   help="Use a single global per-op overhead instead of per-family. "
                        "Default is per-family.")
    p.add_argument("--single-stage", action="store_true",
                   help="Use the joint single-stage fit (F + B + O all together) "
                        "instead of the two-stage variant (F from compute-bound; "
                        "B + O from all with F frozen). Two-stage is the default.")
    p.add_argument("--ridge-flops-per-byte", type=float, default=295.0,
                   help="Arithmetic-intensity threshold for stage 1's compute-bound "
                        "subset (default 295 ≈ H100 vendor ridge point).")
    p.add_argument("--out", type=Path, default=None,
                   help="Write fit + per-op diagnostics here as JSON.")
    args = p.parse_args()

    dtype_filter = args.dtype_filter or None
    ops, measured_ms, prov = load_measurements(args.measurements, dtype_filter=dtype_filter)
    if not ops:
        print(f"[calibrate] no ops loaded from {args.measurements}"
              f" (dtype_filter={dtype_filter!r})", file=sys.stderr)
        return 2

    print("=== Source ===")
    print(f"  file              : {args.measurements}")
    print(f"  gpu_name          : {prov.get('gpu_name')}")
    print(f"  torch             : {prov.get('torch_version')}")
    print(f"  dtype             : {prov.get('dtype')}")
    print(f"  iters per op      : {prov.get('n_iters')} (warmup {prov.get('n_warmup')})")
    print(f"  total ops loaded  : {len(ops)} (filter dtype={dtype_filter!r})")
    print(f"  L2 heuristic      : l2_mb={args.l2_mb}"
          f" {'(disabled)' if args.l2_mb == 0 else '(enabled)'}")
    print(f"  Wave quant (MatMul): n_sm={args.n_sm}"
          f" {'(disabled)' if args.n_sm == 0 else '(enabled)'}")
    print(f"  overhead model    : "
          f"{'global' if args.global_overhead else 'per-family'}")
    fit_mode = ("global single-stage" if args.global_overhead
                else "single-stage per-family" if args.single_stage
                else "two-stage (F on compute-bound; B+O on full)")
    print(f"  fit mode          : {fit_mode}")
    print()

    if args.global_overhead:
        r = calibrate(ops, measured_ms, seed=args.seed, fit_frac=args.fit_frac,
                      l2_mb=args.l2_mb, n_sm=args.n_sm)
    elif args.single_stage:
        r = calibrate_per_family(ops, measured_ms, seed=args.seed, fit_frac=args.fit_frac,
                                  l2_mb=args.l2_mb, n_sm=args.n_sm)
    else:
        r = calibrate_two_stage(ops, measured_ms,
                                 seed=args.seed, fit_frac=args.fit_frac,
                                 l2_mb=args.l2_mb, n_sm=args.n_sm,
                                 ridge_flops_per_byte=args.ridge_flops_per_byte)

    print("=== Fitted scalars (vs H100-SXM vendor nominal) ===")
    print(f"  peak_bf16_tflops    : {r.fitted_peak_bf16_tflops:8.2f}   "
          f"(vendor {H100_SXM.peak_bf16_tflops:.0f})")
    print(f"  hbm_gbps            : {r.fitted_hbm_gbps:8.1f}   "
          f"(vendor {H100_SXM.hbm_gbps:.0f})")
    if r.fitted_overhead_by_family:
        print("  per_op_overhead_us  : <per-family>")
        for fam, ov in sorted(r.fitted_overhead_by_family.items(), key=lambda x: -x[1]):
            print(f"    {fam:20s}  {ov:7.2f} µs")
    else:
        print(f"  per_op_overhead_us  : {r.fitted_per_op_overhead_us:8.3f}")
    print()

    print("=== Accuracy ===")
    print(f"  MAPE  fit          : {r.mape_fit*100:6.2f}%   (n={r.n_fit})")
    if r.mape_held_out is not None:
        print(f"  MAPE  held-out     : {r.mape_held_out*100:6.2f}%   (n={r.n_held_out})")
    print(f"  residual cost      : {r.residual_cost:.4e}")
    print()

    print("=== Per-op MAPE breakdown (held-out) ===")
    for k, v in sorted(r.per_op_mape_held_out.items(), key=lambda x: -x[1]):
        marker = "  ⚠" if v > 0.30 else ""
        print(f"  {k:20s}  {v*100:6.1f}%{marker}")
    print()

    # Show worst predictions for residual analysis.
    overhead_by_family = r.fitted_overhead_by_family or None
    preds = [predict_ms(op, r.fitted_peak_bf16_tflops, r.fitted_hbm_gbps,
                        r.fitted_per_op_overhead_us, args.l2_mb,
                        overhead_by_family, n_sm=args.n_sm) for op in ops]
    rows = [
        (op.name, type(op).__name__, m, p, abs(p - m) / max(m, 1e-9))
        for op, m, p in zip(ops, measured_ms, preds, strict=True)
    ]
    rows.sort(key=lambda x: -x[4])
    print("=== 10 worst-fit ops (across whole dataset) ===")
    print(f"  {'name':32s} {'type':18s} {'measured(ms)':>12s} {'pred(ms)':>10s} {'err':>8s}")
    for name, typ, m, pred, err in rows[:10]:
        print(f"  {name:32s} {typ:18s} {m:12.4f} {pred:10.4f} {err*100:7.1f}%")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(args.measurements),
            "provenance": prov,
            "fit": asdict(r),
            "worst_fit": [
                {"name": n, "op_type": t, "measured_ms": m, "predicted_ms": p, "rel_err": e}
                for n, t, m, p, e in rows[:25]
            ],
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n[calibrate] wrote fit summary to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
