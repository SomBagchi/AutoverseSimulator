"""Generate the validation residual plot for the Tier-1 report (checkpoint 2D).

Reads ``reports/calibration_fit.json`` (produced by ``scripts/calibrate.py``)
plus the source measurements and renders a log-log scatter of
**predicted vs measured** latency, coloured by op family, with a y=x diagonal.

Output: ``reports/figures/measured_vs_predicted.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoverse.calibrate import load_measurements, predict_ms  # noqa: E402
from autoverse.hardware import H100_SXM  # noqa: E402

# Stable colour mapping so plots across runs stay comparable.
_COLOURS = {
    "MatMul":           "#1f77b4",
    "AttentionPrefill": "#ff7f0e",
    "AttentionDecode":  "#d62728",
    "RMSNorm":          "#2ca02c",
    "SiLUGate":         "#9467bd",
    "Residual":         "#8c564b",
    "Embedding":        "#e377c2",
    "RoPE":             "#7f7f7f",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path,
                   default=Path("measurements/h100_sxm/run_20260426_233235.json"))
    p.add_argument("--fit", type=Path,
                   default=Path("reports/calibration_fit.json"))
    p.add_argument("--out", type=Path,
                   default=Path("reports/figures/measured_vs_predicted.png"))
    p.add_argument("--l2-mb", type=float, default=H100_SXM.l2_mb,
                   help="L2 capacity used for the heuristic in predict_ms. "
                        "Must match the value the fit was produced with.")
    p.add_argument("--n-sm", type=int, default=0,
                   help="SM count for wave-quant heuristic. Default 0 (off) "
                        "matches the published Tier-2 fit. Must match the "
                        "value the fit was produced with.")
    args = p.parse_args()

    if not args.fit.exists():
        print(f"[plot] missing {args.fit} — run scripts/calibrate.py first", file=sys.stderr)
        return 2

    fit = json.loads(args.fit.read_text())["fit"]
    F = fit["fitted_peak_bf16_tflops"]
    B = fit["fitted_hbm_gbps"]
    Ov = fit["fitted_per_op_overhead_us"]
    overhead_by_family = fit.get("fitted_overhead_by_family") or None
    mape_ho = fit.get("mape_held_out") or fit["mape_fit"]

    ops, measured_ms, prov = load_measurements(args.measurements, dtype_filter="bf16")
    pred = np.asarray([predict_ms(op, F, B, Ov, args.l2_mb, overhead_by_family,
                                    n_sm=args.n_sm) for op in ops])
    meas = np.asarray(measured_ms)
    types = [type(op).__name__ for op in ops]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Per-type scatter.
    for t in sorted(set(types)):
        mask = np.asarray([x == t for x in types])
        ax.scatter(meas[mask], pred[mask], c=_COLOURS.get(t, "#000"),
                    label=f"{t} (n={int(mask.sum())})",
                    s=18, alpha=0.7, edgecolor="none")

    # y=x reference + ±30% band.
    lo = float(min(meas.min(), pred.min()) * 0.5)
    hi = float(max(meas.max(), pred.max()) * 2.0)
    grid = np.geomspace(lo, hi, 50)
    ax.plot(grid, grid, "k-", linewidth=1, alpha=0.6, label="y = x")
    ax.fill_between(grid, grid * 0.7, grid * 1.3, color="grey",
                     alpha=0.10, label="±30% band")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured (ms)")
    ax.set_ylabel("predicted (ms)")
    tier_label = "Tier-2" if overhead_by_family else "Tier-1"
    overhead_label = "<per-family>" if overhead_by_family else f"{Ov:.1f} μs"
    ax.set_title(
        f"{tier_label} roofline fit on {prov.get('gpu_name')}\n"
        f"F={F:.0f} TFLOPs · B={B:.0f} GB/s · O={overhead_label} · "
        f"MAPE held-out = {mape_ho*100:.1f}%"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[plot] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
