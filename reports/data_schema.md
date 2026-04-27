# Data schema

Two JSON files in this repo carry calibration state. Both are produced
by scripts and consumed by the same scripts (or by tests / notebooks /
your downstream tooling). They are committed so that the validation
report renders without rerunning.

## `measurements/<chip>/run_<UTC-tag>.json`

Output of `scripts/collect_measurements.py`. One file per H100 sweep.

```jsonc
{
  // ---- provenance header ----
  "device":          "cuda",                       // 'cuda' / 'cpu' / 'mps'
  "gpu_name":        "NVIDIA H100 80GB HBM3",      // torch.cuda.get_device_name(0)
  "torch_version":   "2.11.0+cu130",
  "timestamp_utc":   "2026-04-26T23:32:51+00:00",  // collection time
  "n_warmup":        10,                           // untimed warm-up iters per op
  "n_iters":         100,                          // timed iters per op (median reported)
  "dtype":           "bf16",                       // sweep-wide dtype
  "quick":           false,                        // true ⇒ shrunken sweep for CPU dry-run

  // ---- one record per op ----
  "measurements": [
    {
      "op_type":  "MatMul",                        // class name in src/autoverse/ops.py
      "op_name":  "matmul_4096x4096x4096",         // unique within sweep; layer-suffixed in graph walks
      "params":   {                                // exact constructor kwargs minus 'name'
        "m": 4096, "k": 4096, "n": 4096,
        "dtype": "bf16"
      },
      "median_ms": 0.1785,                         // CANONICAL latency
      "p10_ms":    0.1779,                         // 10th-percentile across n_iters
      "p90_ms":    0.1796,                         // 90th-percentile
      "mean_ms":   0.1786,
      "std_ms":    0.0009,
      "n_iters":   100
    },
    // ... typically 530 records for the full Llama-1B sweep
  ]
}
```

**Reconstructing an `Op`.** `autoverse.calibrate._record_to_op` looks up
`op_type` in a class map (`{"MatMul": MatMul, ...}`) and calls
`cls(**params)` plus the `name`. So adding a new op type is two lines:
add to `_OP_CLASSES`, ensure constructor takes the same kwargs as in the
JSON.

**Why median.** CUDA-event timings have a heavy right tail (clock
jitter, preemption, neighbour-pod activity). Median is robust; p10/p90
gives a noise band. For the committed H100 run, the median p90/p10
ratio is < 1.05 on every record — the iteration noise floor is well
below the modelling error.

**Filenames.** UTC timestamp tag (`run_YYYYMMDD_HHMMSS.json`) so chronological
sort works lexically. One sweep per file; multiple files per chip OK.

## `reports/calibration_fit.json`

Output of `scripts/calibrate.py` (and `make calibrate` / `make validate`).
The Tier-1 fit summary, derived from a single `measurements/.../run_*.json`.

```jsonc
{
  "source":    "measurements/h100_sxm/run_20260426_233235.json",
  "provenance": { /* copy of the source's provenance header */ },

  // dataclass dump of CalibrationResult
  "fit": {
    "fitted_peak_bf16_tflops":  1138.51,
    "fitted_hbm_gbps":          5374.78,
    "fitted_per_op_overhead_us":  17.40,
    "mape_fit":                  0.1961,           // 19.6%
    "mape_held_out":             0.2020,           // 20.2%
    "n_fit":                     371,
    "n_held_out":                159,
    "per_op_mape_fit": {                           // op_type → MAPE on fit set
      "MatMul":        0.092,
      "AttentionPrefill": 0.082,
      // ...
    },
    "per_op_mape_held_out": { /* same shape */ },
    "residual_cost":             20.79             // final scipy.least_squares half-SSR
  },

  // Top-K by relative error across the WHOLE dataset (fit + held-out).
  // Useful for residual analysis without re-running calibration.
  "worst_fit": [
    {"name": "rope_13", "op_type": "RoPE",
     "measured_ms": 0.0764, "predicted_ms": 0.0194, "rel_err": 0.747},
    // ... 25 entries
  ]
}
```

**Note on units.** Throughputs in the JSON are in TFLOPs/s and GB/s
(matching `HardwareSpec` field units). The roofline arithmetic in
`predict_ms` and `cost.estimate` does the `* 1e12` / `* 1e9` conversion
internally, so callers always pass the human-readable numbers.

**Filtering.** Calibration only fits ops whose `params.dtype` matches the
filter (default `'bf16'`). FP32 / FP16 / FP8 measurements are silently
dropped. Mixed-precision calibration is a Tier-2+ concern.
