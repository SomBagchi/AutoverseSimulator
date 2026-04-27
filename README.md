# Autoverse Simulator

> An analytical performance model for transformer inference on GPU-like accelerators.
>
> Named after the *Autoverse* in Greg Egan's **Permutation City** — a simulation whose value is in the counterfactual questions you can ask of it, not how fast it runs.

## What this is

Autoverse takes two inputs:

1. A **hardware spec** — SM count, tensor-core TFLOPs, HBM bandwidth, L2 capacity, etc.
2. A **transformer configuration** — d_model, n_layers, n_heads, etc.

…and predicts per-op latency for prefill and decode. Predictions are calibrated against real H100 measurements on Llama-3.2-1B. Once validated, the model answers *"what if HBM bandwidth doubled?"* or *"what if L2 were bigger?"* — counterfactuals that would cost thousands of dollars of real-hardware time to answer by measurement.

## Architecture

Two pipelines, sharing a single `Op` representation:

```
                  PREDICTION PIPELINE
  ┌──────────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
  │ TransformerConfig├───►│ ops.py   ├───►│ cost.py    ├───►│simulator │
  │ (model.py)       │    │ FLOPs +  │    │ roofline   │    │ .py      │
  │                  │    │ bytes    │    │ t_predict  │    │ Σ ops    │
  │ HardwareSpec     ├──┐ │ per op   │    │            │    │          │
  │ (hardware.py)    │  │ └──────────┘    └─────▲──────┘    └────┬─────┘
  └──────────────────┘  │                       │                │
                        │                       │ (F, B, O)      ▼
                        │                       │           per-op-ms
                        │                       │
                        │  CALIBRATION PIPELINE │
                        │  ┌────────────────┐   │
                        │  │ measure.py     │   │
                        │  │ scripts/       │   │
                        │  │ collect_       │   │
                        │  │ measurements   │   │
                        │  │ .py            │   │
                        │  └────────┬───────┘   │
                        │           │           │
                        │           ▼           │
                        │  ┌────────────────┐   │
                        └─►│ calibrate.py   ├───┘
                           │ scipy.least_   │
                           │ squares        │
                           └────────────────┘
```

- **Prediction pipeline (CPU only):** `model.py` lowers a `TransformerConfig`
  to a flat list of `Op`s. Each `Op` knows its FLOPs and HBM bytes (`ops.py`).
  `cost.py` applies a roofline `max(flops/F, bytes/B) + O` per op. `simulator.py`
  sums per-op times.
- **Calibration pipeline (needs CUDA):** `measure.py` times each op type on
  real hardware via `torch.cuda.Event`. `scripts/collect_measurements.py` runs
  a sweep and dumps JSON. `calibrate.py` fits the three scalars (F, B, O) to
  the measurement set via SciPy `least_squares` on log-residuals.
- The two meet at `HardwareSpec`: calibration replaces vendor nominals with
  effective fitted values; the prediction pipeline then uses those to answer
  "what-if" questions in Tier 3.

## Status

**Tier 1 — validated against real H100.** The pipeline now fits three calibration
scalars (`peak_bf16_tflops`, `hbm_gbps`, `per_op_overhead_us`) to 530 measurements
collected on one NVIDIA H100 80GB HBM3, achieving **MAPE 20.2% on a 30% held-out
split** — under the Tier-1 ≤ 30 % target. Big ops (MatMul 9 %, AttentionPrefill 8 %)
modelled well; the remaining error concentrates on small overhead-dominated ops
(RoPE, Residual) and is queued for the per-op-family overhead refinement in Tier 2.

Full numbers, residual plot, and honest analysis of failure modes:
[`reports/01_validation.md`](./reports/01_validation.md).

| | fitted | vendor (H100-SXM) |
|---|---|---|
| `peak_bf16_tflops` | 1138.5 | 989 |
| `hbm_gbps` | 5374.8 | 3350 |
| `per_op_overhead_us` | 17.40 | — |

(Both throughputs land *above* vendor nominal — the optimiser is compensating for
L2-hit-rate effects in tight measurement loops and imperfect compute/memory
overlap. Both are named Tier-2 refinements; see the report.)

Sample output for one-token decode at `ctx_len=1024` on the **fitted** spec:

```
[autoverse] simulate: model=llama1b mode=decode seq_len=1024
  ops simulated: 227
  total latency: 0.749 ms  (uncalibrated Tier-0 roofline)
  per-op-family breakdown (effective_ms):
    mlp_gate        0.160 ms  ( 21.4%, 16 ops)
    mlp_up          0.160 ms  ( 21.4%, 16 ops)
    mlp_down        0.160 ms  ( 21.4%, 16 ops)
    lm_head         0.157 ms  ( 21.0%,  1 op)
    q_proj          0.040 ms  (  5.4%, 16 ops)
    out_proj        0.040 ms  (  5.4%, 16 ops)
    ...
```

See [`CLAUDE.md`](./CLAUDE.md) for the per-day roadmap and pinned modelling decisions.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
# Simulate Llama-1B decode — total latency only
uv run python -m autoverse simulate --model llama1b --mode decode --seq-len 1024

# …or with a per-op-family breakdown
uv run python -m autoverse simulate --model llama1b --mode prefill --seq-len 1024 --breakdown
```

Additional commands (from the Makefile):

```bash
make test       # pytest (66 tests, CPU-only)
make lint       # ruff + mypy
make calibrate  # Tier 1: refit (F, B, O) on the committed H100 JSON
make validate   # Tier 1: calibrate + regenerate residual plot
make measure    # Tier 1: collect a fresh H100 sweep (needs CUDA + uv sync --extra measure)
make whatif     # Tier 3+: run counterfactual experiments
```

### Full end-to-end workflow

The committed measurements + fit are reproducible from CPU; only the
*collection* step needs an H100.

**1. Verify the simulator works (CPU, no measurement).**

```bash
git clone https://github.com/SomBagchi/AutoverseSimulator.git
cd AutoverseSimulator
uv sync
make test                    # 66 tests, ~5s
uv run python -m autoverse simulate --model llama1b --mode decode --seq-len 1024 --breakdown
```

**2. Re-fit on the committed H100 measurements (CPU, no GPU needed).**

```bash
make validate
# Produces:
#   reports/calibration_fit.json          (fitted F, B, O + per-op MAPE)
#   reports/figures/measured_vs_predicted.png
# And prints the headline + worst-fit ops to stdout.
```

**3. Collect a fresh sweep on a new H100 box.** Used here on a RunPod
H100-SXM with no extra setup needed beyond `uv` + a CUDA driver:

```bash
# On the GPU box:
git clone https://github.com/SomBagchi/AutoverseSimulator.git && cd AutoverseSimulator
uv sync --extra measure
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
# → expect e.g. "NVIDIA H100 80GB HBM3"

uv run python scripts/collect_measurements.py \
    --device cuda \
    --out measurements/h100_sxm/run_$(date -u +%Y%m%d_%H%M%S).json
# ~10 seconds for the full 530-op sweep.
```

**4. Use the new sweep.** Either commit it and update `MEASUREMENTS = ...`
at the top of `Makefile`, or pass `MEASUREMENTS=...` inline:

```bash
MEASUREMENTS=measurements/h100_sxm/run_<your-tag>.json make validate
```

The provenance header in each JSON (GPU name, torch version, timestamp,
iter counts) makes the fit reproducible across machines.

## Why this project

Built as a code sample for the [Anthropic Fellows program](https://job-boards.greenhouse.io/anthropic/jobs/5023394008) — specifically the ML Systems & Performance track, which lists "CPU simulators for accelerator workloads" as an example project shape.

## License

MIT — see [`LICENSE`](./LICENSE).
