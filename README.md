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

**Tier 3 — counterfactual experiments shipped.** The simulator is calibrated to
**MAPE 9.4 % held-out** on real H100 measurements (Tier 2 — see
[`reports/02_tier2.md`](./reports/02_tier2.md)) and is now running what-if
scenarios on the resulting model.

Five experiments in [`reports/03_whatif.md`](./reports/03_whatif.md) ask
questions like *"what if HBM bandwidth doubled?"* and *"how does decode latency
scale with context length?"*. The headline non-obvious finding:

> **Llama-1B decode on H100 is overhead-bound, not memory-bound.** ~95 % of
> per-token latency is the launch cost of 227 separate kernels. Doubling HBM
> only buys 1.02 ×; doubling L2 buys 1.01 ×. The big lever for small-model
> single-stream decode is collapsing the launch count itself (CUDA Graphs,
> kernel fusion, batching).

This is the kind of result we built the simulator for — quantitative, surprising,
falsifiable.

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Held-out MAPE | 20.2 % | 9.4 % | 9.4 % |
| Refinements | roofline + global O | + L2 hit-rate + per-family O | (unchanged from T2) |
| Fitted F (TFLOPs) | 1138 (1.15× vendor) | 1172 | 1172 |
| Fitted B (GB/s) | 5375 (1.60× vendor) | **2311** (back in physical range) | 2311 |
| Artifact | [01_validation.md](./reports/01_validation.md) | [02_tier2.md](./reports/02_tier2.md) | [03_whatif.md](./reports/03_whatif.md) |

For a first-time reader, start here: [`reports/tier1_explained.md`](./reports/tier1_explained.md)
walks through MAPE, F, B, O, and the L2-caching artefact from first principles.

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
make test       # pytest (73 tests, CPU-only)
make lint       # ruff + mypy
make calibrate  # Tier 2: refit (F, B, per-family O) on the committed H100 JSON
make validate   # Tier 2: calibrate + regenerate residual plot
make measure    # Tier 1: collect a fresh H100 sweep (needs CUDA + uv sync --extra measure)
make whatif     # Tier 3: regenerate the what-if report
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
