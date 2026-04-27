# Autoverse Simulator

> An analytical performance model for transformer inference on GPU-like accelerators.
>
> Named after the *Autoverse* in Greg Egan's **Permutation City**.

## What I built — at a glance

Given a `HardwareSpec` (TFLOPs, HBM bandwidth, L2 capacity, SMs…) and a
`TransformerConfig` (`d_model`, `n_layers`, `n_heads`…), Autoverse predicts the
per-op latency of a forward pass. The model is **calibrated against 530 real
H100 measurements on Llama-3.2-1B** to **MAPE 10.1 %** on a 30 % held-out split,
with both fitted throughputs in physically meaningful ranges. With that
calibrated model in hand, the simulator answers **counterfactual hardware
questions** — *"what if HBM doubled?", "how does decode latency scale with
context length?"* — that would otherwise cost real GPU time per question.

Built as a code sample for the Anthropic Fellows programme.

## Headline results

| | value |
|---|---|
| **Held-out MAPE** | **10.1 %** (n = 159 ops) |
| Fit MAPE | 9.6 % (n = 371 ops) |
| Fitted F (effective BF16 throughput) | **943 TFLOPs** (vendor 989 → 0.95×, physical) |
| Fitted B (effective HBM bandwidth) | **2286 GB/s** (vendor 3350 → 0.68×, physical) |
| Fitted O (per-op-family launch overhead) | RoPE 52 µs, MatMul 17 µs, Residual 12 µs, … |
| Per-op-family MAPE | best 1 % (RoPE, Embedding) → worst 14 % (AttentionPrefill) |

Full breakdown: [`reports/01_methodology.md`](./reports/01_methodology.md). If
you only read one file, read that one.

The headline counterfactual finding from the simulator:

> **Llama-1B decode on H100 is overhead-bound, not memory-bound.** ~95 % of
> per-token latency is the launch cost of 227 separate kernels. Doubling HBM
> only buys 1.02×; doubling L2 buys 1.01×. The big lever for small-model
> single-stream decode is collapsing the launch count itself (CUDA Graphs,
> kernel fusion, batching).

Full counterfactual experiments: [`reports/03_whatif.md`](./reports/03_whatif.md).

## Modelling features

The cost equation per op:

```
predict_ms(op) = max( flops / F,  effective_bytes / B ) + O_family[op]

where  effective_bytes  = bytes_written + bytes_read · (1 − hit_rate)
       hit_rate         = min(1, L2_capacity / bytes_read)
       O_family[op]     = launch overhead for op's family (8 fitted scalars)
```

Three modelling features beyond the bare roofline:

1. **L2 hit-rate heuristic on inputs only.** Caches reduce *re-reads*, so
   the input-bytes term gets a `(1 − hit_rate)` factor; output writes always
   stream to HBM. Brings fitted B back below vendor (was 1.6× vendor without
   it — the optimiser was inflating B to absorb the L2 effect).
2. **Per-op-family launch overhead.** A single global `O` can't capture the
   ~5× spread between RoPE's ~52 µs index-arithmetic launch and Residual's
   ~12 µs elementwise add. We fit one `O` per op type. Was the dominant MAPE
   win (collapsed RoPE / Residual error from 50–65 % to under 10 %).
3. **Two-stage least-squares fit.** F is under-constrained by memory-bound
   ops (∂t/∂F = 0 for them by construction of `max(...)`), so the joint fit
   drifts F to unphysical values. Stage 1 fits F on a compute-bound subset
   only; stage 2 freezes F and fits B + per-family O on the full dataset.
   Brings fitted F back below vendor.

Tried, **rejected**:

- **Wave quantisation for GEMMs** (the trailing-partial-wave-on-N_SM penalty).
  Implemented, behind a `--n-sm` flag. Strictly worsens MAPE on this dataset
  (9.4 → 11.3 %): per-family overhead already absorbs the kernel-launch
  wallclock floor that wave-quant tries to model, so adding wave-quant
  double-counts. Code, tests, and ablation flag retained;
  [`reports/02_refinements.md`](./reports/02_refinements.md) §"Wave
  quantisation: tried, rejected" has the full forensic.

## Highlights

The interesting parts are not the roofline formula (a textbook one). The
interesting parts are the modelling discipline layered on top:

1. **Identifying physical-implausibility as a diagnostic, not just an error.**
   The first fit produced `B = 5375 GB/s` against an H100 vendor peak of
   3350 GB/s. Rather than shrugging at "fit error", I traced it to the
   measurement methodology: 100-iter loops let inputs sit in the 50 MB L2
   cache. Adding the L2 hit-rate heuristic dropped B from 1.6× vendor to
   0.68× vendor — physical. Same diagnostic move applied later to F.
   Documented in [`reports/01_methodology.md`](./reports/01_methodology.md).

2. **Two-stage least-squares to handle a degenerate joint fit.** The roofline's
   `max(t_c, t_m)` makes ∂t/∂F = 0 on memory-bound ops, so they don't constrain
   F — yet they DO add noise via shared per-family overhead coupling. The joint
   fit drifted F to 1.18× vendor. Fixed by fitting F on a compute-bound subset
   first, then refitting B and per-family overhead on the full dataset with F
   frozen. F dropped to 0.95× vendor.

3. **A negative-result section in the report.** Wave quantisation is a
   spec-pinned refinement, and I implemented it. It *strictly worsens* MAPE
   on this dataset (per-family overhead already absorbs the kernel-launch
   floor that wave-quant tries to model, so adding wave-quant double-counts).
   Code, tests, and ablation flag retained.

4. **A non-obvious counterfactual finding the simulator surfaced — and that
   contradicted my prior.** I expected *"double HBM ⇒ ~2× decode speedup"*
   because "decode is memory-bound". The calibrated model predicted **1.02×**.
   Reason: on Llama-1B, ~95 % of decode latency is the launch cost of 227
   separate kernels — decode is overhead-bound, not memory-bound on this model
   class. Hardware throughput knobs (HBM, compute, L2) act on the 5 %
   non-overhead slice; the leverage is collapsing the launch count itself.

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
                        │  │ measure.py +   │   │
                        │  │ scripts/       │   │
                        │  │ collect_       │   │
                        │  │ measurements   │   │
                        │  └────────┬───────┘   │
                        │           ▼           │
                        │  ┌────────────────┐   │
                        └─►│ calibrate.py   ├───┘
                           │ scipy.least_   │
                           │ squares        │
                           └────────────────┘
```

- **Prediction (CPU only).** `model.py` lowers a `TransformerConfig` to a flat
  list of `Op`s. `ops.py` knows the FLOPs and HBM bytes of each. `cost.py`
  applies the roofline `max(flops/F, eff_bytes/B) + O_family` per op (with the
  L2 hit-rate heuristic on `eff_bytes`). `simulator.py` sums them.
- **Calibration (needs CUDA).** `measure.py` times each op type via
  `torch.cuda.Event`. `scripts/collect_measurements.py` runs a 530-op sweep
  and dumps JSON. `calibrate.py` runs the two-stage fit via
  `scipy.least_squares` on log-residuals.
- The two meet at `HardwareSpec`: calibration replaces vendor nominals with
  fitted-effective values; the prediction pipeline uses those for what-ifs.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SomBagchi/AutoverseSimulator.git
cd AutoverseSimulator
uv sync                 # CPU-only — runs the simulator and re-fits on committed measurements
uv sync --extra measure # also installs torch — only needed to collect a fresh H100 sweep
```

## Run

```bash
# Predict Llama-1B decode latency
uv run python -m autoverse simulate --model llama1b --mode decode --seq-len 1024 --breakdown

# Re-fit + regenerate the residual plot from the committed H100 measurements
make validate

# Regenerate the what-if report from the calibrated fit
make whatif
```

Other Make targets:

```bash
make test       # 81 tests, CPU-only, ~5s
make lint       # ruff + mypy --strict
make measure    # collect a fresh H100 sweep (needs CUDA)
```

### Collecting a fresh measurement run on an H100 pod

```bash
# On the H100 box:
git clone https://github.com/SomBagchi/AutoverseSimulator.git && cd AutoverseSimulator
uv sync --extra measure
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
# → expect e.g. "NVIDIA H100 80GB HBM3"

uv run python scripts/collect_measurements.py \
    --device cuda \
    --out measurements/h100_sxm/run_$(date -u +%Y%m%d_%H%M%S).json
# Full 530-op sweep takes ~10 seconds.
```

Then either commit the JSON and update `MEASUREMENTS = ...` at the top of
`Makefile`, or pass it inline: `MEASUREMENTS=measurements/h100_sxm/run_<tag>.json make validate`.

## License

MIT — see [`LICENSE`](./LICENSE).
