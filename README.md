# Autoverse Simulator

> An analytical performance model for transformer inference on GPU-like accelerators.
>
> Named after the *Autoverse* in Greg Egan's **Permutation City** — a simulation whose value is in the counterfactual questions you can ask of it, not how fast it runs.

## What I built — at a glance

Given a `HardwareSpec` (TFLOPs, HBM bandwidth, L2 capacity, SMs…) and a
`TransformerConfig` (`d_model`, `n_layers`, `n_heads`…), Autoverse predicts the
per-op latency of a forward pass. The model is **calibrated against 530 real
H100 measurements on Llama-3.2-1B** to **MAPE 10.1 %** on a 30 % held-out split,
with both fitted throughputs in physically meaningful ranges. With that
calibrated model in hand, the simulator answers **counterfactual hardware
questions** — *"what if HBM doubled?", "how does decode latency scale with
context length?"* — that would otherwise cost real GPU time per question.

Built solo, end-to-end, in a 4-day sprint as a code sample for the
[Anthropic Fellows programme](https://job-boards.greenhouse.io/anthropic/jobs/5023394008)
(ML Systems & Performance track).

## Highlights for a reviewer

The interesting parts are not "I implemented a roofline" — that's a textbook
formula. The interesting parts are the **modelling discipline** layered on top:

1. **End-to-end scientific loop in 4 days.** Build → measure on a real H100 →
   calibrate via `scipy.least_squares` → diagnose where the model is wrong →
   add one refinement → re-fit → use the calibrated model for what-ifs. Six
   named checkpoints per tier, one commit each, every commit green on
   `make test && make lint`. 81 tests, `mypy --strict` clean.

2. **Identifying physical-implausibility as a diagnostic, not just an error.**
   The Tier-1 fit produced `B = 5375 GB/s` against an H100 vendor peak of
   3350 GB/s. Rather than shrugging at "fit error", I traced it to the
   measurement methodology: 100-iter loops let inputs sit in the 50 MB L2
   cache. Added a one-line L2 hit-rate heuristic
   (`hit_rate = min(1, L2_cap / bytes_read)`); B fell from 1.6× vendor to
   0.68× vendor — physical. Same diagnostic move applied later to F.
   Documented in [`reports/tier1_explained.md`](./reports/tier1_explained.md).

3. **Two-stage least-squares to handle a degenerate joint fit.** The roofline's
   `max(t_c, t_m)` makes ∂t/∂F = 0 on memory-bound ops, so they don't constrain
   F — but they DO add noise via shared per-family overhead coupling. The joint
   fit drifted F to 1.18× vendor (impossible). Fixed by fitting F on a
   compute-bound subset (where ∂t/∂F is nonzero everywhere), then refitting B
   and per-family overhead on the full dataset with F frozen. F dropped to
   0.95× vendor — now physical. Algorithmic fix; no new measurements.

4. **A negative-result section in the report.** Wave quantisation is the
   spec-pinned next refinement, and I implemented it. It *strictly worsens*
   MAPE on this dataset (9.4 → 11.3 %) — per-family overhead already absorbs
   the kernel-launch floor that wave-quant tries to model, so adding wave-quant
   double-counts. Code, tests, and ablation flags retained; documented in
   [`reports/02_tier2.md`](./reports/02_tier2.md) §"Wave quantisation: tried,
   rejected".

5. **A non-obvious counterfactual finding the simulator surfaced — and that
   contradicted my prior.** I expected *"double HBM ⇒ ~2× decode speedup"*
   because "decode is memory-bound". The calibrated model predicted **1.02×**.
   Reason: on Llama-1B, ~95 % of decode latency is the launch cost of 227
   separate kernels. **Llama-1B decode on H100 is overhead-bound, not
   memory-bound.** Hardware throughput knobs (HBM, compute, L2) act on the 5 %
   non-overhead slice; the leverage is collapsing the launch count itself
   (CUDA Graphs, fusion, batching). Documented in
   [`reports/03_whatif.md`](./reports/03_whatif.md).

## Status

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Refinements | roofline + global O | + L2 hit-rate + per-family O + two-stage F/B fit | (unchanged from T2) |
| Held-out MAPE | 20.2 % | **10.1 %** | 10.1 % |
| Fitted F (TFLOPs) | 1138 (1.15× vendor) | **943** (0.95× vendor) ✓ physical | 943 |
| Fitted B (GB/s) | 5375 (1.60× vendor) | **2286** (0.68× vendor) ✓ physical | 2286 |
| Artifact | [01_validation.md](./reports/01_validation.md) | [02_tier2.md](./reports/02_tier2.md) | [03_whatif.md](./reports/03_whatif.md) |

The full pedagogical walkthrough — what F, B, O, and MAPE mean, and what those
numbers actually say about an H100 — is in
[`reports/tier1_explained.md`](./reports/tier1_explained.md). If you only read
one file, read that one.

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
  `torch.cuda.Event`. `scripts/collect_measurements.py` runs a 530-op sweep and
  dumps JSON. `calibrate.py` runs the two-stage fit via `scipy.least_squares`
  on log-residuals.
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
# ~10 seconds for the full 530-op sweep.
```

Then either commit the JSON and update `MEASUREMENTS = ...` at the top of
`Makefile`, or pass it inline: `MEASUREMENTS=measurements/h100_sxm/run_<tag>.json make validate`.

## Reading order for a new engineer

1. **[`reports/tier1_explained.md`](./reports/tier1_explained.md)** — pedagogical walkthrough of MAPE, F, B, O, and the L2-caching artefact. *Start here.*
2. **[`reports/02_tier2.md`](./reports/02_tier2.md)** — Tier-2 refinements (L2 + per-family O + two-stage fit), including the wave-quant negative result.
3. **[`reports/03_whatif.md`](./reports/03_whatif.md)** — five counterfactual experiments and the overhead-bound finding.
4. **[`reports/figures/measured_vs_predicted.png`](./reports/figures/measured_vs_predicted.png)** — one-picture residual summary.
5. Then code, in dataflow order: `src/autoverse/ops.py` → `model.py` → `cost.py` → `simulator.py` for prediction; `measure.py` → `calibrate.py` for calibration.

For deeper context: [`CLAUDE.md`](./CLAUDE.md) has the day-by-day roadmap, pinned modelling decisions, and the "things to actively resist" list that scoped the sprint.

## License

MIT — see [`LICENSE`](./LICENSE).
