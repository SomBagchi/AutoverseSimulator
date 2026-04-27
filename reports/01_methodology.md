# Calibration methodology — what F, B, O, and MAPE mean

> A pedagogical walkthrough of how the simulator is calibrated against
> real-hardware measurements: what each fitted parameter represents,
> what loss we minimise, and what the headline numbers actually say
> about an H100. If you're new to the codebase, read this first.

## What we're doing

The simulator predicts per-op latency given a `HardwareSpec`. But every
number in the vendor `HardwareSpec` is a *nominal peak* — what the chip
would do under ideal conditions, not what cuBLAS and the runtime
actually deliver in practice. **Calibration replaces the vendor
nominals with measured-effective values**, so that `simulate(model, hw)`
predicts something close to wall-clock reality.

The pipeline:

```
                                    ┌──────────────────────────────┐
        ┌─────────────────┐         │ scripts/collect_measurements │
        │ HardwareSpec    │         │   (runs on H100, 530 ops)    │
        │ (vendor peaks)  │         └──────────────┬───────────────┘
        └────────┬────────┘                        │ measured_ms per op
                 │                                 ▼
                 │                       ┌──────────────────┐
                 ├──────────────────────►│ calibrate.py     │
                 │                       │ (least-squares)  │
                 │                       └────────┬─────────┘
                 │                                │ fitted (F, B, O)
                 │                                ▼
                 │                       ┌──────────────────┐
                 └──────────────────────►│ HardwareSpec     │
                                         │ (calibrated)     │
                                         └──────────────────┘
```

So the deliverable is a small set of effective parameters and a single
accuracy headline (**MAPE**). Once those are good, every downstream
"what-if" experiment inherits the calibration.

## What is MAPE?

**MAPE = Mean Absolute Percentage Error.** For each measurement `m_i`
and the model's prediction `p_i`:

```
            1     n   |p_i − m_i|
MAPE   =   ───   ∑   ─────────────
            n   i=1     m_i
```

Read in plain English: *"on average, how far off (as a percentage of the
true value) are the predictions?"*

- 0 % MAPE = the model nails every measurement exactly.
- 20 % MAPE = on average, predictions are off by 20 % of the measured value
  (in either direction).
- 100 %+ MAPE = the model is barely better than guessing.

Why MAPE rather than RMSE or MSE? Because latencies in this dataset span
**three orders of magnitude** (≈ 0.012 ms to ≈ 0.82 ms). An absolute-error
loss would let one big-op error dwarf hundreds of small-op errors. We
care about *relative* accuracy — getting a tiny op wrong by 0.01 ms is a
~50 % miss; getting the LM head wrong by 0.01 ms is < 1 %. MAPE encodes
that asymmetry directly.

We report two MAPEs:

- **MAPE on the fit set** (70 % of measurements). The optimiser has seen
  these — they're how it picked F, B, O. Low MAPE here just means the
  model can *fit* the data.
- **MAPE on the held-out set** (30 %). The optimiser has *not* seen these.
  This is the honest accuracy number — it shows how the model
  *generalises* to ops it wasn't trained on.

## What are F, B, and O?

Three scalars in our roofline. Predicted latency for op `i`:

```
                ┌  flops_i      bytes_i  ┐                 ms
   p_i  =  max  │ ─────────  ,  ──────── │   +   O · 10⁻³
                └  F · 10¹²    B · 10⁹  ┘
                  ──────────  ──────────
                  t_compute    t_memory
                  (seconds)    (seconds)
```

| symbol | unit | physical meaning |
|---|---|---|
| **F** | TFLOPs/s | Effective BF16 tensor-core throughput. Sets `t_compute = flops / F`. |
| **B** | GB/s | Effective HBM bandwidth. Sets `t_memory = bytes / B`. |
| **O** | µs | Per-op launch overhead — the cost of dispatching *any* kernel, no matter how trivial. Added once per op. |

For Llama-1B on H100 these would naively be the vendor numbers from
`HardwareSpec`: F = 989, B = 3350, O = 0. **Calibration replaces them
with whatever values make the model best match what the chip actually
does**, so that downstream simulations reflect deliverable performance,
not specsheet performance.

Two further modelling features extend this:

- **Per-op-family overhead.** A single global `O` can't capture the
  4-5× spread between cheap ops (Residual, ≈ 12 µs launch) and expensive
  ones (RoPE, ≈ 52 µs). We fit one `O` per op family — 8 scalars
  instead of 1. `predict_ms` looks up the right one by `type(op).__name__`.
- **L2 hit-rate heuristic.** Inputs that fit in L2 don't need to be
  re-read from HBM on every iter; the model's `bytes_read / B` term
  over-counts. Multiplier `(1 − hit_rate)` applied to inputs only;
  output writes always stream to HBM. Detail and motivation in
  "Why fitted numbers initially exceeded vendor — and how we fixed it"
  below.

So the calibrated parameter set is `(F, B, {O_family})` — 10 scalars
total, fitted by `scipy.least_squares`.

## What the optimiser actually does

Given 530 `(op, measured_ms)` pairs:

```python
loss(F, B, O_family) = Σ ( log p_i(F, B, O_family[i]) − log m_i )²
```

SciPy's `least_squares` (Trust-Region Reflective with positivity bounds)
finds the parameters that minimise this loss. **Log-space** because we
care about relative error — `log(p) − log(m) = log(p/m)`, so the loss
penalises ratios, which matches MAPE's relative-error notion.

The split: 70 % of ops are used for the fit, 30 % held out. The split is
deterministic (seeded shuffle). Held-out MAPE is the headline.

**One subtlety: a two-stage fit.** The roofline's `max(t_c, t_m)` makes
∂t/∂F = 0 on memory-bound ops, so F is **under-constrained** by ~75 %
of the dataset. A naive joint fit drifts F to unphysical values. We fix
this by fitting in two stages: first F on the compute-bound subset
(arithmetic intensity > 295 FLOP/byte ≈ H100 ridge), then B and
per-family O on the full dataset with F frozen. Detail in "F was
under-constrained — and how we fixed it" below.

## Headline result

```
  peak_bf16_tflops     :  943.21    (vendor 989, ratio 0.95)
  hbm_gbps             : 2285.55    (vendor 3350, ratio 0.68)
  per_op_overhead_us   : <per-family>
    RoPE                52.13 µs    (highest — index-arithmetic kernel)
    AttentionDecode     22.62 µs
    AttentionPrefill    20.33 µs
    SiLUGate            18.89 µs
    MatMul              17.06 µs
    RMSNorm             14.63 µs
    Embedding           13.37 µs
    Residual            11.55 µs    (lowest — single elementwise add)

  MAPE  fit            :   9.55 %   (n=371)
  MAPE  held-out       :  10.14 %   (n=159)
```

Both fitted throughputs are below vendor — physical, as they should be.
The 4-5× spread in per-family overhead matches what one expects from
PyTorch's kernel-launch costs across these primitives.

## Why fitted numbers initially exceeded vendor — and how we fixed it

Getting to the headline above wasn't a one-shot fit. The first model —
plain roofline + a single global `O`, no L2 heuristic — gave:

```
  peak_bf16_tflops    : 1138.51   (vendor 989, ratio 1.15)  ⚠ above peak
  hbm_gbps            : 5374.8    (vendor 3350, ratio 1.60) ⚠ above peak
  MAPE held-out       :   20.2 %
```

**Both fitted throughputs above vendor peaks.** *You cannot run faster
than the chip can run.* So why did the fit say we were?

**Short answer: it wasn't the chip exceeding peak; it was our *model*
over-counting bytes, and the optimiser compensating.**

Long answer in four steps.

### Step 1: how the measurement actually works

When we measure the latency of an op, we don't do it once. We:

1. Allocate the input tensors **once** on the GPU.
2. Run the op **10 times** to warm up (compile, JIT, populate caches).
3. Run the op **100 more times** with `cudaEventRecord` brackets,
   capturing per-iter latency.
4. Take the **median**.

Crucially: we use the **same input tensors** for all 110 iters. We never
reallocate, never zero-out caches between iters.

### Step 2: where the data lives between iters

H100 has a **50 MB L2 cache**, which is *huge* by historical standards.
After iter 1, the input tensors are in L2 (or part of them are, if they
exceed 50 MB). Iter 2 doesn't have to fetch them from HBM again — the
loads are served from L2.

So:
- **Iter 1** pays the full HBM round-trip.
- **Iters 2–100** mostly hit L2 for inputs.

The **median** across all 100 iters is dominated by the warm-cache time,
not the cold-cache time. That's actually what we want for predicting
real-world inference latency *within* a layer or *within* a token, where
caches similarly stay warm. But it means our *measurements* reflect a
different cost than our *model* assumes.

### Step 3: where our model is over-counting

`Op.bytes_read()` and `Op.bytes_written()` assume **every byte goes to
HBM**, every time. For a `MatMul(M, K, N)` in BF16:

```python
bytes_read    = (M·K  +  K·N) · 2
bytes_written =  M·N · 2
```

But on iter 50, the activation `(M, K)` and weight `(K, N)` tensors are
sitting in L2. The actual HBM traffic on that iter is *much* smaller.

So our model claims "this op moves 30 MB from HBM" but the chip actually
moved maybe 5 MB. To make `t_memory = bytes / B = 30 MB / B` come out
equal to the small measured time, the optimiser must inflate `B`. The
math is forced.

### Step 4: empirical proof

If the L2-caching story is correct, we'd predict: **calibrating only on
ops too large to fit in L2 should give B ≤ vendor**, because there's no
cache effect for the optimiser to absorb.

Done. The numbers:

|                                | n   | fitted F (TFLOPs) | fitted B (GB/s) | held-out MAPE |
|---|---|---|---|---|
| **All 530 ops** (full fit)     | 530 | 1138 (1.15× vendor) | 5375 (1.60× vendor) | 20.2 % |
| **Subset: working set > L2**   |  54 |  797 (0.81× vendor) | 3111 (0.93× vendor) |  3.4 % |
| **Subset: working set < 5 MB** | 162 |  989 (= vendor)*    | 3350 (= vendor)*    | 25.4 % |

*\* On the 5 MB subset, the optimiser doesn't move from its initial
guess (989, 3350) because every op is overhead-dominated — neither F nor
B affects predicted time, so the gradient is zero. That row tells us
nothing about F or B; it tells us the small-op MAPE is bad because a
single global O can't capture per-op-family overhead variation.*

The middle row is the clincher:

- **F = 797 TFLOPs** is 81 % of vendor 989 — exactly the realistic cuBLAS
  efficiency we already saw on the 4096³ GEMM (78 %).
- **B = 3111 GB/s** is 93 % of vendor 3350 — typical effective HBM3
  performance with normal access patterns.
- **MAPE 3.4 %** says the roofline + global overhead nails these big ops
  to within a few percent.

So **the chip behaves exactly as physics requires** — 81 % of compute
peak, 93 % of bandwidth peak. The full-dataset fit's above-vendor
numbers are not the chip overachieving; they are the optimiser papering
over an L2-shaped hole in our model.

### What we did about it

The fix was a one-line `cost.l2_hit_rate(input_bytes, l2_mb)`:

```python
hit_rate = min(1, L2_capacity / bytes_read)        # H100 L2 = 50 MB
effective_bytes = bytes_written + bytes_read · (1 − hit_rate)
t_memory = effective_bytes / B
```

Hit rate is computed on `bytes_read` only, not the full working set —
caches reduce *re-reads*; first-time output writes always stream to
HBM. With this in place, the optimiser no longer needs to inflate B to
absorb the L2 effect; B falls from 5375 to 2286 GB/s (0.68× vendor) —
physically realistic.

## F was under-constrained — and how we fixed it

After adding the L2 heuristic, B was physical but **F was still above
vendor** (1172 TFLOPs, 1.18× peak). The cause is structural in the
roofline equation:

For an op that's compute-bound (`flops/F > bytes/B`), the prediction is
`flops/F + O` and `∂(predicted)/∂B = 0` — that op tells the optimiser
nothing about B.

For an op that's memory-bound (the converse), the prediction is
`bytes/B + O` and `∂(predicted)/∂F = 0` — that op tells the optimiser
nothing about F.

In our 530-op sweep, only ~25 % of ops are compute-bound. The other
~75 % carry zero gradient on F: they don't constrain it. The joint
fit's F then drifts along the unconstrained direction, settling
unphysically high because of small numerical couplings via the shared
per-family overheads.

The fix is a **two-stage fit** (`autoverse.calibrate.calibrate_two_stage`):

1. **Stage 1 — compute-bound subset only.** Filter to ops with
   arithmetic intensity > 295 FLOP/byte (the H100 ridge point). On this
   subset, every op constrains F, so F is well-conditioned.
2. **Stage 2 — full dataset, F frozen at the stage-1 value.** Fit B and
   per-family overhead from data that actually constrains them.

After this: F = 943 TFLOPs (0.95× vendor), B = 2286 GB/s (0.68× vendor)
— both physical. We pay 0.7 percentage points of MAPE for this (10.1 %
vs 9.4 % from the joint fit), which is worth it: counterfactual
experiments need `B` to mean what it says.

## Per-op MAPE breakdown

Held-out MAPE on the final calibrated model, worst-first:

| op family | MAPE | n |
|---|---|---|
| AttentionPrefill | 14.0 % | 4 |
| SiLUGate | 12.4 % | 11 |
| MatMul | 11.7 % | 90 |
| RMSNorm | 8.4 % | 21 |
| Residual | 7.9 % | 23 |
| AttentionDecode | 1.7 % | 6 |
| RoPE | 1.0 % | 4 |
| Embedding | 1.0 % | — |

Compare this with the very first single-global-O fit, where RoPE was at
65.5 % and Residual at 48.2 %. Per-family overhead alone cut those by
60×+. The remaining double-digit miss on AttentionPrefill / SiLUGate /
MatMul reflects shape-dependent cuBLAS efficiency — the model doesn't
have a per-shape efficiency factor. Adding wave quantisation was tried
and rejected (worsens MAPE; see `02_refinements.md`).

## TL;DR

- 530 H100 measurements, calibrated to **MAPE 10.1 %** held-out.
- Three modelling features in the cost equation: roofline,
  per-op-family launch overhead, L2 hit-rate heuristic on inputs.
- Two-stage least-squares fit so that F (set by compute-bound ops only)
  doesn't get pulled unphysically high by the memory-bound majority.
- Both fitted throughputs are below vendor — F = 943 TFLOPs (0.95×),
  B = 2286 GB/s (0.68×) — physically meaningful.
- Big ops (MatMul, AttentionPrefill) at 10–14 %; small ops at 1–8 %.
  Largest residual is the prefill LM head, a tall-skinny GEMM where
  cuBLAS picks a sub-peak algorithm — expected, documented, accepted.
