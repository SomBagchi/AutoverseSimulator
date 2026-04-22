# CLAUDE.md — Autoverse Simulator

> Context for Claude Code sessions working on this repo. Read this first.

## Project purpose

This repo is Som's code sample for the **Anthropic Fellows program** application (ML Systems & Performance track). It's built in a **4-day sprint** (Tue Apr 21 – Sun Apr 26, 2026) on a **~$1k compute budget**. The artifact is intentionally scoped to fit that window.

**What Autoverse is:** an analytical (roofline++) performance model for transformer inference on GPU-like accelerators. Takes a `HardwareSpec` + a `TransformerConfig`, predicts per-op latency, calibrates to real H100 measurements, then runs "what-if" counterfactuals on hardware-spec changes.

**What it is not:** cycle-accurate; a training-time simulator; a multi-node simulator (stretch only); a general-purpose ML framework.

## Planning docs (read these for context beyond this file)

Located one directory up, at `../`:

- `01_evaluation_criteria.md` — rubric we're grading ourselves against.
- `02_project_proposals.md` — the 10 proposals we considered; Proposal 2 is this one.
- `03_autoverse_end_product.md` — **the spec.** What the end product contains, per-tier definitions of done, repo layout, pinned modeling choices, risk register.
- `04_autoverse_implementation_plan.md` — **the plan.** Day-by-day, checkpoint-by-checkpoint, with the 24 named checkpoints (1A–4F). Includes "things to actively resist" and failure-mode contingencies.

**If anything in this file conflicts with those, the spec (03) and plan (04) win.**

## Tier structure — where we are

Every tier is a complete, submittable artifact. The rule is **end-to-end first, then layer fidelity**.

- **Day 0 (Tue Apr 21) — scaffolding.** Interfaces sketched, CI green, uv venv set up, compute booked.
- **Tier 0 (Day 1, Wed Apr 22):** E2E skeleton. `python -m autoverse simulate --model llama1b --mode decode` prints a plausible number. No validation yet.
- **Tier 1 (Day 2, Thu Apr 23):** Validated against real H100. Calibration fits spec params; MAPE ≤ 30% on fitted set. First real report committed.
- **Tier 2 (Day 3, Fri Apr 24):** Fidelity refinements (wave quantization, overlap, L2 hits); held-out MAPE ≤ 20%.
- **Tier 3 (Day 4, Sat Apr 25):** 3–5 what-if experiments; findings report; README polish.
- **Submit (Sun Apr 26).**

Check current status: look at `README.md` "Status" section and the latest commit message.

## Conventions

### Dev environment

- **`uv` for everything.** `uv sync` to install; `uv run pytest` to test; `uv run ruff check` to lint.
- Python 3.11+. Pinned in `.python-version`.
- Never `pip install` directly. Add deps to `pyproject.toml` and `uv sync`.

### Core commands

```bash
uv sync                 # install + create venv
uv run pytest -q        # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy src/        # type-check
make test               # = uv run pytest -q
make lint               # = ruff + mypy
make validate           # Tier 1+: calibrate + generate validation report
make whatif             # Tier 3+: run counterfactual experiments
```

### Commit discipline

- **One commit per green checkpoint.** The 24 named checkpoints are in `../04_autoverse_implementation_plan.md`.
- Commit messages include the checkpoint number: `"1A: Op FLOP/byte accounting + tests"`.
- No "WIP" commits on `main`. If you need a checkpoint, branch.
- Every commit must pass `make test` and `make lint`. CI enforces this.
- Default branch is `main`, not `master`.

### Code style

- Type hints everywhere. `mypy --strict`-clean is the target.
- `ruff` does both lint and format.
- Line length: 100.
- Dataclasses are `frozen=True` unless there's a mutation reason.
- Prefer pure functions for cost-model code — easier to test in isolation.

### Testing

- Every op's FLOP and byte accounting has a unit test against textbook formulas.
- Every cost-model refinement has a before/after test showing the expected behavior change.
- No test depends on GPU availability. Measurement code imports `torch` but lives behind a guard so CI stays CPU-only.

## Pinned modeling decisions — do not revisit without cause

These were pinned in `03_autoverse_end_product.md` §8. Summary:

| Decision | Choice |
|---|---|
| Abstraction level | Analytical (roofline++) per op, not cycle-accurate. |
| Overlap model | Single scalar `alpha`: `t = max(t_c, t_m) + (1-alpha) * min(t_c, t_m)`, calibratable per op-family. |
| Attention | Two modes: materialized (O(n²) HBM) and flash-style tiled. |
| L2 hit rate | Simple heuristic: `hit_rate = min(1, L2_capacity / working_set)`. |
| Wave quantization | Per-GEMM, based on tile count vs SM count. |
| Precision | BF16 baseline. FP8 is stretch only. |
| Ops in scope | Embedding, RMSNorm, QKV+Out matmul, RoPE, Attention (prefill & decode), MLP (Gate+Up+Down), SiLU, LM head. Nothing else without a good reason. |
| Calibration optimizer | SciPy `least_squares` on log-latency. |

## Things to actively resist

From `../04_autoverse_implementation_plan.md`, copied for visibility:

1. **Premature op-graph generality.** Hardcode Llama-3.2-1B first. Generalize only if Tier 3 actually needs it.
2. **Calibration perfectionism.** 22% MAPE with 3 great what-ifs > 8% MAPE with no what-ifs.
3. **Cycle-level temptation.** If you feel like simulating warp scheduling, stop.
4. **Web UI creep.** Streamlit is a stretch for a reason.
5. **"Let me add TPU support too."** No. One calibrated spec (H100).
6. **Not committing.** Hour without a commit = something's wrong.

## Git & remote

- Local repo: `/Users/somsubhrobagchi/projects/Anthropic Fellows Application Project/autoverse-simulator`
- Remote: `https://github.com/SomBagchi/AutoverseSimulator.git`
- Default branch: `main`. Never rename to master.

## When stuck

1. Re-read the relevant section of `../04_autoverse_implementation_plan.md` — specifically the checkpoint you're on and its definition of done.
2. If the current checkpoint is stuck, ask: "is this checkpoint bigger than I thought, or am I overengineering?"
3. The failure-mode contingency table in `../04_autoverse_implementation_plan.md` §"Failure-mode contingencies" has responses for the most likely failure modes.
4. The fallback of last resort: ship the last green tier.
