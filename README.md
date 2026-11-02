# Coordinate-wise SVRG for low signal-to-noise objectives

A from-scratch variance-reduced optimizer with adaptive coordinate scaling, in two backends (PyTorch, JAX),
validated against an independent NumPy oracle, with a written theory note, a tuned ablation, and an
honest account of what the evidence does and does not show.

**Start here:** [`paper/one_page_summary.pdf`](paper/one_page_summary.pdf) (one page), [`paper/technical_report.pdf`](paper/technical_report.pdf) (10-page technical report), [`docs/CLAIMS.md`](docs/CLAIMS.md) (every claim, its evidence and its limits), [`docs/THEORY.md`](docs/THEORY.md) (what is proved, cited, open) and [`docs/RESULTS.md`](docs/RESULTS.md) (experiments and limitations).

## Headline findings (mostly synthetic testbeds, plus one real-data row; see `docs/RESULTS.md` for protocol, paired statistics and caveats)

| Finding | Evidence |
|---|---|
| On well-scaled convex problems, variance reduction dominates: SVRG variants reach float64 round-off and win against the best tuned Adam / SGD-momentum (constant and cosine-decay lr) on 20/20 (least squares) and 12/12 (logistic) held-out seeds in every noise regime | Experiments 1, 4 |
| Adaptive scaling is conditional: it helps on badly scaled features (1.5 decades better than SVRG + momentum, 95% CI [-2.0, -1.1], 20/20 seeds) and hurts on a larger well-scaled problem. With variance reduction and lr decay it beats tuned Adam + cosine by 5.3 decades there | Experiments 5, 6 |
| Non-convex network, learning rates tuned on training loss: SVRG reaches a sharper stationary point (1.3 decades smaller gradient norm, 8/8 seeds), but on held-out data the adaptive variants generalise worse (+0.0135 test loss, CI [+0.0106, +0.0150]); see Experiment 15 below for what that deficit is attributable to | Experiments 7, 8 |
| Held-out logistic regression: all methods are indistinguishable (test log-loss 0.68775 to 0.68776, ~54% accuracy) | Experiment 8 |
| Snapshot interval: broad optimum (any K from 16 to 256 within about 2x of the best); Proposition 3 holds at every K but is loose by 3 to 7 orders of magnitude | Experiment 9 |
| The conditioning numbers predict where adaptive scaling helps: Jacobi scaling cuts kappa from about 1.1e7 to about 82 on badly scaled features and not at all on equal-variance features (Proposition 5, a corollary of the Johnson-Zhang theorem) | Experiment 10 |
| **A method read off the theory**: Jacobi-preconditioned SVRG with hyper-parameters taken from the Johnson-Zhang recipe and no tuning reaches float64 round-off on 20/20 seeds of the badly scaled problem, where the best tuned adaptive variant reaches 3.5e-11 and needs 4.7x more evaluations to hit the target; measured contraction 0.08 to 0.10 per epoch against a guaranteed 0.5 | Experiment 11, Proposition 5 |
| The Jacobi result carries to logistic regression with badly scaled features: tuned Jacobi SVRG reaches round-off on 12/12 seeds with 2.9x fewer evaluations than Coordinate SVRG with cosine decay (constant-rate Coordinate SVRG does not plateau there, so that effect is problem-dependent) | Experiment 12 |
| Real data (breast cancer, wine, diabetes), raw features: preconditioning dominates but **no method reaches the target** (Hessian condition numbers 1e7). **Standardised, the round-off result reproduces** (SVRG variants at 1e-20 to 1e-31, best baselines 6e-8 to 2e-2): the shortfall was raw-scale conditioning, not 'dirty data' | Experiments 13, 16 |
| Non-convex network, validation-tuned learning rates: held-out deficit of Coordinate SVRG shrinks to +0.0038; SVRG + momentum ties the best baseline; Adam is worse than SGD, so the deficit tracks adaptive scaling, not variance reduction | Experiment 15 |
| Ablating invariants: misaligned batches are 2.1x worse than no variance reduction; textbook Adam fails 76% to 100% of adversarial float32 trials where the full guard set fails 0%; the gradient bound, not the denominator floor, is the operative guard | Experiment 14 |
| Real market data (10 ETFs, daily, 2008 to 2025): **no detectable tradable signal**; every net-of-cost Sharpe is negative; Jacobi SVRG reproduces OLS exactly and Adam is indistinguishable from it, so the optimizer does not matter; validated ridge beats OLS by +0.0101 IC (CI [+0.0012, +0.0188]) | `experiments/real_market_study.py` |
| GPU wall-clock (T4, float32): SVRG ties the best direct solve only at the largest size tested (d = 2048: 0.49 s vs 0.49 s) and loses at smaller sizes; the engine step costs about 1.6 ms even with telemetry off, so the 2 ms step was framework overhead, not host synchronisation | `benchmarks/timing_study.py` |
| Gradient variance falls about 1e15x along the trajectory while SGD variance stays flat, and stays under the proved bounds in 25/25 probes | Experiment 2, Lemma 2, Proposition 3 |
| No wall-clock win over a direct solve at `n = 40000`, `d = 200` (normal equations 0.06 s vs SVRG 0.08 s) | Experiment 5 |
| Walk-forward signal tracking: differences from OLS are at most 0.005 in IC, none favouring SVRG over Adam | Experiment 3 (null result for SVRG) |

## The four invariants

| # | Invariant | How it is enforced / verified |
|---|---|---|
| 1 | **Variance reduction**: `E||g_hat - grad F||^2 -> 0` | snapshot control variate; Lemma 2 + exact least-squares scaling identity (tested); Proposition 3 gives a `snapshot_interval` design rule |
| 2 | **Deterministic sample alignment** | one batch id / batch object feeds both the live and snapshot gradient; `DeterministicBatchSampler` |
| 3 | **Hardware contiguity** | torch: parameters re-pointed into one flat buffer, snapshot / mu / m / v in one `(4, N)` block |
| 4 | **Coordinate boundary safeguards** | sanitised + bounded gradients (the operative guard, Experiment 14), floored denominator, clipped ratio; Proposition 4 |

## Algorithm

```
g_hat = g_t(w_t) - g_t(w~) + mu~            # variance-reduced gradient (same batch for both terms)
m     = b1*m + (1-b1)*g_hat
v     = b2*v + (1-b2)*g_hat^2
w    <- w - lr * clip( m_hat / max(sqrt(v_hat), floor), +-update_clip )
```

## Quick start (Codespace / Colab)

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                       # invariants, oracle, differential and fuzz tests
python -m experiments.run_experiments            # regenerates results/ and docs/figures/ (~40 min, CPU)
python -m benchmarks.compare_optimizers --device cuda --wandb-mode disabled   # torch engine on a T4 (use online after `wandb login`)
```

`notebooks/colab_runner.ipynb` does all of this on a Colab T4 (upload the zip or clone your repo, then run the cells in order). `make test`, `make experiments`, `make report` wrap the same commands. A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the test suite on CPU PyTorch and JAX for every push, so the engine tests run automatically once the repository is on GitHub.

## Scope

This repository is an **offline research and calibration** project for linear and generalised-linear signal models on noisy data. It is **not** a production, latency-critical trading component, and no latency-sensitive deployment is claimed. A compiled C kernel (below) shows the algorithm itself is cheap per step, but the PyTorch and JAX engines, snapshot-based variance reduction and the research tooling around them are not what a microsecond trading loop would run.

## Deployment notes and known constraints

* **Memory.** The PyTorch engine keeps the flat parameters (N), a (4, N) state block (snapshot, full gradient, both moments) and a (3, N) scratch block: about **8N floats**, plus a transient gradient, versus roughly 4N for Adam including gradients. That is about twice Adam's footprint, which is trivial for linear models (even N = 10 million float32 parameters is 8N x 4 B = 320 MB) and only matters near a billion parameters. It is not intended for such models, and variance reduction is known to help little in deep learning anyway; the target regime is linear and generalised-linear models.
* **Per-step cost of the PyTorch engine (measured on a T4).** About **1.6 ms per step with telemetry off, 1.9 ms with `telemetry="lazy"` and 2.0 ms with the default eager telemetry**, independent of problem size from d = 256 to 2048 (`results/timing/report.md`). So host synchronisation from telemetry is a minor part (eager costs about 1.2x the telemetry-off step; lazy saves about 6%); the floor is per-step framework overhead (kernel launches and autograd for the two gradient evaluations). An earlier version of this README attributed the 2 ms step to host synchronisation; that was wrong. `telemetry="lazy"` keeps statistics on the device and reads them back only when `last_stats` is accessed; `telemetry=False` records nothing. The JAX engine has no host reads inside a step, and `make_svrg_scan` runs K steps in one compiled call.
* **Sample alignment.** The engine passes the same batch id to the live and snapshot evaluations, but the PyTorch closure must itself be a pure function of that id (no shuffling loader, dropout or stateful counter). `opt.verify_closure_determinism(closure)` checks this. The JAX engine receives the batch object itself, so it cannot misalign.
* **Latency of the algorithm itself.** `python -m benchmarks.latency_kernel` compiles `benchmarks/svrg_kernel.c` (a fourth implementation, checked against the NumPy oracle to 3e-16) and measures nanoseconds per operation on one CPU core. At d = 32 on two machines: signal-inference dot product 15 ns (development sandbox) and 20 ns (Colab); SVRG single-sample step 129 ns and 217 ns; 64-sample batch 3.9 µs and 6.9 µs; recursive least squares update 0.96 µs and 1.5 µs; one snapshot pass over a 100,000-row window 1.8 ms and 3.1 ms (`results/latency.md`, `results/latency_colab.md`). The 2 ms GPU step is framework overhead, not the algorithm. Model fitting and signal inference are different paths: inference is a dot product; a refit on a trailing window can run off the critical path. These are machine-specific numbers on shared hardware, and RLS is exact per tick, so none of this is a claim that SVRG is the better online estimator.
* **GPU wall-clock against direct solves (measured on a T4, float32, Gaussian features, fixed untuned hyper-parameters).** Jacobi SVRG (the PyTorch engine) reached the target in 3 epochs at every size and took 1.91 s, 0.94 s and 0.49 s at (n, d) = (100k, 256), (200k, 1024), (200k, 2048). Direct normal equations took 0.16 s, 0.15 s and 0.49 s: **faster at the two smaller sizes and tied at d = 2048**, where `lstsq` was 3.6x slower than SVRG and used 3.3 GB of extra memory against 2 MB. Untuned Adam did not reach the target in 60 epochs at any size. The crossover with the best direct method is at about d = 2048 on this hardware; larger sizes, badly conditioned problems and tuned hyper-parameters were not tested (`results/timing/report.md`).

## Real-market study (run by the repository owner on Colab)

`experiments/real_market_study.py` runs a purged walk-forward comparison of OLS, ridge, budget-limited Adam, budget-limited Jacobi SVRG and 21-day momentum on daily returns of ten ETFs (Yahoo Finance, 2008 to 2025; 3,749 out-of-sample days; nine causal features; 756-day window refitted every 21 days; dollar-neutral long-short strategy; 5 bps cost per unit traded). Safeguards: features use only past data and refits train only on rows whose targets are already known (both unit-tested), and a placebo run on circularly shifted targets shows what noise looks like. Full output: `results/real_market/report.md`.

**Result: no detectable tradable signal, and no effect of the optimizer.**
* The mean daily cross-sectional rank IC of the fitted models (-0.0022 to +0.0079) lies inside the placebo's range (-0.0091 to +0.0090). The pooled IC is weakly positive (+0.019 for OLS, +0.029 for ridge), but the net-of-cost Sharpe ratio is negative for every fitted model (best -0.76; OLS -1.21 at a turnover of 1.09 per day).
* Budget-limited Jacobi SVRG reproduces OLS to four decimals (IC difference +0.0000, 95% CI [-0.0000, +0.0001]) and budget-limited Adam is statistically indistinguishable from it (+0.0004, CI [-0.0078, +0.0087]): on this small, well-conditioned problem the iterative methods converge to the closed-form solution.
* **Regularisation, not optimisation, is what mattered:** validated ridge beat OLS by +0.0101 in mean daily IC (CI [+0.0012, +0.0188]) and by +0.45 in net Sharpe (CI [+0.20, +0.69]), though every net Sharpe remains negative.
* Caveats: ten ETFs, daily data, simple features, a flat cost assumption, one placebo draw (so the placebo range is itself uncertain), no multiple-comparison correction. This supports no claim of a tradable edge and says nothing about other assets or frequencies.

```bash
pip install yfinance
python -m experiments.real_market_study --tickers SPY QQQ IWM EFA EEM TLT GLD XLF XLE XLK --start 2008-01-01 --end 2025-12-31 --placebo
```

## Repository layout

```
configs/stochastic_regime.yaml    regime, noise sigma levels, seeds, optimizer knobs
src/environment.py                toxic market simulator + deterministic batch sampler
src/torch_optimizer.py            stateful PyTorch engine (flat contiguous memory, telemetry)
src/jax_optimizer.py              stateless JAX engine (PyTree state, jax.jit)
src/reference_numpy.py            independent NumPy oracle + 2x2 ablation switches + frozen preconditioner
src/preconditioning.py            Jacobi diagonals for least squares / logistic, Johnson-Zhang recipe
tests/test_math.py                invariant tests, theory checks, engine-vs-oracle differential tests
experiments/real_market_study.py  walk-forward study on real or synthetic returns (placebo, block bootstrap)
experiments/run_experiments.py    sixteen experiments with paired-bootstrap statistics (NumPy; Experiments 13 and 16 need scikit-learn)
benchmarks/compare_optimizers.py  torch engine vs Adam / SGD, logged to Weights & Biases
benchmarks/timing_study.py        GPU wall-clock, memory and telemetry-overhead study (run on a Colab T4; output in results/timing/)
benchmarks/latency_kernel.py      compiled-C per-operation latency (SVRG step, RLS, dot product), checked against the oracle
benchmarks/svrg_kernel.c          the C kernels
paper/technical_report.tex/.pdf   10-page report; every number is a macro generated from results/experiments.json
paper/one_page_summary.tex/.pdf   one-page summary (same generated numbers)
paper/make_numbers.py             results JSON -> LaTeX macros and tables (--check verifies sync)
docs/CLAIMS.md                    claim -> kind of support -> evidence -> limits
docs/REVIEWER_FAQ.md              fifteen sceptical questions with honest answers
docs/AUDIT_RESPONSE.md            point-by-point response to an external critique: fair, overstated, or wrong, and what was done
CHANGELOG.md                      additions and, above all, corrections
results/ENVIRONMENT.txt           software versions that produced results/
docs/THEORY.md                    lemmas with proofs, cited theorem, open questions
docs/RESULTS.md                   tables, figures, interpretation, limitations
results/                          experiments.json + tables from the last run; latency.{json,md} and latency_colab.*;
                                  real_market/report.md and timing/ (outputs of the scripts run on Colab)
```

## Validation status

* **Google Colab (T4 GPU, torch 2.11), run by the repository owner:** the **full current suite, 68 tests, passed** (135 s). That includes every PyTorch and JAX engine test, the step-for-step differential tests against the NumPy oracle, the frozen-preconditioner tests, the fuzz tests, and the newest tests (lazy telemetry, the JAX scan, the closure-determinism check, the compiled C kernel).
* **GPU engine against the oracle (earlier run):** the PyTorch engine on the T4 stayed within **1.8e-7** of the NumPy reference over 100 steps (float32); a 640-step benchmark reproduced the oracle's final SVRG and Adam numbers to four significant digits.
* **Compiled C kernel:** reproduces the oracle to 3.3e-16 on both the development machine and Colab.
* An earlier GitHub Actions run (Python 3.10 log) passed 44 of 45 tests; the one failure was a wrong assertion in a fuzz test, since corrected. The state of the latest CI run is not recorded here; check the Actions tab.
* All experiment numbers in `docs/RESULTS.md` come from the NumPy oracle, not from the torch/JAX engines. Timing results come from separate scripts run on Colab (see below).

## PyTorch usage

```python
from src.torch_optimizer import CoordinateSVRG

opt = CoordinateSVRG([w], lr=0.02, snapshot_interval=64)
for _ in range(steps):
    if opt.needs_snapshot:
        opt.refresh_snapshot(chunk_closure, num_chunks=8)   # mean loss over chunk `id`
    opt.step(batch_closure)                                  # loss for batch `id`; called twice per step
    # wandb.log(opt.last_stats)   # step_us, snapshot_distance, correction_norm, ...
```

Frozen diagonal preconditioner (Proposition 5, the Jacobi variant; use `betas=(0.0, 0.999)` for the exact theorem):

```python
from src.preconditioning import jacobi_diag_least_squares   # or jacobi_diag_logistic(X, l2)

opt = CoordinateSVRG([w], lr=0.05, betas=(0.0, 0.999), snapshot_interval=64)
opt.set_frozen_preconditioner(torch.as_tensor(jacobi_diag_least_squares(X_numpy)))   # step = lr * m_hat / D
```

In JAX: `make_svrg_step(loss_fn, config, preconditioner=jnp.asarray(diag))`.

Closures take an integer id and must be deterministic in it, otherwise the alignment invariant cannot hold.
Build the optimizer after the model is on its final device/dtype and mutate parameters in place.

## JAX usage

```python
from src.jax_optimizer import SVRGConfig, init_state, make_svrg_step, compute_full_gradient, refresh_snapshot

config = SVRGConfig(lr=0.02, snapshot_interval=64)
step = make_svrg_step(loss_fn, config)            # jitted pure function
state = init_state(params)
state = refresh_snapshot(params, state, compute_full_gradient(loss_fn, params, chunk_stack))
params, state, stats = step(params, state, batch)
```

## References

See `docs/THEORY.md` (Johnson & Zhang 2013; Kingma & Ba 2015; Reddi et al. 2016, 2018; Defazio & Bottou 2019; and others).
