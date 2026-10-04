# Coordinate-wise SVRG for low signal-to-noise objectives

A from-scratch variance-reduced optimizer with adaptive coordinate scaling, in two backends (PyTorch, JAX),
validated against an independent NumPy oracle, with a written theory note, a tuned ablation, and an
honest account of what the evidence does and does not show.

**Start here:** [`paper/technical_report.pdf`](paper/technical_report.pdf) (8-page technical report), [`docs/THEORY.md`](docs/THEORY.md) (what is proved, cited, open) and [`docs/RESULTS.md`](docs/RESULTS.md) (experiments and limitations).

## Headline findings (synthetic testbeds; see `docs/RESULTS.md` for protocol, paired statistics and caveats)

| Finding | Evidence |
|---|---|
| On well-scaled convex problems, variance reduction dominates: SVRG variants reach float64 round-off and win against the best tuned Adam / SGD-momentum (constant and cosine-decay lr) on 20/20 (least squares) and 12/12 (logistic) held-out seeds in every noise regime | Experiments 1, 4 |
| Adaptive scaling is conditional: it helps on badly scaled features (1.5 decades better than SVRG + momentum, 95% CI [-2.0, -1.1], 20/20 seeds) and hurts on a larger well-scaled problem. With variance reduction and lr decay it beats tuned Adam + cosine by 5.3 decades there | Experiments 5, 6 |
| Non-convex network: SVRG reaches a sharper stationary point (1.3 decades smaller gradient norm, 8/8 seeds), but on held-out data the adaptive variants generalise *worse* (+0.0135 test loss, CI [+0.0106, +0.0150]); optimisation precision is not the lever on a noisy fit | Experiments 7, 8 |
| Held-out logistic regression: all methods are indistinguishable (test log-loss 0.68775 to 0.68776, ~54% accuracy) | Experiment 8 |
| Snapshot interval: broad optimum (any K from 16 to 256 within about 2x of the best); Proposition 3 holds at every K but is loose by 3 to 7 orders of magnitude | Experiment 9 |
| The conditioning numbers predict where adaptive scaling helps: Jacobi scaling cuts kappa from about 1.1e7 to about 82 on badly scaled features and not at all on equal-variance features (Proposition 5, a corollary of the Johnson-Zhang theorem) | Experiment 10 |
| **A method read off the theory**: Jacobi-preconditioned SVRG with hyper-parameters taken from the Johnson-Zhang recipe and no tuning reaches float64 round-off on 20/20 seeds of the badly scaled problem, where the best tuned adaptive variant reaches 3.5e-11 and needs 4.7x more evaluations to hit the target; measured contraction 0.08 to 0.10 per epoch against a guaranteed 0.5 | Experiment 11, Proposition 5 |
| The Jacobi result carries to logistic regression with badly scaled features: tuned Jacobi SVRG reaches round-off on 12/12 seeds with 2.9x fewer evaluations than Coordinate SVRG with cosine decay (constant-rate Coordinate SVRG does not plateau there, so that effect is problem-dependent) | Experiment 12 |
| Gradient variance falls about 1e15x along the trajectory while SGD variance stays flat, and stays under the proved bounds in 25/25 probes | Experiment 2, Lemma 2, Proposition 3 |
| No wall-clock win over a direct solve at `n = 40000`, `d = 200` (normal equations 0.06 s vs SVRG 0.08 s) | Experiment 5 |
| Walk-forward signal tracking: differences from OLS are at most 0.005 in IC, none favouring SVRG over Adam | Experiment 3 (null result for SVRG) |

## The four invariants

| # | Invariant | How it is enforced / verified |
|---|---|---|
| 1 | **Variance reduction**: `E||g_hat - grad F||^2 -> 0` | snapshot control variate; Lemma 2 + exact least-squares scaling identity (tested); Proposition 3 gives a `snapshot_interval` design rule |
| 2 | **Deterministic sample alignment** | one batch id / batch object feeds both the live and snapshot gradient; `DeterministicBatchSampler` |
| 3 | **Hardware contiguity** | torch: parameters re-pointed into one flat buffer, snapshot / mu / m / v in one `(4, N)` block |
| 4 | **Coordinate boundary safeguards** | sanitised + bounded gradients, clamped (not offset) denominator, clipped ratio; Proposition 4 |

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
python -m experiments.run_experiments            # regenerates results/ and docs/figures/ (~20 min, CPU)
python -m benchmarks.compare_optimizers --device cuda --wandb-mode online   # torch engine on a T4
```

`notebooks/colab_runner.ipynb` does all of this on a Colab T4 (edit `REPO_URL` first). `make test`, `make experiments`, `make report` wrap the same commands. A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the test suite on CPU PyTorch and JAX for every push, so the engine tests run automatically once the repository is on GitHub.

## Repository layout

```
configs/stochastic_regime.yaml    regime, noise sigma levels, seeds, optimizer knobs
src/environment.py                toxic market simulator + deterministic batch sampler
src/torch_optimizer.py            stateful PyTorch engine (flat contiguous memory, telemetry)
src/jax_optimizer.py              stateless JAX engine (PyTree state, jax.jit)
src/reference_numpy.py            independent NumPy oracle + 2x2 ablation switches + frozen preconditioner
src/preconditioning.py            Jacobi diagonals for least squares / logistic, Johnson-Zhang recipe
tests/test_math.py                invariant tests, theory checks, engine-vs-oracle differential tests
experiments/run_experiments.py    twelve experiments with paired-bootstrap statistics (NumPy only)
benchmarks/compare_optimizers.py  torch engine vs Adam / SGD, logged to Weights & Biases
paper/technical_report.tex/.pdf   8-page report; every number is a macro generated from results/experiments.json
paper/make_numbers.py             results JSON -> LaTeX macros and tables
docs/THEORY.md                    lemmas with proofs, cited theorem, open questions
docs/RESULTS.md                   tables, figures, interpretation, limitations
results/                          raw JSON + tables from the last run
```

## Validation status

* The first GitHub Actions run (CPU PyTorch and JAX, Python 3.10 log) passed 44 of 45 tests: the environment and oracle tests, **all PyTorch engine tests** (flat contiguity, sample alignment, checkpoint round-trip, variance law, convergence), **all JAX engine tests**, the torch-vs-JAX parity test, and the **step-for-step differential tests of both engines against the NumPy oracle**.
* The one failure was a wrong assertion in the torch fuzz test (it required the stored snapshot gradient, which holds its inputs verbatim, to be finite; the engine guarantees finiteness of the weights and moments, which the test also checks and which passed). The assertion is corrected in this version. Re-run CI to confirm 45 of 45 on both Python versions.
* **Added after that CI run, not yet confirmed by CI:** a frozen-diagonal-preconditioner mode in both engines (`set_frozen_preconditioner` in PyTorch, `preconditioner=` in JAX) with differential tests against the oracle, plus tests for `src/preconditioning.py`. The default (adaptive) behaviour is unchanged. Push and check the Actions tab; the new tests are the ones to watch.
* All experiment numbers come from the NumPy oracle, not from the torch/JAX engines. The engines have been validated against the oracle on CPU; they have not yet been benchmarked on a GPU.

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
