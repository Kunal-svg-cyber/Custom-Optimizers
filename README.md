# Coordinate-wise SVRG for low signal-to-noise objectives

A from-scratch variance-reduced optimizer with adaptive coordinate scaling, in two backends (PyTorch, JAX),
validated against an independent NumPy oracle, with a written theory note, a tuned ablation, and an
honest account of what the evidence does and does not show.

**Start here:** [`docs/THEORY.md`](docs/THEORY.md) (what is proved, cited, open) and [`docs/RESULTS.md`](docs/RESULTS.md) (experiments and limitations).

## Headline findings (synthetic least-squares testbed, see RESULTS.md for protocol and caveats)

| Finding | Evidence |
|---|---|
| With the same sample-gradient budget, SVRG variants converge to float64 round-off while tuned Adam / SGD-momentum (constant and cosine-decay lr) stall at 1e-5 to 1e-7 | Experiment 1, three noise regimes, held-out seeds |
| The gain comes from variance reduction, **not** from the adaptive scaling (SVRG + momentum matches Coordinate SVRG) | Experiment 1 ablation |
| Gradient variance falls ~1e15x along the trajectory while SGD variance stays flat, and stays under the proved bounds in 25/25 probes | Experiment 2, Lemma 2, Proposition 3 |
| No measurable benefit for walk-forward signal tracking at this problem size | Experiment 3 (null result, reported as such) |

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
python -m pytest tests/ -v                       # invariants + oracle + differential tests
python -m experiments.run_experiments            # regenerates results/ and docs/figures/ (~2 min, CPU)
python -m benchmarks.compare_optimizers --device cuda --wandb-mode online   # torch engine on a T4
```

`notebooks/colab_runner.ipynb` does all of this on a Colab T4 (edit `REPO_URL` first).

## Repository layout

```
configs/stochastic_regime.yaml    regime, noise sigma levels, seeds, optimizer knobs
src/environment.py                toxic market simulator + deterministic batch sampler
src/torch_optimizer.py            stateful PyTorch engine (flat contiguous memory, telemetry)
src/jax_optimizer.py              stateless JAX engine (PyTree state, jax.jit)
src/reference_numpy.py            independent NumPy oracle + 2x2 ablation switches
tests/test_math.py                invariant tests, theory checks, engine-vs-oracle differential tests
experiments/run_experiments.py    tuned ablation, variance decay, walk-forward (NumPy only)
benchmarks/compare_optimizers.py  torch engine vs Adam / SGD, logged to Weights & Biases
docs/THEORY.md                    lemmas with proofs, cited theorem, open questions
docs/RESULTS.md                   tables, figures, interpretation, limitations
results/                          raw JSON + tables from the last run
```

## Validation status

* Environment, sampler, NumPy oracle, and the theory checks (Lemmas 1 and 2, Propositions 3 and 4 behaviour) are run and passing.
* The PyTorch and JAX engines and their tests are **written to be validated against the oracle**; run `pytest` to confirm on your machine.
* All experiment numbers come from the NumPy oracle, not from the torch/JAX engines.

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
