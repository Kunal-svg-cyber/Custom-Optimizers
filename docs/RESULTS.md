# Results

All numbers below are produced by `python -m experiments.run_experiments` (about two minutes on one CPU core,
NumPy only) and are stored in `results/experiments.json`. They use the NumPy reference implementation
(`src/reference_numpy.py`), which the PyTorch and JAX engines are tested against step for step.
**The PyTorch/JAX engines themselves are not what produced these numbers.** Figures are in `docs/figures/`.

Protocol: every method gets its own learning-rate sweep (13 log-spaced values from 1e-5 to 10^-0.5) on
*tuning* seeds, and is then scored on *disjoint* evaluation seeds. Budgets are counted in sample-gradient
evaluations; SVRG is charged for its double gradients and for every full-gradient snapshot pass. Baselines
include a cosine-decay learning-rate schedule, because a constant-step baseline is a weak comparison for
a method whose selling point is convergence with a constant step.

![ablation](figures/ablation_curves.png)

*(The curves bottom out at 1e-18 only because the plot clamps values there; the underlying gaps are about 1e-31, i.e. float64 round-off.)*

![variance](figures/variance_decay.png)

![walk-forward](figures/walk_forward_ic.png)

### Experiment 1: tuned 2x2 ablation

Budget: 100 epochs of sample-gradient evaluations. Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; 'evals to target' = sample-gradient evaluations to reach 1e-08 x the initial gap.

**calm** (noise sigma = 0.25)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 5.66e-06 [5.15e-06, 8.24e-06] | not reached |
| SGD + momentum, cosine lr | 4.22e-03 | 8.35e-08 [6.90e-08, 1.17e-07] | not reached |
| Adam (clamped) | 3.16e-04 | 4.05e-06 [3.98e-06, 4.10e-06] | not reached |
| Adam (clamped), cosine lr | 7.50e-04 | 8.83e-08 [6.88e-08, 1.50e-07] | not reached |
| SVRG + momentum | 3.16e-01 † | 3.29e-31 [2.16e-31, 5.65e-31] | 92,800 |
| Coordinate SVRG (ours) | 1.78e-03 | 3.27e-31 [2.17e-31, 6.00e-31] | 85,376 |

**volatile** (noise sigma = 1.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 7.96e-06 [7.12e-06, 1.09e-05] | not reached |
| SGD + momentum, cosine lr | 4.22e-03 | 9.42e-08 [9.20e-08, 1.27e-07] | not reached |
| Adam (clamped) | 3.16e-04 | 4.35e-06 [4.01e-06, 4.63e-06] | not reached |
| Adam (clamped), cosine lr | 7.50e-04 | 9.72e-08 [8.94e-08, 1.75e-07] | not reached |
| SVRG + momentum | 1.33e-01 | 3.48e-31 [2.47e-31, 5.56e-31] | 72,448 |
| Coordinate SVRG (ours) | 1.78e-03 | 3.83e-31 [2.62e-31, 5.43e-31] | 85,376 |

**toxic** (noise sigma = 3.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 7.50e-04 | 2.04e-05 [1.89e-05, 2.64e-05] | not reached |
| SGD + momentum, cosine lr | 4.22e-03 | 2.76e-07 [2.57e-07, 3.30e-07] | not reached |
| Adam (clamped) | 7.50e-04 | 2.31e-05 [2.25e-05, 2.58e-05] | not reached |
| Adam (clamped), cosine lr | 1.78e-03 | 2.68e-07 [2.54e-07, 3.27e-07] | not reached |
| SVRG + momentum | 1.33e-01 | 3.58e-31 [3.37e-31, 6.25e-31] | 72,448 |
| Coordinate SVRG (ours) | 4.22e-03 | 5.15e-31 [3.85e-31, 6.59e-31] | 61,440 |

† best learning rate sat on the edge of the sweep grid (the true optimum may lie outside it).

### Experiment 2: variance along the trajectory

Probes taken at maximum snapshot staleness (every 64 steps). Measured variance fell within the Lemma 2 bound in 100% of probes and within the Proposition 3 bound in 100%.

| step | Var(g_hat) | Var(g_sgd) | ratio sgd/vr | snapshot distance | gap |
|---|---|---|---|---|---|
| 63 | 2.55e-01 | 1.78e+00 | 7.0x | 7.05e-01 | 1.17e-02 |
| 447 | 8.83e-08 | 1.89e+00 | 21,380,259.1x | 3.95e-04 | 1.26e-08 |
| 831 | 2.01e-11 | 2.30e+00 | 114,409,236,829.9x | 6.54e-06 | 4.26e-12 |
| 1215 | 4.06e-14 | 2.45e+00 | 60,458,610,620,267.5x | 2.82e-07 | 1.16e-14 |
| 1599 | 9.91e-16 | 2.15e+00 | 2,171,437,696,096,573.8x | 4.53e-08 | 4.18e-16 |

### Experiment 3: walk-forward alpha tracking under a compute budget

8192 ticks, window 512, re-fit every 64 ticks, 8 seeds, iterative methods warm-started at lr 0.02 (untuned). Each row gives the sample-gradient evaluations allowed per re-fit. IC = out-of-sample Pearson correlation of the prediction with the *clean* hidden signal (observable only in simulation); median [IQR] over seeds. OLS is the closed-form reference and ignores the budget.

| Budget / window | OLS | Adam | Coordinate SVRG |
|---|---|---|---|
| 1,024 | 0.788 [0.769, 0.824] | 0.793 [0.776, 0.830] | 0.784 [0.762, 0.821] |
| 2,048 | 0.788 [0.769, 0.824] | 0.789 [0.772, 0.827] | 0.786 [0.766, 0.824] |
| 4,096 | 0.788 [0.769, 0.824] | 0.787 [0.768, 0.824] | 0.787 [0.767, 0.822] |
| 8,192 | 0.788 [0.769, 0.824] | 0.788 [0.768, 0.823] | 0.787 [0.767, 0.824] |
| 16,384 | 0.788 [0.769, 0.824] | 0.787 [0.768, 0.824] | 0.788 [0.769, 0.824] |

Secondary metrics at the largest budget (16384):

| Method | IC vs realised target | sign hit-rate | per-tick Sharpe |
|---|---|---|---|
| OLS | 0.431 [0.388, 0.479] | 0.771 [0.745, 0.788] | 0.329 [0.292, 0.402] |
| Adam | 0.430 [0.387, 0.478] | 0.770 [0.744, 0.788] | 0.329 [0.290, 0.403] |
| Coordinate SVRG | 0.431 [0.388, 0.478] | 0.771 [0.745, 0.788] | 0.334 [0.292, 0.401] |

## Reading the results

**1. Variance reduction is what matters; the adaptive scaling adds nothing measurable here.**
Both SVRG variants reach float64 round-off (gap around 1e-31) inside the budget in all three regimes, while the
best tuned baselines stop around 1e-7 (cosine decay) or 1e-5 to 1e-6 (constant step). SVRG reaches 1e-8 times the
initial gap in 15 to 23 epochs of gradient evaluations; no baseline reaches it in 100. But **SVRG + momentum
(no normalisation) does as well as Coordinate SVRG**: both hit round-off, and the evaluations needed to reach the 1e-8 target
are mixed (the adaptive variant is faster in calm, 85k vs 93k, and toxic, 61k vs 72k; the momentum variant is faster in volatile,
72k vs 85k), with differences of roughly 10 to 20% from 7 seeds, which I do not treat as significant. So this ablation attributes the gain to variance reduction and gives no evidence
for the adaptive coordinate scaling on this problem. That is an honest negative result for one of the project's
design choices; the scaling may matter on badly conditioned or heavy-tailed-gradient problems that this testbed does not contain.

**2. The Variance Reduction Law holds along a real trajectory, and the bounds in `THEORY.md` hold.**
Mini-batch SGD variance stays around 2 for the whole run (it has a noise floor), while the variance of the
variance-reduced gradient falls from 0.26 to 1e-15, a ratio of about 2e15 at the end. Measured variance was
below the Lemma 2 bound and the Proposition 3 bound in 25 of 25 probes, each taken at the worst-case staleness within
a snapshot cycle. The bounds are loose upper bounds, so this checks they are not violated; it does not show they are tight.

**3. The walk-forward experiment is a null result.**
At window 512 and 32 features, all three methods give the same out-of-sample IC as the closed-form solution to within
seed noise, at every budget from 1,024 to 16,384 gradient evaluations per re-fit. The small differences at the
smallest budget (Adam 0.793, OLS 0.788, SVRG 0.784) are far inside the interquartile ranges (about 0.77 to 0.82), and
no significance test was run. This test therefore supplies **no evidence that SVRG improves signal tracking**.
The problem is too small for the choice of optimizer to matter: a closed-form solve is exact, and
warm-started gradient methods converge to it almost immediately. A meaningful version needs a high-dimensional or
very-large-n setting where the direct solve is expensive, or a model with no closed form.

## Limitations (read before citing any number)

* **Synthetic data only.** The simulator generates what the model assumes (linear hidden signal, additive noise,
  jumps). Nothing here is evidence of real-market performance.
* **One objective class.** Least squares is strongly convex, which is the best case for SVRG. No logistic, non-convex,
  or deep-learning results exist yet; variance reduction is known to help much less there (Defazio & Bottou, 2019).
* **Scale.** `n = 4096`, `d = 32`: a direct solve is instant. SVRG's advantage is asymptotic in `n` and `d`.
* **Sample-gradient evaluations are not wall-clock time.** SVRG adds memory traffic and a synchronised snapshot pass.
* **The adaptive variant has no convergence proof** (see `THEORY.md`, Section 4).
* **Small seed counts** (2 tuning, 7 evaluation seeds for Experiment 1; 8 for Experiment 3). Intervals shown are
  interquartile ranges, not confidence intervals.
* **One learning-rate grid cell sits on the boundary** (SVRG + momentum in the calm regime, marked †). The true optimum
  could be slightly better; this cannot change the conclusion because the method is already at round-off.
* **The PyTorch and JAX engines have not been run by the author's sandbox**; run `pytest` to validate them against the oracle.

## Next steps that would make the claims stronger

1. Logistic regression on the sign of the return (a proper generalised-linear signal model).
2. A high-dimensional regime (`d` in the thousands, `n` in the millions) where a direct solve is not free, with wall-clock timing on GPU from the PyTorch engine.
3. Ablate the snapshot interval against the Proposition 3 design rule.
4. Attempt the conjecture in `THEORY.md` Section 4 (convergence of the adaptive variant), or construct a counterexample.
