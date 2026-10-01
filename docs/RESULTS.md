# Results

All numbers are produced by `python -m experiments.run_experiments` (about 6 minutes on one CPU core, NumPy only)
and stored in `results/experiments.json`. They use the NumPy reference implementation (`src/reference_numpy.py`),
which the PyTorch and JAX engines are tested against step for step.
**The PyTorch/JAX engines did not produce these numbers.** Figures are in `docs/figures/`.

Protocol: every method gets its own learning-rate sweep (log-spaced, 1e-5 to 10^-0.5) on *tuning* seeds, then is
scored on *disjoint* evaluation seeds. Budgets are counted in sample-gradient evaluations; SVRG is charged for its double
gradients and every full-gradient snapshot pass. Baselines include a cosine-decay learning-rate schedule, because a
constant-step baseline is a weak comparison for a method whose selling point is convergence at a constant step.
Cosine-decay runs are tied to the budget (the rate reaches zero exactly at the end), so their mid-run curves are not
comparable with constant-rate curves; compare final values.

![least squares](figures/ablation_curves.png)
*(Curves bottom out at 1e-18 only because the plot clamps there; the underlying gaps are about 1e-31, i.e. float64 round-off.)*

![logistic](figures/logistic_curves.png)

![larger problem](figures/highdim_curves.png)

![badly scaled features](figures/hetero_curves.png)

![variance](figures/variance_decay.png)

![walk-forward](figures/walk_forward_ic.png)

### Experiment 1: tuned ablation on least squares

Least squares, n=4096, d=32, batch 64, snapshot every 64 steps. Budget: 100 epochs of sample-gradient evaluations. Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; 'evals to target' = sample-gradient evaluations to reach 1e-08 x the initial gap.

**calm** (noise sigma = 0.25)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 5.66e-06 [5.15e-06, 8.24e-06] | not reached |
| SGD + momentum, cosine lr | 4.22e-03 | 8.35e-08 [6.90e-08, 1.17e-07] | not reached |
| Adam (clamped) | 3.16e-04 | 4.05e-06 [3.98e-06, 4.10e-06] | not reached |
| Adam (clamped), cosine lr | 7.50e-04 | 8.83e-08 [6.88e-08, 1.50e-07] | not reached |
| SVRG + momentum | 3.16e-01 † | 3.29e-31 [2.16e-31, 5.65e-31] | 92,800 |
| Coordinate SVRG (ours) | 1.78e-03 | 3.27e-31 [2.17e-31, 6.00e-31] | 85,376 |
| Coordinate SVRG, cosine lr (ours) | 1.00e-02 | 4.17e-31 [2.23e-31, 5.65e-31] | 85,376 |

**volatile** (noise sigma = 1.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 7.96e-06 [7.12e-06, 1.09e-05] | not reached |
| SGD + momentum, cosine lr | 4.22e-03 | 9.42e-08 [9.20e-08, 1.27e-07] | not reached |
| Adam (clamped) | 3.16e-04 | 4.35e-06 [4.01e-06, 4.63e-06] | not reached |
| Adam (clamped), cosine lr | 7.50e-04 | 9.72e-08 [8.94e-08, 1.75e-07] | not reached |
| SVRG + momentum | 1.33e-01 | 3.48e-31 [2.47e-31, 5.56e-31] | 72,448 |
| Coordinate SVRG (ours) | 1.78e-03 | 3.83e-31 [2.62e-31, 5.43e-31] | 85,376 |
| Coordinate SVRG, cosine lr (ours) | 1.00e-02 | 3.79e-31 [2.76e-31, 5.99e-31] | 85,376 |

**toxic** (noise sigma = 3.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 7.50e-04 | 2.04e-05 [1.89e-05, 2.64e-05] | not reached |
| SGD + momentum, cosine lr | 4.22e-03 | 2.76e-07 [2.57e-07, 3.30e-07] | not reached |
| Adam (clamped) | 7.50e-04 | 2.31e-05 [2.25e-05, 2.58e-05] | not reached |
| Adam (clamped), cosine lr | 1.78e-03 | 2.68e-07 [2.54e-07, 3.27e-07] | not reached |
| SVRG + momentum | 1.33e-01 | 3.58e-31 [3.37e-31, 6.25e-31] | 72,448 |
| Coordinate SVRG (ours) | 4.22e-03 | 5.15e-31 [3.85e-31, 6.59e-31] | 61,440 |
| Coordinate SVRG, cosine lr (ours) | 4.22e-03 | 5.24e-31 [3.21e-31, 6.94e-31] | 61,440 |

† best learning rate sat on the edge of the sweep grid (the true optimum may lie outside it).

### Experiment 4: tuned ablation on logistic regression (sign of the return)

Logistic regression on sign(return), L2=0.01, n=20000, d=64, batch 64, snapshot every 64 steps. Budget: 40 epochs of sample-gradient evaluations. Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; 'evals to target' = sample-gradient evaluations to reach 1e-08 x the initial gap.

**calm** (noise sigma = 0.25)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 1.11e-06 [1.06e-06, 1.42e-06] | not reached |
| SGD + momentum, cosine lr | 6.49e-03 | 2.39e-07 [2.19e-07, 2.46e-07] | not reached |
| Adam (clamped) | 1.33e-04 | 1.49e-06 [1.43e-06, 1.74e-06] | not reached |
| Adam (clamped), cosine lr | 4.87e-04 | 2.80e-07 [2.71e-07, 2.81e-07] | not reached |
| SVRG + momentum | 3.16e-01 † | 4.27e-32 [3.31e-32, 5.88e-32] | 189,280 |
| Coordinate SVRG (ours) | 4.87e-04 | 1.65e-30 [2.61e-31, 7.68e-30] | 302,048 |
| Coordinate SVRG, cosine lr (ours) | 1.78e-03 | 5.96e-32 [4.99e-32, 9.88e-32] | 194,016 |

**volatile** (noise sigma = 1.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 1.06e-06 [1.01e-06, 1.50e-06] | not reached |
| SGD + momentum, cosine lr | 6.49e-03 | 2.57e-07 [2.33e-07, 2.60e-07] | not reached |
| Adam (clamped) | 1.33e-04 | 1.40e-06 [1.39e-06, 1.65e-06] | not reached |
| Adam (clamped), cosine lr | 4.87e-04 | 3.00e-07 [2.67e-07, 3.06e-07] | not reached |
| SVRG + momentum | 3.16e-01 † | 3.25e-27 [3.96e-32, 1.33e-26] | 189,280 |
| Coordinate SVRG (ours) | 4.87e-04 | 3.25e-27 [2.08e-30, 1.33e-26] | 273,856 |
| Coordinate SVRG, cosine lr (ours) | 1.78e-03 | 3.25e-27 [6.44e-32, 1.33e-26] | 194,016 |

**toxic** (noise sigma = 3.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 1.78e-03 | 1.13e-06 [1.07e-06, 1.35e-06] | not reached |
| SGD + momentum, cosine lr | 6.49e-03 | 2.68e-07 [2.22e-07, 2.71e-07] | not reached |
| Adam (clamped) | 1.33e-04 | 1.49e-06 [1.46e-06, 1.83e-06] | not reached |
| Adam (clamped), cosine lr | 4.87e-04 | 3.09e-07 [2.66e-07, 3.20e-07] | not reached |
| SVRG + momentum | 8.66e-02 | 1.02e-30 [1.00e-30, 2.17e-29] | 189,280 |
| Coordinate SVRG (ours) | 4.87e-04 | 1.05e-30 [1.03e-30, 2.16e-29] | 194,016 |
| Coordinate SVRG, cosine lr (ours) | 4.87e-04 | 1.26e-30 [7.84e-31, 2.50e-29] | 194,016 |

† best learning rate sat on the edge of the sweep grid (the true optimum may lie outside it).

### Experiment 5: larger problem with wall-clock timing

Least squares, n=40000, d=200, batch 256, snapshot every 128 steps, wall-clock on one CPU core (NumPy). Budget: 30 epochs of sample-gradient evaluations. Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; 'evals to target' = sample-gradient evaluations to reach 1e-08 x the initial gap.

**volatile** (noise sigma = 1.0)

| Method | tuned lr | final gap, median [IQR] | evals to target | seconds to target |
|---|---|---|---|---|
| SGD + momentum | 1.78e-03 | 1.15e-05 [1.14e-05, 1.16e-05] | not reached | not reached |
| SGD + momentum, cosine lr | 6.49e-03 | 2.83e-06 [2.75e-06, 2.85e-06] | not reached | not reached |
| Adam (clamped) | 1.33e-04 | 7.29e-06 [7.02e-06, 7.44e-06] | not reached | not reached |
| Adam (clamped), cosine lr | 4.87e-04 | 1.99e-06 [1.99e-06, 2.03e-06] | not reached | not reached |
| SVRG + momentum | 2.37e-02 | 5.59e-26 [5.09e-26, 6.91e-26] | 405,760 | 0.07 |
| Coordinate SVRG (ours) | 4.87e-04 | 1.85e-16 [9.64e-17, 5.28e-16] | 716,224 | 0.14 |
| Coordinate SVRG, cosine lr (ours) | 1.78e-03 | 2.14e-18 [1.98e-18, 2.62e-18] | 582,528 | 0.12 |

Direct solves on the same data (one CPU core): `lstsq` 0.31 s, normal equations 0.04 s. The iterative times above exclude the cost of the learning-rate sweep.

† best learning rate sat on the edge of the sweep grid (the true optimum may lie outside it).

### Experiment 6: badly scaled features (adaptive scaling's home turf)

Least squares with column scales spread over 3 decades, n=4096, d=32, batch 64, snapshot every 64 steps. Budget: 100 epochs of sample-gradient evaluations. Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; 'evals to target' = sample-gradient evaluations to reach 1e-08 x the initial gap.

**volatile** (noise sigma = 1.0)

| Method | tuned lr | final gap, median [IQR] | evals to target |
|---|---|---|---|
| SGD + momentum | 3.16e-04 | 4.33e-02 [3.25e-02, 5.92e-02] | not reached |
| SGD + momentum, cosine lr | 1.00e-02 | 1.74e-02 [1.48e-02, 3.20e-02] | not reached |
| Adam (clamped) | 7.50e-04 | 5.79e-03 [2.95e-03, 9.52e-03] | not reached |
| Adam (clamped), cosine lr | 1.00e-02 | 7.77e-06 [6.15e-06, 1.66e-05] | not reached |
| SVRG + momentum | 1.00e-02 | 1.95e-02 [1.75e-02, 3.51e-02] | not reached |
| Coordinate SVRG (ours) | 1.00e-02 | 2.29e-04 [4.75e-05, 2.99e-04] | not reached |
| Coordinate SVRG, cosine lr (ours) | 5.62e-02 | 8.32e-11 [4.56e-12, 8.39e-10] | 377,216 |

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

**1. Variance reduction is the dominant effect on well-scaled problems.**
On least squares (Experiment 1), logistic regression (Experiment 4) and the larger problem (Experiment 5), every SVRG
variant reaches float64 round-off or close to it, while the best tuned baselines (with or without cosine decay) stop around
1e-5 to 1e-8 of loss gap. SVRG reaches 1e-8 times the initial gap within about 10 to 20 epochs of gradient evaluations; no
baseline reaches it in 30 to 100. On these problems SVRG + momentum (no adaptive scaling) is as good as or better than
Coordinate SVRG: the features are close to equal-variance, so a diagonal preconditioner has little to fix.

**2. Adaptive scaling earns its place only when the problem needs it (Experiment 6).**
With feature scales spread over three decades (typical of unnormalised price, volume and spread features), SVRG +
momentum stalls at a gap of about 2e-2 and Coordinate SVRG at a constant rate reaches 2e-4, a 100x improvement from the
scaling alone. The constant-rate adaptive method then plateaus, as `THEORY.md` Section 4 anticipates: normalisation keeps the
step near `lr` per coordinate. Adding a cosine decay (the **Coordinate SVRG, cosine lr** row) removes the plateau and
reaches about 8e-11, which beats the best tuned baseline, Adam with cosine decay (about 8e-6). It is the only method to hit the
1e-8 target on this problem. So the full design, variance reduction plus adaptive scaling plus a decaying rate, is what wins here;
none of the three components alone does.

**3. Wall-clock: no win over a direct solve at this size.**
On the larger problem (`n = 40000`, `d = 200`) SVRG + momentum reaches the 1e-8 target in 0.07 s of one NumPy core, versus 0.31 s for
`lstsq` but **0.04 s for the normal equations**, which is faster. So a direct solve still wins at this scale; SVRG's advantage here is over
the other *iterative* methods, none of which reach the target. The iterative times also exclude the cost of the learning-rate sweep (13 or 9
runs per method), which a direct solve does not need.

**4. The Variance Reduction Law holds along a real trajectory, and the proved bounds hold.**
Mini-batch SGD variance stays around 2 for the whole run (a noise floor), while the variance of the variance-reduced gradient falls from
0.26 to 1e-15. Measured variance was below the Lemma 2 bound and the Proposition 3 bound in every probe (25 of 25), each taken at worst-case snapshot
staleness. The bounds are loose upper bounds, so this checks they are not violated; it does not show they are tight.

**5. The walk-forward experiment is a null result.**
At window 512 and 32 features, all methods give the same out-of-sample IC as the closed-form solution within seed noise, at every budget
from 1,024 to 16,384 gradient evaluations per re-fit. The small differences at the smallest budget (Adam 0.793, OLS 0.788, SVRG 0.784) lie
far inside the interquartile ranges (about 0.77 to 0.82), and no significance test was run. This test supplies **no evidence that SVRG
improves signal tracking**; the problem is too small for the choice of optimizer to matter.

## Limitations (read before citing any number)

* **Synthetic data only.** The simulator generates what the model assumes (linear hidden signal, additive noise, jumps). Nothing here is evidence of real-market performance.
* **Convex objectives only** (least squares, regularised logistic regression), the best case for SVRG. No non-convex or deep-learning results exist; variance reduction is known to help much less there (Defazio & Bottou, 2019).
* **Modest scale.** The largest problem is `n = 40000`, `d = 200`, and a direct solve is faster than any iterative method on it. SVRG's wall-clock case needs problems where `X^T X` does not fit or form cheaply.
* **Sample-gradient evaluations are not wall-clock time**, and the wall-clock figures are NumPy on one CPU core, not the PyTorch/JAX engines on a GPU.
* **Tuning cost is excluded** from the iterative timings. Each iterative method needed a learning-rate sweep; the direct solve needed none.
* **The adaptive variant has no convergence proof** (see `THEORY.md`, Section 4). The cosine-decay result is empirical.
* **Small seed counts** (1 to 2 tuning and 3 to 7 evaluation seeds depending on the experiment). Intervals are interquartile ranges, not confidence intervals, and no significance tests were run.
* **Some learning rates sit on the grid edge** (marked †; SVRG + momentum in the calm regime of Experiments 1 and 4 and in the volatile regime of Experiment 4). The true optimum could be slightly better; this cannot change the conclusion where the method is already at round-off, but it means the SVRG + momentum versus Coordinate SVRG ordering in those cells should not be over-read.
* **The PyTorch and JAX engines have not been run by the author's sandbox**; run `pytest` to validate them against the oracle.

## Next steps that would make the claims stronger

1. Run the PyTorch engine on a GPU on a problem large enough that wall-clock favours iterative methods, and report time to target including tuning.
2. Significance testing (paired bootstrap over seeds) and more seeds.
3. A non-convex objective, to test whether the conclusions survive outside the convex case.
4. Ablate the snapshot interval against the Proposition 3 design rule.
5. Attempt the conjecture in `THEORY.md` Section 4, or construct a counterexample.
