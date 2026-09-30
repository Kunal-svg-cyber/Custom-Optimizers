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
