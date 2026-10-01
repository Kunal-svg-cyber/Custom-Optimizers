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
