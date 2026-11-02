# Compiled-kernel latency on a Google Colab CPU (Intel Xeon @ 2.00 GHz, one core)

Differential test against the NumPy oracle: max |C - oracle| = 3.33e-16 after 200 steps. Best of 7 timings of 20000 operations each; nanoseconds per operation unless stated. Machine-specific and noisy.

| d | dot product (inference) | RLS update | SGD step b=1 | SVRG step b=1 | SVRG step b=8 | SVRG step b=64 | snapshot over window (ms) |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 84 | 62 | 95 | 370 | 1464 | 0.58 (window 100,000) |
| 32 | 20 | 1467 | 119 | 217 | 836 | 6854 | 3.09 (window 100,000) |
| 128 | 126 | 29620 | 318 | 641 | 4614 | 36265 | 20.97 (window 100,000) |
