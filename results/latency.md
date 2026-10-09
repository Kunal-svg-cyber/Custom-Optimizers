# Compiled-kernel latency (Intel(R) Xeon(R) Processor @ 2.10GHz, one core)

Differential test against the NumPy oracle: max |C - oracle| = 3.33e-16 after 200 steps.

Best of 7 timings of 20000 operations each; nanoseconds per operation unless stated. Machine-specific and noisy.

| d | dot product (inference) | RLS update | SGD step b=1 | SVRG step b=1 | SVRG step b=8 | SVRG step b=64 | snapshot over window (ms) |
|---|---|---|---|---|---|---|---|
| 8 | 2 | 57 | 36 | 59 | 198 | 1256 | 0.43 (window 100,000) |
| 32 | 15 | 959 | 70 | 129 | 533 | 3940 | 1.84 (window 100,000) |
| 128 | 62 | 16653 | 239 | 388 | 1643 | 12288 | 8.34 (window 100,000) |
