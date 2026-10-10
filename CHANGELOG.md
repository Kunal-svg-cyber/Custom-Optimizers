# Changelog

A running record of what was added and, importantly, **what was corrected**, so a reader can see how the conclusions
changed as the evidence came in.

## Corrections (things earlier versions got wrong or overstated)

| What | Earlier statement | Correction | Where it was caught |
|---|---|---|---|
| Role of the denominator clamp | A clamp rather than an additive `eps` was said to matter in float32 | An ablation shows textbook Adam and a clamp-only variant fail identically on adversarial inputs; the operative guard is bounding and sanitising the gradient | Experiment 14 |
| Real-data ranking | Jacobi SVRG was "best on all three" real datasets | On breast cancer it ties with Coordinate SVRG (cosine): medians 6.015e-3 versus 6.005e-3 | proofreading against the stored results |
| Johnson-Zhang statement | The theorem was stated without requiring each component function to be convex | Convexity added to the theorem and to Proposition 5 | proofreading the theory note |
| van der Sluis scope | Jacobi scaling "within a factor of d of the best diagonal scaling" was attached to `kappa_D` | The result concerns the preconditioned Hessian's condition number, not the per-sample constant in `kappa_D` | proofreading the theory note |
| Plateau explanation | The constant-rate plateau was attributed to `beta2 = 0.999` memory | Relabelled as untested: the Experiment 1 runs are longer than that memory, and Experiment 12 shows no plateau | Experiments 1 and 12 |
| Engine validation | The engines were described as never executed | Updated after the first CI run (44 of 45) and the Colab run (all tests, GPU within 1.8e-7 of the oracle) | CI and Colab output |
| Torch fuzz test | Asserted every stored state row was finite | The engine guarantees finite weights and moments; stored snapshot rows keep their inputs verbatim; assertion corrected | first CI run |
| Report abstract | "All data are synthetic" | Three small real datasets were added; wording corrected | proofreading the report |
| Real-data conclusion | "The round-off result does not carry to real data" (Experiment 13) | It carries once the features are standardised; the shortfall was raw-scale conditioning, not real-data difficulty | Experiment 16, prompted by an external audit |
| Source of the held-out deficit | Attributed to the optimizer's precision (Experiment 8) | With validation-tuned rates it shrinks and tracks adaptive scaling (Adam is also worse than SGD); SVRG + momentum ties the best baseline | Experiment 15, prompted by an external audit |
| Reading of the 2 ms GPU step | Quoted as a property of the method ("dominated by host synchronisation") | It is PyTorch plus telemetry on a tiny problem; a compiled C kernel runs the same update in about 129 ns per step at d = 32 | compiled-kernel latency study, prompted by a second external review |

## Additions, in order

1. Environment simulator, PyTorch and JAX engines, NumPy oracle, invariant tests (the original blueprint).
2. Tuned ablation, variance-law check, walk-forward tracking (Experiments 1 to 3).
3. Logistic, larger-problem, badly scaled, non-convex, held-out, snapshot-interval and conditioning experiments (4 to 10).
4. Proposition 5 (frozen diagonal preconditioner), Jacobi SVRG with theory-prescribed hyper-parameters (Experiment 11) and its logistic generalisation (12).
5. Frozen-preconditioner mode in both engines, with differential tests.
6. Real datasets with raw features (13) and invariant-violation ablations (14).
7. Technical report, claims-to-evidence map, one-page summary, CI workflow, Colab notebook.
8. Real-market walk-forward study script with placebo and block bootstrap (provided; no results yet).
9. Report-number consistency check (`python paper/make_numbers.py --check`).
10. Compiled C kernel (`benchmarks/svrg_kernel.c`) and latency study (`benchmarks/latency_kernel.py`), checked against the oracle to 3e-16; Scope section; second external-review response.
11. `telemetry="lazy"` mode (no per-step host synchronisation) in the PyTorch engine and `make_svrg_scan` (K steps in one compiled call) in the JAX engine, with tests; third external-review response.
12. External-audit response (`docs/AUDIT_RESPONSE.md`); validation-tuned held-out experiment (15); standardised real-data experiment (16); closure-determinism checker; GPU timing-study script (provided, not run); deployment notes.

## Known open items

* No market data has been analysed; run `experiments/real_market_study.py` and report the result.
* No timing study of the PyTorch engine; the Colab run established agreement with the oracle, not speed. `benchmarks/timing_study.py` is provided but has not been run.
* The closure-determinism test, the frozen-preconditioner tests, the lazy-telemetry test and the JAX scan test have not yet been confirmed by CI or Colab.
* No convergence proof for the adaptive engine.
