# Claims and evidence

Every substantive claim in this repository, the kind of support it has, where to check it, and its limits.
**Kind** is one of: *proved* (short proof in `THEORY.md`), *cited* (known theorem, not re-proved), *tested*
(a unit or property test), *measured* (an experiment in `results/experiments.json`, with its protocol), or
*not claimed*. "Oracle" means the NumPy reference implementation that produced all experiment numbers.

## Algorithm and engineering

| # | Claim | Kind | Evidence | Limits |
|---|---|---|---|---|
| 1 | The variance-reduced gradient is unbiased for any snapshot | proved | Lemma 1; `test_lemma1_*` | none |
| 2 | Its variance is at most `Lbar2/b * ||w - w~||^2`; for least squares it scales exactly with the squared displacement | proved, tested | Lemma 2; `test_lemma2_*`; Experiment 2 (25/25 probes within the bound) | the bound is loose in practice |
| 3 | Variance of the engine's gradient is bounded in terms of the snapshot interval `K` | proved, measured | Proposition 3; Experiment 9 (held at every `K`, loose by 3 to 7 orders of magnitude) | cannot be used to pick `K` tightly |
| 4 | The update cannot produce NaN or Inf and each coordinate moves at most `lr * update_clip` | proved, tested | Proposition 4; fuzz tests (oracle, torch, JAX); Experiment 14 | covers the update path; stored snapshot rows keep their inputs verbatim |
| 5 | The gradient bound, not the denominator floor, is the operative numerical guard | measured | Experiment 14 (textbook Adam and clamp-only fail identically, full guards never) | adversarial float32 inputs, not typical training |
| 6 | Batch alignment between the live and snapshot gradients is what delivers variance reduction | measured, tested | Experiment 14a (misaligned is 2.1x worse than plain SGD); `test_misaligned_*` | least squares |
| 7 | PyTorch and JAX engines follow the same update rule as the oracle | tested, measured | step-for-step differential tests; torch-vs-JAX parity; the full suite (58 tests at the time) passed on a Colab T4 runtime; GPU engine within 1.8e-7 of the oracle over 100 steps; a 640-step GPU benchmark reproduced the oracle's final numbers to four digits | correctness, not speed; a fresh GitHub Actions run will confirm the newest tests on Python 3.10 and 3.12 |
| 8 | Parameters, snapshot and moments are contiguous in PyTorch | tested | `test_torch_flat_contiguity_*` | PyTorch engine only |

## Theory

| # | Claim | Kind | Evidence | Limits |
|---|---|---|---|---|
| 9 | SVRG with a frozen diagonal preconditioner converges linearly at a rate set by the preconditioned condition number | proved (corollary of a cited theorem), tested | Proposition 5; change-of-variables identity test; Experiment 11 (measured contraction 0.08 to 0.10 per epoch against a guaranteed 0.5) | frozen `D` only; not the adaptive engine; theorem is for random-iterate snapshots |
| 10 | Jacobi scaling makes the condition number invariant to feature scales | proved, tested | Proposition 5 discussion; `test_jacobi_conditioning_*`; Experiment 10 | |
| 11 | Convergence of the adaptive engine | **not claimed** | `THEORY.md` Section 4 | open; Adam-type methods can fail in general |
| 12 | Convergence of non-convex or deep models | **not claimed** | | variance reduction is known to help less there |

## Empirical findings (all on the oracle; synthetic unless stated)

| # | Claim | Kind | Evidence | Limits |
|---|---|---|---|---|
| 13 | On well-scaled convex problems SVRG variants reach float64 round-off and beat the best tuned Adam / SGD-momentum (constant and cosine) | measured | Experiments 1, 4 (20/20 and 12/12 seeds); paired bootstrap | synthetic; magnitude of the difference is not meaningful at round-off |
| 14 | Adaptive scaling helps when features are badly scaled and hurts on a larger well-scaled problem | measured | Experiments 5, 6, 12 | one problem family each; Experiment 5 has 3 seeds |
| 15 | A method with theory-prescribed hyper-parameters (Jacobi SVRG, no tuning) beats the best tuned adaptive variant on badly scaled least squares | measured | Experiment 11 (20/20 seeds; 4.7x fewer evaluations) | needs the Hessian diagonal; one extra data pass not charged |
| 16 | The Jacobi result carries to logistic regression with bad scaling | measured | Experiment 12 (12/12 seeds; tuned, not theory-prescribed) | synthetic; constant-rate Coordinate SVRG does not plateau there, so that effect is problem-dependent |
| 17 | On three real datasets with raw features, preconditioned methods dominate unpreconditioned ones; Jacobi SVRG has the lowest median gap on wine and diabetes and ties with Coordinate SVRG (cosine) on breast cancer, by only 0.06 to 0.52 decades over the best tuned baseline | measured | Experiment 13 | three small non-financial datasets; seeds vary batch order only; no method reached the target; one baseline learning rate on the grid edge |
| 18 | The round-off result does not carry to the real datasets with raw features, but it does once the features are standardised (SVRG variants at 1e-20 to 1e-31, best baselines 6e-8 to 2e-2): the shortfall was raw-scale conditioning, not real-data difficulty | measured | Experiments 13 and 16 | three small datasets; collinearity remains but is moderate; on standardised features Jacobi scaling is near the identity and has no advantage over SVRG + momentum |
| 19 | Non-convex network: SVRG reaches a sharper stationary point; the held-out deficit seen with training-loss tuning shrinks under validation-loss tuning and tracks adaptive scaling (Adam also worse than SGD), not variance reduction (SVRG + momentum ties the best baseline) | measured | Experiments 7, 8, 15 | one small network; sharp-versus-flat minima not measured |
| 20 | No wall-clock win over a direct solve at `n = 40000`, `d = 200` | measured | Experiment 5 | NumPy on one CPU core; tuning cost excluded |
| 21 | No benefit for walk-forward signal tracking | measured | Experiment 3 (differences at most 0.005 in IC; SVRG never beats Adam) | simulated alpha; small problem |

## Engineering constraints

| # | Claim | Kind | Evidence | Limits |
|---|---|---|---|---|
| 26 | A non-deterministic closure can be detected before training | tested (test written, first run pending) | `test_torch_verify_closure_determinism_*` | PyTorch only; the JAX engine cannot misalign because it receives the batch itself |
| 27 | The PyTorch engine uses about 8N floats (parameters, a (4, N) state block, a (3, N) scratch block) versus roughly 4N for Adam including gradients | stated from the code | `src/torch_optimizer.py` (`_FlatBlock`) | peak memory not yet measured; `timing_study.py` reports it |
| 28 | Telemetry-on versus telemetry-off step cost; GPU wall-clock versus direct solves | **not claimed** | `benchmarks/timing_study.py` provided, not run | to be filled in after a GPU run |

## Latency and scope

| # | Claim | Kind | Evidence | Limits |
|---|---|---|---|---|
| 29 | A compiled C kernel reproduces the NumPy oracle's Jacobi-SVRG weights | tested | `test_c_kernel_matches_the_numpy_oracle` (max difference 3e-16); `python -m benchmarks.latency_kernel` | double precision, one problem family |
| 30 | In compiled code on one core, at d = 32, a single-sample SVRG step costs about 129 ns, an RLS update about 0.96 µs and an inference dot product about 15 ns; a 100,000-row snapshot costs about 1.8 ms | measured | `results/latency.json`, `results/latency.md` | machine-specific (Xeon @ 2.1 GHz, shared cloud VM), best-of-7 timings, no thread pinning or fixed-point; says nothing about statistical usefulness |
| 31 | The 2 ms GPU step reflects framework and host-synchronisation overhead rather than the algorithm | measured (indirectly) | the compiled step is four orders of magnitude cheaper | the split between PyTorch overhead and telemetry synchronisation is not measured; `timing_study.py` would do it |
| 33 | `telemetry="lazy"` gives the same statistics as eager telemetry without a host read each step, and does not change the trajectory | tested (test written; first run pending) | `test_torch_lazy_telemetry_matches_eager_*` | whether it removes the 2 ms step on a GPU is unmeasured; `timing_study.py` measures it |
| 34 | `make_svrg_scan` equals K sequential JAX steps | tested (test written; first run pending) | `test_jax_scan_matches_sequential_steps` | CPU-tested only; no timing claim |
| 32 | Suitability for production high-frequency trading | **not claimed** | `README.md` Scope | the repository is offline research; RLS is exact per tick and the project does not show SVRG beating it online |

## Real-market study (protocol only)

| # | Claim | Kind | Evidence | Limits |
|---|---|---|---|---|
| 22 | The walk-forward pipeline uses no future information | tested | `test_market_features_are_causal`, `test_walk_forward_predictions_have_no_lookahead` | tests use synthetic data; the live data path (`yfinance`/CSV) is not unit-tested beyond the CSV loader |
| 23 | The pipeline detects a genuine signal and not a shifted placebo | tested | `test_signal_is_detected_and_placebo_is_not` (three seeds, strong synthetic signal) | a weak signal (strength 1.0) is detected on some seeds and not others, as expected at that signal-to-noise |
| 24 | The numbers in the report and the one-page summary are exactly those in `results/experiments.json`, and the report references no undefined number | tested | `test_report_numbers_are_in_sync_*`; `python paper/make_numbers.py --check` | covers generated numbers, not hand-written prose |
| 25 | Any statement about real-market predictability or optimizer effects on real returns | **not claimed** | script provided, no results run by the author | to be filled in only after the owner runs it |

## Explicitly not claimed

* Performance on real financial data. No market data was used.
* Any GPU or large-scale speed-up. The PyTorch engine has been validated for agreement with the oracle on a T4, but no timing study was done (a telemetry-enabled step took about 2 ms on a 32-parameter problem, dominated by host synchronisation).
* Superiority of Coordinate SVRG in general. On the problems tested, the simpler Jacobi SVRG beat it wherever the
  Hessian diagonal was available; Coordinate SVRG's scaling is a gradient-only proxy.
* Statistical significance beyond the paired bootstrap intervals reported; no multiple-comparison correction was applied.
