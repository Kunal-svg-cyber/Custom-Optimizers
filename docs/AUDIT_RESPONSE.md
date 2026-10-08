# Response to an external audit

An independent critique listed six "loopholes". Each is assessed below against the code and the stored results: **fair**,
**partly fair**, or **overstated/incorrect**, with what was done. Where a point could be tested, it was.

## 1. Wall-clock versus sample efficiency: *partly fair*

*Fair:* there is no wall-clock win over a direct solve at the sizes tested (normal equations 0.06 s, SVRG 0.08 s at n = 40000, d = 200; NumPy, one CPU core).

*Not accurate:* the sample-efficiency results do not hide the snapshot overhead. Every SVRG number in the experiments charges the double
gradient per step and every full-gradient snapshot pass, so the reported advantage is net of that cost. What is *untested* is wall-clock behaviour on a GPU at larger sizes.

*Done:* `benchmarks/timing_study.py` measures time-to-target for direct solves, Jacobi SVRG (the real PyTorch engine) and Adam across sizes, plus peak memory. It has **not been run**; the question stays open until it is.

## 2. Host synchronisation: *fair for telemetry, overstated otherwise*

*Fair:* the measured 2 ms step on a T4 (32 parameters) was taken with `telemetry=True`, which reads values back to the host every step.

*Overstated:* the claim that alternating closures "forces the CPU to wait" is not what the code does. The closures and the update contain no host reads; only telemetry does. Whether the engine is fast with telemetry off has not been measured.

*Done:* the README now says to use `telemetry=False` for speed, and `timing_study.py` reports microseconds per step with telemetry on versus off. The default stays `True` because diagnostics and tests read `last_stats`.

## 3. Rigid determinism requirement: *fair as a deployment concern, "diverge wildly" unsupported*

*Fair:* the PyTorch engine needs the user's closure to be a pure function of the batch id; a shuffling loader, dropout or a stateful counter silently breaks the control variate.

*Unsupported:* Experiment 14 measured the gradient estimator's error (2.1 times worse than plain SGD with misaligned batches), not divergence of an optimizer run. And the JAX engine cannot misalign, because it receives the batch object itself.

*Done:* `CoordinateSVRG.verify_closure_determinism(closure)` evaluates the closure repeatedly at the current weights and compares losses and gradients (tolerant of floating-point summation noise). A unit test covers a pure closure, a randomised closure and a stateful closure. **The test has not yet been run on Colab or CI.**

## 4. Worse generalisation on the non-convex problem: *fair, but the explanation is not supported*

*Fair:* with learning rates tuned on training loss, the adaptive variants had the lowest training loss and the highest test loss (Experiment 8).

*Not supported:* the "sharp versus flat minima" story was never measured. **Experiment 15** re-ran the comparison with learning rates chosen on validation loss:
the deficit shrinks (+0.0135 to +0.0038), SVRG with plain momentum ties the best baseline (+0.0000, CI [-0.0001, +0.0007]) and beats its own non-variance-reduced counterpart (-0.0011), and Adam is itself worse than SGD (+0.0012, 8 of 8 seeds). The deficit tracks **adaptive scaling**,
not the removal of gradient variance. One small network on one synthetic task.

## 5. Synthetic data disconnect: *premise partly wrong; conclusion refuted by an experiment*

*Not accurate:* the simulator is not a "Gaussian vacuum". It includes Student-t (df 3) toxic-flow shocks, arbitrage jumps, regime switches, drifting alpha and autocorrelated features.

*Fair:* the round-off result did not reproduce on three real datasets with raw features (Experiment 13). But **Experiment 16** shows the cause: standardising the same features cuts the Hessian condition numbers from about 1e7 to 34 to 470, and then SVRG variants reach round-off on all three datasets (best of them at worst 2e-20) while the best tuned baselines stop between 6e-8 and 2e-2. The shortfall was raw-scale ill-conditioning, not "dirty real data". Three small non-financial datasets; no market data was used.

## 6. Memory footprint: *fair and quantified*

The PyTorch engine keeps the flat parameters (N), a (4, N) state block and a (3, N) scratch block: about 8N floats plus a transient gradient, versus roughly 4N for Adam with gradients, so about twice Adam's footprint. It is not intended for billion-parameter models, and the project's stated regime is linear and generalised-linear models (variance reduction is known to help little in deep learning).
`timing_study.py` reports measured peak extra memory.

## Net effect

Two of the six points led to experiments that changed conclusions (4 and 5), one led to a new safeguard (3), one to documentation and a measurement script (1, 2, 6). None changes the headline
findings, but the real-data and generalisation statements are now more precise than before. The critique's strongest remaining point is that **wall-clock performance on a GPU is unmeasured**.
