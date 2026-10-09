# Questions a sceptical reviewer would ask, with honest answers

Every answer points to evidence in this repository. Where the honest answer is "not shown", it says so.

**1. Is this a new algorithm?**
No. SVRG, Adam and diagonal preconditioning are all established. What the project contributes is (a) an engineered and thoroughly
verified implementation, (b) a short corollary (Proposition 5) that expresses SVRG's rate through the preconditioned condition number, and
(c) a tested prediction from it. I did not do a systematic literature search for prior work on preconditioned SVRG, and related ideas almost
certainly exist, so treat Jacobi SVRG as an application and verification of known ideas, not a claim of novelty.

**2. Why should variance reduction help at all on noisy data?**
Plain mini-batch SGD has a variance floor because per-sample gradients do not vanish at the optimum. The control-variate gradient's variance is
bounded by the squared distance to the snapshot (Lemma 2), so it vanishes as the iterate approaches it. Experiment 2 shows SGD variance flat near 2 while
SVRG's falls by about fifteen orders of magnitude along one run.

**3. Are your baselines tuned fairly?**
Each method gets its own learning-rate sweep on tuning seeds and is scored on disjoint seeds; baselines include cosine decay; budgets are counted in
sample-gradient evaluations, with SVRG charged for double gradients and snapshot passes. Limits: batch size, momentum and weight decay were not tuned for
the baselines, and several best learning rates sit on grid edges (marked in the tables).

**4. Float64 round-off sounds too good. What does it mean?**
It means the method converged until floating-point error dominated on these strongly convex synthetic problems. A "23-decade" difference is not a speed factor;
read those rows as win counts. On three real datasets with raw features the round-off result did **not** reproduce (Experiment 13), but it did once the features were standardised (Experiment 16).

**5. Why did the adaptive method plateau, and is that general?**
On badly scaled least squares, constant-rate Coordinate SVRG plateaued while a decaying rate did not. That is consistent with normalisation keeping the step near the
learning rate. It is not general: on badly scaled logistic regression (Experiment 12) it did not plateau. I have no proof about the adaptive engine (THEORY.md Section 4).

**6. If Jacobi SVRG is better, why keep the adaptive engine?**
Jacobi SVRG needs the Hessian diagonal, available for linear and generalised-linear models; the adaptive engine uses gradients only. Where the Hessian diagonal is available,
the simpler method won every comparison I ran; that is stated in the report.

**7. Does any of this beat just solving the problem directly?**
Not at the sizes tested. At 40,000 by 200 the normal equations were slightly faster than SVRG (0.06 s versus 0.08 s, Experiment 5). The regime where SVRG could win (very large dimension, no closed form) was not tested.

**8. What do the invariant ablations show?**
Misaligned batches make the estimator about twice as bad as plain SGD, so alignment is what delivers the benefit. Textbook Adam failed in most adversarial float32 trials where the full
guard set never did, and the gradient bound, not the denominator floor, was the operative guard (Experiment 14). The inputs are deliberately hostile and describe a robustness guarantee.

**9. How do I know the engines are correct?**
Three independent implementations (PyTorch, JAX, NumPy oracle) agree step for step in automated tests, including fuzzing; the PyTorch engine matched the oracle to 1.8e-7 on a Colab T4 GPU; and the report's
numbers are machine-checked against the stored results. Limits: the experiment numbers come from the oracle, and no timing study was done.

**10. Does this say anything about real markets?**
No. The data are synthetic apart from three small non-financial datasets. `experiments/real_market_study.py` is provided, with a placebo and block-bootstrap intervals, but no results from it are included.
Daily returns are barely predictable, so I expect small or null differences between optimizers.

**11. What did you get wrong along the way?**
See `CHANGELOG.md`: a claim about denominator clamping was retracted after an ablation, a "best on all datasets" statement became a tie, a missing convexity hypothesis was added to a theorem, and a theory
citation's scope was narrowed. Finding and logging these is part of the method.

**12. What would you do next with more time?**
Run the real-market study and report it, whatever it shows; time the engine on a problem large enough that wall-clock could favour iterative methods; and try to prove or refute the convergence conjecture for the adaptive engine.

**13. Isn't the worse held-out loss just what you get from removing SGD's noise?**
It was not measured as such. With learning rates chosen on validation loss (Experiment 15) the deficit shrinks, SVRG with plain momentum ties the best baseline, and Adam, which keeps its gradient noise, is itself worse than SGD. That points at adaptive scaling, not at variance reduction.
A sharp-versus-flat-minima explanation would need a sharpness measurement, which I did not make.

**14. Isn't real data simply harder than your simulator?**
The simulator already has heavy tails, jumps, regime switches and autocorrelation. The real-data shortfall with raw features disappeared when the same features were standardised (Experiment 16): the cause was raw-scale conditioning (Hessian condition numbers near 1e7), not the data being real.
See `docs/AUDIT_RESPONSE.md` for the full response to an external critique.

**15. Could this run in a high-frequency trading loop?**
Not as it stands, and the repository does not claim it. It is an offline research and calibration project. Two things are worth separating, though. First, the 2 ms GPU step quoted in the docs measures PyTorch with host synchronisation on a tiny problem;
a compiled C implementation of the same update costs about 129 ns per single-sample step at d = 32 (`results/latency.md`). Second, latency budgets of tens of microseconds apply to signal *inference* (a dot product, about 15 ns here), not to *fitting*; a refit on a trailing window can run off the critical path.
What the project does not show is that variance-reduced fitting beats recursive least squares for online estimation: RLS is exact at every tick and costs about 1 µs at d = 32.
