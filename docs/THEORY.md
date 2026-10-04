# Theory: what is proven, what is cited, what is open

This note separates three kinds of statements so a reader can audit them:

* **Proved here** (short proofs included, checked numerically in `tests/` and `experiments/`).
* **Cited** (known results; stated precisely, not re-proved). Proposition 5 is a short corollary of a cited theorem and says so.
* **Open / empirical** (observed, no theorem claimed).

## 1. Setting and notation

Finite-sum objective `F(w) = (1/n) * sum_i f_i(w)` on `R^d`. Each `f_i` is differentiable with `L_i`-Lipschitz gradient.
Write `Lbar2 = (1/n) * sum_i L_i^2`.

For a mini-batch `B` of size `b` drawn uniformly, `g_B(w) = (1/b) * sum_{i in B} grad f_i(w)`.
Given a snapshot `w~` and its full gradient `mu = grad F(w~)`, the variance-reduced gradient is

```
g_hat(w) = g_B(w) - g_B(w~) + mu.
```

Both `g_B` evaluations use the **same** `B` (Deterministic Sample Alignment, invariant 2). Everything below fails if they do not.

For least squares, `f_i(w) = 0.5 * (x_i . w - y_i)^2`, so `grad f_i(w) = x_i (x_i . w - y_i)` and `L_i = ||x_i||^2`.

## 2. Results proved here

### Lemma 1 (unbiasedness)
`E_B[g_hat(w)] = grad F(w)`.

*Proof.* `E_B[g_B(w)] = grad F(w)` and `E_B[g_B(w~)] = grad F(w~) = mu`, so the expectation is `grad F(w) - mu + mu`. ∎

Unbiasedness holds for any snapshot, however stale. Staleness costs variance, never bias.

### Lemma 2 (variance bound: the Variance Reduction Law, invariant 1)
Let `Y_i = grad f_i(w) - grad f_i(w~)` and `Ybar = (1/n) * sum_i Y_i`. Then

```
g_hat(w) - grad F(w) = Ybar_B - Ybar,
```

because `g_hat(w) = Ybar_B + mu` and `grad F(w) = Ybar + mu`. Hence:

* sampling **with replacement**: `E||g_hat - grad F||^2 = (1/b) * sigma^2`,
* sampling **without replacement** (this repo's sampler): `= (1/b) * ((n-b)/(n-1)) * sigma^2`,

where `sigma^2 = (1/n) * sum_i ||Y_i - Ybar||^2`. In both cases, since variance is at most the second moment and `||Y_i|| <= L_i ||w - w~||`,

```
E||g_hat(w) - grad F(w)||^2  <=  (Lbar2 / b) * ||w - w~||^2.
```

*Proof.* The two displayed identities are the standard variance of a sample mean (with and without the finite-population correction). Then `sigma^2 <= (1/n) * sum_i ||Y_i||^2 <= (1/n) * sum_i L_i^2 ||w - w~||^2`. ∎

**Consequence.** The variance of `g_hat` is controlled by the *snapshot distance* `||w - w~||`, not by how noisy the data are. Plain mini-batch SGD has no such control. At a stationary point of a noisy problem, per-sample gradients `grad f_i(w*)` do not vanish, so SGD keeps an irreducible noise floor `(1/b) * (1/n) * sum_i ||grad f_i(w*)||^2` (with replacement). Once `w` is close to `w~` the variance of `g_hat` goes to zero regardless. This is what "Variance Reduction Law: `E||g_hat - grad F||^2 -> 0` as `w -> w*`" means precisely: it needs `w_t -> w*` **and** `w~ -> w*`, hence the snapshot refresh.

### Corollary 2a (exact identity for least squares)
`Y_i = x_i x_i^T (w - w~)`, so `g_hat - grad F = (H_B - H)(w - w~)` with `H_B = (1/b) * sum_{i in B} x_i x_i^T` and `H = (1/n) * X^T X`. The error is **linear** in the displacement, so scaling the displacement by `s` scales the variance by exactly `s^2`. `tests/test_math.py` uses this: a 10x smaller displacement must give a 100x smaller variance, and zero displacement must give zero variance.

### Proposition 3 (staleness-controlled variance for the adaptive engine)
The engine clips every coordinate of its normalised step: `|update_j| <= c` (`c = update_clip`). So one step moves each coordinate by at most `lr * c`, and after `k <= K` steps since the last snapshot, `||w - w~||_2 <= sqrt(d) * lr * c * k`. Combining with Lemma 2:

```
E||g_hat - grad F||^2  <=  (Lbar2 * d / b) * (lr * c * K)^2,       K = snapshot_interval.
```

*Proof.* Coordinatewise displacement `<= lr * c * k`, so `||w - w~||_2^2 <= d * (lr * c * k)^2`; substitute in Lemma 2. ∎

This holds for *any* adaptive normalisation as long as the update is clipped, so it survives the analytic difficulties of Adam-type schemes (Section 4). It is a design rule: **choose `snapshot_interval` so that the right-hand side is below the target variance budget.** It is very loose (it assumes worst-case coordinate movement): in Experiment 9, across snapshot intervals from 8 to 1024, measured variance was at most 2.2e-4 of the bound and typically around 1e-7. It is a valid guarantee but not a tool for choosing `snapshot_interval` tightly. `experiments/` checks that measured variance never exceeds it; tightening it is an open improvement.

### Proposition 4 (numerical safety, invariant 4)
Let `G = 0.25 * sqrt(realmax)` for the working dtype. After sanitisation, every coordinate of `g_hat` lies in `[-G, G]` (NaN -> 0, ±Inf -> ±G). Then by induction `|m_t| <= G` and `0 <= v_t <= G^2` (convex combinations, start at 0), all finite. The denominator satisfies `denom >= floor > 0`, and the final ratio is clipped to `[-c, c]`, so no step can be NaN or Inf. In float32 the unclipped quotient `|m_t| / (denom * bias1) <= G / (floor * (1 - beta1))` is itself finite whenever `1 - beta1 >= 1e-4`, so the clip acts on a finite number rather than repairing an overflow. ∎

The safeguard is a *clamp* on the denominator, not an additive `eps`. That is why a floor of `1e-15` is meaningful even in float32, where adding `1e-15` to a moderate number would do nothing.

## 3. Cited results

**Theorem (Johnson & Zhang, NeurIPS 2013, Thm 1).** Let each `f_i` be `L`-smooth and `F` be `gamma`-strongly convex. Run SVRG with step `eta < 1/(2L)` and `m` inner steps per epoch, with the SGD-form update `w <- w - eta * g_hat`, taking the next snapshot as a randomly chosen inner iterate. Then

```
E[F(w~_s) - F(w*)]  <=  alpha^s * (F(w~_0) - F(w*)),
alpha = 1 / (gamma * eta * (1 - 2 L eta) * m)  +  2 L eta / (1 - 2 L eta)  <  1.
```

For example, `eta = 0.1 / L` and `m = 50 L / gamma` give `alpha = 0.5`. This is *linear* convergence with constant step size, which plain SGD cannot achieve on a noisy finite sum (it needs decaying steps and converges sublinearly).

This applies to the **non-adaptive** variants in the ablation (`svrg_momentum` with `beta1 = 0` is the theorem's algorithm). It is not re-proved here. Two gaps between the theorem and the code are worth stating: the theorem is for Option II snapshots (a uniformly random inner iterate) while the engine takes the last iterate (Option I), and it is for single-sample steps with uniform sampling while the engine uses mini-batches without replacement (a mini-batch average of `L`-smooth convex functions is `L`-smooth, so the mini-batch version is covered with the same constants, conservatively).

### Proposition 5 (SVRG with a frozen diagonal preconditioner; explains Experiment 6)
Let `D` be a fixed positive diagonal matrix and consider the preconditioned SVRG step
`w <- w - eta * D^{-1} g_hat`, with `g_hat` the usual variance-reduced gradient. Define the rescaled problem
`f~_i(u) = f_i(D^{-1/2} u)`, `F~(u) = F(D^{-1/2} u)`. Suppose each `f~_i` is `L_D`-smooth, i.e.
`lambda_max(D^{-1/2} Hess f_i D^{-1/2}) <= L_D`, and `F~` is `gamma_D`-strongly convex. Then the Johnson–Zhang theorem
(Section 3) applies verbatim with `(L_D, gamma_D)`:

```
E[F(w~_s) - F(w*)]  <=  alpha_D^s * (F(w~_0) - F(w*)),
alpha_D = 1 / (gamma_D * eta * (1 - 2 L_D eta) * m)  +  2 L_D eta / (1 - 2 L_D eta).
```

*Proof.* Put `u = D^{1/2} w`. Then `grad f~_i(u) = D^{-1/2} grad f_i(w)`, so the SVRG estimator in `u`-coordinates is `D^{-1/2} g_hat`, and the plain SVRG step `u <- u - eta * (D^{-1/2} g_hat)` is exactly `D^{1/2}` applied to the preconditioned step in `w`. The two iterations are the same sequence in different coordinates (this is checked numerically in `tests/test_math.py`), and `F~(u) = F(w)`, so the function values coincide. The hypotheses are the theorem's hypotheses for `{f~_i}`. ∎

**What it says.** The rate depends on the *preconditioned* condition number `kappa_D = L_D / gamma_D`. For least squares,
`L_D = max_i ||D^{-1/2} x_i||^2` and `gamma_D = lambda_min(D^{-1/2} H D^{-1/2})` with `H = X^T X / n`. With the Jacobi choice
`D = diag(H)`, `kappa_D` is invariant to rescaling any feature column, because `D^{-1/2} H D^{-1/2}` is the correlation matrix of the
features. Jacobi scaling is within a factor of the dimension `d` of the best possible diagonal scaling (van der Sluis, 1969).
So a feature with a thousand-fold larger scale inflates the *raw* `kappa` by orders of magnitude and the preconditioned
`kappa_D` not at all. Experiment 10 measures this: on the badly scaled problem of Experiment 6, `kappa` falls from about
1e7 to about 80, while on equal-variance features it does not move (81 to 82). That matches Experiment 6, where adaptive scaling
helped by orders of magnitude, and Experiments 1, 4 and 5, where it did not help.

**Tested directly (Experiment 11).** Taking every hyper-parameter from the Johnson-Zhang recipe with `D = diag(X^T X / n)` (`eta = 0.1 / L_D`, `m = ceil(50 L_D / gamma_D)`, single-sample steps, random-iterate snapshots, so `alpha = 1/2`) and **no tuning**, the measured mean epoch-to-epoch contraction of the gap was 0.08 to 0.10, inside the guaranteed 0.5 in expectation (individual epochs reached 0.86 to 0.93, which the in-expectation guarantee permits). The method reached float64 round-off on all 20 seeds of both the well-scaled and the badly scaled problem. It beat the best tuned adaptive variant on badly scaled features on 20/20 seeds (see `RESULTS.md`). The proposition therefore predicted a method, with its hyper-parameters, that outperformed the project's more elaborate engine. The pattern carried to logistic regression with badly scaled features (Experiment 12, tuned rather than theory-prescribed, because the global strong-convexity modulus of the logistic loss is only the L2 weight and the worst-case recipe is uninformative): Jacobi SVRG reached round-off on 12/12 seeds and needed 2.9x fewer evaluations than Coordinate SVRG with cosine decay. There, constant-rate Coordinate SVRG did not plateau, so the plateau seen on least squares is problem-dependent.

**What it does not say.** The engine's preconditioner `diag(sqrt(v_hat_t))` is **not frozen** and tracks *gradient* scale, not
Hessian-diagonal scale; the proposition covers the frozen-diagonal idealisation only. The link to the actual adaptive engine is an
empirical consistency, not a theorem. It also explains why a *constant-rate* adaptive method plateaus while a fixed preconditioner would
not (the proposition's preconditioner does not renormalise as the gradient shrinks).

## 4. What is open (no claim made)

**The adaptive variant has no convergence theorem here.** The update is `w <- w - lr * clip(m_hat / max(sqrt(v_hat), floor), ±c)`. Two known difficulties apply:

1. Adam-type methods can fail to converge even on convex problems with noise-free structure when `v_t` forgets large gradients (Reddi, Kale & Kumar, ICLR 2018).
2. Normalisation makes the step size roughly `lr` per coordinate even when gradients are tiny, which by itself prevents exact convergence at a constant `lr`, unless `v_t` retains the memory of earlier, larger gradients.

Empirically (see `docs/RESULTS.md`), Coordinate SVRG at a constant rate converges far past the noise floor at which Adam with its own tuned learning rate stalls on well-scaled problems, but on a badly scaled problem it plateaus (median gap about 7e-4) exactly as difficulty (2) predicts, and a decaying learning rate removes the plateau (median gap about 4e-11, 20 seeds). This is evidence, not proof. The working explanation is that `beta2 = 0.999` gives `v_t` a memory of about 1000 steps, longer than the horizon of a run, so the effective preconditioner is close to a fixed diagonal matrix and the analysis of preconditioned SVRG applies approximately. **Conjecture:** with `beta2` close enough to 1 relative to the run length, linear convergence holds up to a horizon-dependent floor. Proving or refuting this is open; the experiments include an SVRG + momentum variant (no normalisation) precisely so the effect of the normalisation can be separated from the effect of variance reduction.

**Non-convex and deep learning.** Variance reduction is known to help far less on deep networks than on convex finite sums (Defazio & Bottou, NeurIPS 2019). Non-convex SVRG has weaker guarantees (Reddi, Hefny, Sra, Póczos & Smola, ICML 2016). This project makes no claim about deep-learning speedups; its stated regime is finite-sum, low-SNR estimation (calibration of linear/generalised-linear signal models on tick data).

**Scale.** On the default problem (`n = 4096`, `d = 32`) a closed-form solve is instant and exact. SVRG's advantage is asymptotic in `n` and `d`, where per-iteration cost matters. The simulator is a controlled testbed for optimizer behaviour, not evidence that SVRG beats direct solvers at this size.

## 5. References

1. R. Johnson, T. Zhang. *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction.* NeurIPS 2013.
2. D. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
3. S. Reddi, S. Kale, S. Kumar. *On the Convergence of Adam and Beyond.* ICLR 2018.
4. S. Reddi, A. Hefny, S. Sra, B. Póczos, A. Smola. *Stochastic Variance Reduction for Nonconvex Optimization.* ICML 2016.
5. A. Defazio, L. Bottou. *On the Ineffectiveness of Variance Reduced Optimization for Deep Learning.* NeurIPS 2019.
6. A. Defazio, F. Bach, S. Lacoste-Julien. *SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives.* NeurIPS 2014.
7. L. Nguyen, J. Liu, K. Scheinberg, M. Takáč. *SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient.* ICML 2017.
8. Z. Allen-Zhu. *Katyusha: The First Direct Acceleration of Stochastic Gradient Methods.* STOC 2017.
9. A. van der Sluis. *Condition numbers and equilibration of matrices.* Numerische Mathematik 14, 1969.
