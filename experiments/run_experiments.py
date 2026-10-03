"""Reproducible experiment suite (NumPy only, no GPU needed).

Experiment 1 - tuned 2x2 ablation
    {variance reduction on/off} x {adaptive scaling on/off}, in three noise
    regimes. Every variant gets its own learning-rate sweep on *tuning* seeds,
    then is scored on disjoint *evaluation* seeds. Progress is measured against
    sample-gradient evaluations (SVRG's snapshot passes and double gradients are
    charged), never against step count.

Experiment 2 - the Variance Reduction Law along a real trajectory
    Measures E||g_hat - grad f(w_t)||^2 for SVRG against plain mini-batch SGD
    while the optimizer runs, and checks it against the analytical bounds in
    docs/THEORY.md (Lemma 2 and Proposition 3).

Experiment 3 - walk-forward alpha tracking (quant evaluation)
    Rolling-window re-estimation of a drifting hidden alpha under a fixed
    compute budget per window; out-of-sample IC, hit rate, and per-tick Sharpe
    of a sign strategy, against a closed-form OLS reference.

Usage (from repo root):
    python -m experiments.run_experiments                # everything
    python -m experiments.run_experiments --quick        # smoke test
    python -m experiments.run_experiments --only 1 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import (  # noqa: E402
    DeterministicBatchSampler,
    MarketDataset,
    RegimeConfig,
    generate_market,
    load_config,
)
from src.reference_numpy import ReferenceConfig, ReferenceSVRG  # noqa: E402

FloatArray = npt.NDArray[np.float64]

# name -> (variance_reduction, adaptive, lr_schedule)
VARIANTS: Dict[str, Tuple[bool, bool, str]] = {
    "sgd_momentum": (False, False, "const"),
    "sgd_momentum_cosine": (False, False, "cosine"),
    "adam": (False, True, "const"),
    "adam_cosine": (False, True, "cosine"),
    "svrg_momentum": (True, False, "const"),
    "svrg_adam": (True, True, "const"),
    "svrg_adam_cosine": (True, True, "cosine"),
}
VARIANT_LABELS: Dict[str, str] = {
    "sgd_momentum": "SGD + momentum",
    "sgd_momentum_cosine": "SGD + momentum, cosine lr",
    "adam": "Adam (clamped)",
    "adam_cosine": "Adam (clamped), cosine lr",
    "svrg_momentum": "SVRG + momentum",
    "svrg_adam": "Coordinate SVRG (ours)",
    "svrg_adam_cosine": "Coordinate SVRG, cosine lr (ours)",
}


# --------------------------------------------------------------------------- #
# Objective and optimisation loop
# --------------------------------------------------------------------------- #
class LeastSquares:
    """f(w) = 1/(2n) ||Xw - y||^2 with mini-batch, full and reference metrics."""

    def __init__(self, features: FloatArray, targets: FloatArray) -> None:
        self.features: FloatArray = features
        self.targets: FloatArray = targets
        self.n: int = int(features.shape[0])
        self.d: int = int(features.shape[1])
        self._optimum: Optional[FloatArray] = None
        self._optimal_loss: Optional[float] = None
        self._hessian: Optional[FloatArray] = None

    def grad(self, w: FloatArray, rows: npt.NDArray[np.int64]) -> FloatArray:
        xb: FloatArray = self.features[rows]
        return xb.T @ (xb @ w - self.targets[rows]) / float(len(rows))

    def full_grad(self, w: FloatArray) -> FloatArray:
        return self.features.T @ (self.features @ w - self.targets) / float(self.n)

    def loss(self, w: FloatArray) -> float:
        r: FloatArray = self.features @ w - self.targets
        return float(0.5 * np.mean(r * r))

    @property
    def optimum(self) -> FloatArray:
        if self._optimum is None:
            self._optimum = np.linalg.lstsq(self.features, self.targets, rcond=None)[0]
        return self._optimum

    @property
    def optimal_loss(self) -> float:
        if self._optimal_loss is None:
            self._optimal_loss = self.loss(self.optimum)
        return self._optimal_loss

    @property
    def hessian(self) -> FloatArray:
        if self._hessian is None:
            self._hessian = self.features.T @ self.features / float(self.n)
        return self._hessian

    def gap(self, w: FloatArray) -> float:
        """f(w) - f(w*) via the exact quadratic form 0.5 d'Hd, d = w - w*.

        Subtracting two losses cancels catastrophically below ~1e-16; the
        quadratic form keeps full relative precision all the way down.
        """
        delta: FloatArray = w - self.optimum
        return float(0.5 * delta @ (self.hessian @ delta))

    def prepare(self) -> None:
        """Force lazy reference quantities so they never pollute wall-clock timings."""
        _ = self.optimal_loss
        _ = self.hessian

    def smoothness_second_moment(self) -> float:
        """(1/n) sum_i L_i^2 with L_i = ||x_i||^2 (per-sample smoothness)."""
        l_i: FloatArray = np.sum(self.features ** 2, axis=1)
        return float(np.mean(l_i ** 2))


class LogisticRegression:
    """L2-regularised logistic loss for predicting the sign of the return.

    f(w) = (1/n) sum_i log(1 + exp(-z_i x_i.w)) + (l2/2) ||w||^2,   z_i in {-1, +1}.
    Strongly convex (modulus l2), per-sample smoothness L_i = ||x_i||^2 / 4 + l2.
    The optimum is found by damped Newton to ||grad|| < 1e-13.
    """

    def __init__(self, features: FloatArray, labels: FloatArray, l2: float) -> None:
        self.features: FloatArray = features
        self.labels: FloatArray = labels
        self.l2: float = l2
        self.n: int = int(features.shape[0])
        self.d: int = int(features.shape[1])
        self._optimum: Optional[FloatArray] = None
        self._optimal_loss: float = float("nan")
        self._hessian_opt: Optional[FloatArray] = None

    @staticmethod
    def _sigmoid_neg(margin: FloatArray) -> FloatArray:
        """sigma(-m) = 1 / (1 + exp(m)), overflow-safe."""
        return 1.0 / (1.0 + np.exp(np.clip(margin, -500.0, 500.0)))

    def grad(self, w: FloatArray, rows: npt.NDArray[np.int64]) -> FloatArray:
        xb: FloatArray = self.features[rows]
        z: FloatArray = self.labels[rows]
        s: FloatArray = self._sigmoid_neg(z * (xb @ w))
        return -(xb.T @ (z * s)) / float(len(rows)) + self.l2 * w

    def full_grad(self, w: FloatArray) -> FloatArray:
        s: FloatArray = self._sigmoid_neg(self.labels * (self.features @ w))
        return -(self.features.T @ (self.labels * s)) / float(self.n) + self.l2 * w

    def loss(self, w: FloatArray) -> float:
        margin: FloatArray = self.labels * (self.features @ w)
        return float(np.mean(np.logaddexp(0.0, -margin)) + 0.5 * self.l2 * float(w @ w))

    def _hessian(self, w: FloatArray) -> FloatArray:
        s: FloatArray = self._sigmoid_neg(self.labels * (self.features @ w))
        weights: FloatArray = s * (1.0 - s)
        return (self.features.T * weights) @ self.features / float(self.n) + self.l2 * np.eye(self.d)

    def prepare(self) -> None:
        if self._optimum is not None:
            return
        w: FloatArray = np.zeros(self.d)
        for _ in range(100):
            g: FloatArray = self.full_grad(w)
            if float(np.linalg.norm(g)) < 1e-13:
                break
            step: FloatArray = -np.linalg.solve(self._hessian(w), g)
            t: float = 1.0
            f0: float = self.loss(w)
            while self.loss(w + t * step) > f0 + 1e-4 * t * float(g @ step) and t > 1e-12:
                t *= 0.5
            w = w + t * step
        self._optimum = w
        self._optimal_loss = self.loss(w)
        self._hessian_opt = self._hessian(w)

    @property
    def optimum(self) -> FloatArray:
        self.prepare()
        return self._optimum  # type: ignore[return-value]

    def gap(self, w: FloatArray) -> float:
        """f(w) - f(w*); below 1e-7 switch to the exact local quadratic form.

        Differencing two O(1) losses cannot resolve gaps below ~1e-16; the quadratic
        form 0.5 d'H*d keeps full relative precision near the optimum.
        """
        self.prepare()
        direct: float = self.loss(w) - self._optimal_loss
        if direct >= 1e-7:
            return direct
        delta: FloatArray = w - self._optimum  # type: ignore[operator]
        return float(0.5 * delta @ (self._hessian_opt @ delta))  # type: ignore[operator]


class MLPRegression:
    """One-hidden-layer tanh network, mean-squared loss: a small NON-CONVEX finite sum.

    Parameters are flattened as [W1 (in x h), b1 (h), w2 (h), b2 (1)]. ``d`` is the
    flat parameter dimension (what the optimizers see); ``gap`` is the squared
    full-gradient norm, the standard stationarity measure for non-convex problems,
    because no global optimum is known.
    """

    def __init__(self, features: FloatArray, targets: FloatArray, hidden: int, init_seed: int) -> None:
        self.features: FloatArray = features
        self.targets: FloatArray = targets
        self.n: int = int(features.shape[0])
        self.in_dim: int = int(features.shape[1])
        self.hidden: int = hidden
        self.d: int = self.in_dim * hidden + hidden + hidden + 1
        rng = np.random.default_rng(init_seed)
        w1: FloatArray = rng.standard_normal((self.in_dim, hidden)) / math.sqrt(self.in_dim)
        w2: FloatArray = rng.standard_normal(hidden) / math.sqrt(hidden)
        self.init_weights: FloatArray = np.concatenate([w1.ravel(), np.zeros(hidden), w2, np.zeros(1)])
        self._all_rows: npt.NDArray[np.int64] = np.arange(self.n, dtype=np.int64)

    def _unpack(self, w: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
        i, h = self.in_dim, self.hidden
        return (w[: i * h].reshape(i, h), w[i * h: i * h + h], w[i * h + h: i * h + 2 * h], float(w[-1]))

    def grad(self, w: FloatArray, rows: npt.NDArray[np.int64]) -> FloatArray:
        w1, b1, w2, b2 = self._unpack(w)
        x: FloatArray = self.features[rows]
        hidden_act: FloatArray = np.tanh(x @ w1 + b1)
        resid: FloatArray = (hidden_act @ w2 + b2 - self.targets[rows]) / float(len(rows))
        d_hidden: FloatArray = (resid[:, None] * w2[None, :]) * (1.0 - hidden_act * hidden_act)
        return np.concatenate([
            (x.T @ d_hidden).ravel(), d_hidden.sum(axis=0), hidden_act.T @ resid, np.array([resid.sum()]),
        ])

    def full_grad(self, w: FloatArray) -> FloatArray:
        return self.grad(w, self._all_rows)

    def loss(self, w: FloatArray) -> float:
        w1, b1, w2, b2 = self._unpack(w)
        pred: FloatArray = np.tanh(self.features @ w1 + b1) @ w2 + b2
        return float(0.5 * np.mean((pred - self.targets) ** 2))

    def gap(self, w: FloatArray) -> float:
        g: FloatArray = self.full_grad(w)
        return float(g @ g)

    def prepare(self) -> None:
        return None


@dataclass
class RunResult:
    weights: FloatArray
    final_gap: float
    evals_used: int
    curve_evals: List[float]
    curve_gaps: List[float]
    curve_seconds: List[float]
    diverged: bool


def optimize(
    objective: Any,
    sampler: DeterministicBatchSampler,
    config: ReferenceConfig,
    budget_evals: int,
    w0: Optional[FloatArray] = None,
    checkpoints: Optional[Sequence[float]] = None,
    track_gap: bool = True,
    lr_schedule: str = "const",
) -> RunResult:
    """Run one optimizer for a fixed budget of sample-gradient evaluations.

    ``objective`` needs ``n``, ``d``, ``grad``, ``full_grad`` and ``gap``; call
    ``objective.prepare()`` first so reference solves are not timed.
    """
    n: int = objective.n
    batch: int = sampler.batch_size
    w: FloatArray = np.zeros(objective.d) if w0 is None else np.array(w0, dtype=np.float64)
    opt: ReferenceSVRG = ReferenceSVRG(config, objective.d)
    per_step_cost: int = batch * (2 if config.variance_reduction else 1)
    cps: List[float] = list(checkpoints) if checkpoints is not None else []
    ci: int = 0
    evals: int = 0
    step: int = 0
    curve_e: List[float] = []
    curve_g: List[float] = []
    curve_t: List[float] = []
    diverged: bool = False
    t_start: float = time.perf_counter()

    while evals < budget_evals:
        if opt.needs_snapshot:
            opt.refresh_snapshot(w, objective.full_grad(w))
            evals += n
        if lr_schedule == "cosine":
            opt.lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(evals / budget_evals, 1.0)))
        rows = sampler.batch(step)
        g_live: FloatArray = objective.grad(w, rows)
        g_snap: Optional[FloatArray] = (
            objective.grad(opt.snapshot, rows) if config.variance_reduction else None  # type: ignore[arg-type]
        )
        evals += per_step_cost
        w = opt.step(w, g_live, g_snap)
        step += 1
        while ci < len(cps) and evals >= cps[ci]:
            curve_e.append(float(evals))
            curve_g.append(objective.gap(w) if track_gap else float("nan"))
            curve_t.append(time.perf_counter() - t_start)
            ci += 1
        if step % 32 == 0 and (not np.all(np.isfinite(w)) or float(np.linalg.norm(w)) > 1e6):
            diverged = True
            break

    if diverged:
        while ci < len(cps):
            curve_e.append(float(cps[ci]))
            curve_g.append(float("inf"))
            curve_t.append(float("inf"))
            ci += 1
    final_gap: float = float("inf") if diverged else (objective.gap(w) if track_gap else float("nan"))
    return RunResult(w, final_gap, evals, curve_e, curve_g, curve_t, diverged)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_dataset(base: RegimeConfig, seed: int, noise_sigma: float) -> MarketDataset:
    return generate_market(dataclasses.replace(base, seed=seed, noise_sigma=noise_sigma))


def quartiles(values: Sequence[float]) -> Tuple[float, float, float]:
    arr: FloatArray = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
    )


def reference_config(opt_cfg: Dict[str, Any], variant: str, lr: float) -> ReferenceConfig:
    vr, adaptive, _ = VARIANTS[variant]
    return ReferenceConfig(
        lr=lr,
        beta1=float(opt_cfg["betas"][0]),
        beta2=float(opt_cfg["betas"][1]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        update_clip=float(opt_cfg["update_clip"]),
        snapshot_interval=int(opt_cfg["snapshot_interval"]),
        variance_reduction=vr,
        adaptive=adaptive,
    )


def fmt(x: float) -> str:
    if math.isinf(x):
        return "diverged"
    if math.isnan(x):
        return "n/a"
    return f"{x:.2e}"


# --------------------------------------------------------------------------- #
# Experiment 1: tuned 2x2 ablation
# --------------------------------------------------------------------------- #
ObjectiveFactory = Callable[[int, float], Any]


def ls_factory(base: RegimeConfig) -> ObjectiveFactory:
    def make(seed: int, sigma: float) -> Any:
        ds = make_dataset(base, seed, sigma)
        return LeastSquares(ds.features, ds.targets)
    return make


def scaled_ls_factory(base: RegimeConfig, decades: float) -> ObjectiveFactory:
    """Least squares with column scales spread log-uniformly over 10**(+-decades/2).

    Unnormalised quant features (price levels, volumes, spreads) look like this.
    Equal-variance features give a diagonal preconditioner nothing to fix, so this
    is the regime where adaptive coordinate scaling should earn its keep, if ever.
    """
    def make(seed: int, sigma: float) -> Any:
        ds = make_dataset(base, seed, sigma)
        scale_rng = np.random.default_rng(seed + 17)
        scales: FloatArray = 10.0 ** scale_rng.uniform(-decades / 2.0, decades / 2.0, size=ds.n_features)
        return LeastSquares(ds.features * scales, ds.targets)
    return make


def logistic_factory(base: RegimeConfig, l2: float) -> ObjectiveFactory:
    def make(seed: int, sigma: float) -> Any:
        ds = make_dataset(base, seed, sigma)
        labels: FloatArray = np.where(ds.targets >= 0.0, 1.0, -1.0)
        return LogisticRegression(ds.features, labels, l2)
    return make


def run_ablation(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    factory: ObjectiveFactory,
    *,
    tag: str,
    regimes: Dict[str, float],
    epochs: int,
    batch_size: int,
    tune_seeds: int,
    eval_seeds: int,
    seed_offset: int,
    lr_points: int,
    snapshot_interval: Optional[int] = None,
    time_direct_solve: bool = False,
    tune_on_loss: bool = False,
) -> Dict[str, Any]:
    """Tuned variant comparison: lr sweep on tuning seeds, scoring on disjoint seeds."""
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    if snapshot_interval is not None:
        opt_cfg["snapshot_interval"] = snapshot_interval
    lr_grid: FloatArray = 10.0 ** np.linspace(args.lr_min_exp, args.lr_max_exp, lr_points)
    seed0: int = int(cfg["seed"]) + seed_offset
    results: Dict[str, Any] = {
        "lr_grid": lr_grid.tolist(), "regimes": {}, "budget_epochs": epochs,
        "target_ratio": args.target_ratio, "batch_size": batch_size,
        "snapshot_interval": int(opt_cfg["snapshot_interval"]),
    }

    for regime, sigma in regimes.items():
        tune_sets: List[Any] = [factory(seed0 + 1000 + i, sigma) for i in range(tune_seeds)]
        eval_sets: List[Any] = [factory(seed0 + 2000 + i, sigma) for i in range(eval_seeds)]
        eval_seed_ids: List[int] = [seed0 + 2000 + i for i in range(eval_seeds)]
        for obj in tune_sets + eval_sets:
            obj.prepare()
        n: int = tune_sets[0].n
        budget: int = int(epochs * n)
        checkpoints: FloatArray = np.geomspace(4 * n, budget, args.curve_points)
        regime_out: Dict[str, Any] = {"sigma": sigma, "n": n, "d": tune_sets[0].d, "variants": {}}
        if time_direct_solve:
            t0 = time.perf_counter()
            np.linalg.lstsq(eval_sets[0].features, eval_sets[0].targets, rcond=None)
            regime_out["direct_solve_seconds"] = time.perf_counter() - t0
            xe, ye = eval_sets[0].features, eval_sets[0].targets
            t0 = time.perf_counter()
            np.linalg.solve(xe.T @ xe, xe.T @ ye)
            regime_out["normal_equations_seconds"] = time.perf_counter() - t0

        for variant in VARIANTS:
            tune_scores: List[float] = []
            for lr in lr_grid:
                gaps: List[float] = []
                for j, obj in enumerate(tune_sets):
                    sampler = DeterministicBatchSampler(n, batch_size, seed0 + 1000 + j)
                    run = optimize(
                        obj, sampler, reference_config(opt_cfg, variant, float(lr)), budget,
                        w0=getattr(obj, "init_weights", None), lr_schedule=VARIANTS[variant][2],
                    )
                    gaps.append(run.final_gap if not tune_on_loss else obj.loss(run.weights))
                tune_scores.append(float(np.median(gaps)))
            best_idx: int = int(np.argmin(np.where(np.isfinite(tune_scores), tune_scores, np.inf)))
            best_lr: float = float(lr_grid[best_idx])

            final_gaps: List[float] = []
            final_losses: List[float] = []
            curves: List[List[float]] = []
            curve_secs: List[List[float]] = []
            evals_to_target: List[float] = []
            secs_to_target: List[float] = []
            for obj, seed in zip(eval_sets, eval_seed_ids):
                sampler = DeterministicBatchSampler(n, batch_size, seed)
                initial_gap: float = obj.gap(
                    np.zeros(obj.d) if getattr(obj, "init_weights", None) is None else obj.init_weights
                )
                run = optimize(
                    obj, sampler, reference_config(opt_cfg, variant, best_lr), budget,
                    w0=getattr(obj, "init_weights", None),
                    checkpoints=checkpoints, lr_schedule=VARIANTS[variant][2],
                )
                final_gaps.append(run.final_gap)
                final_losses.append(obj.loss(run.weights))
                curves.append(run.curve_gaps)
                curve_secs.append(run.curve_seconds)
                target: float = args.target_ratio * initial_gap
                hit_idx: List[int] = [i for i, g in enumerate(run.curve_gaps) if g <= target]
                evals_to_target.append(run.curve_evals[hit_idx[0]] if hit_idx else float("inf"))
                secs_to_target.append(run.curve_seconds[hit_idx[0]] if hit_idx else float("inf"))
            q25, q50, q75 = quartiles(final_gaps)
            curve_arr: FloatArray = np.asarray(curves, dtype=np.float64)
            regime_out["variants"][variant] = {
                "best_lr": best_lr,
                "best_lr_at_grid_edge": bool(best_idx in (0, len(lr_grid) - 1)),
                "tune_scores": tune_scores,
                "final_gap_q25": q25,
                "final_gap_median": q50,
                "final_gap_q75": q75,
                "final_gaps": final_gaps,
                "final_loss_median": float(np.median(final_losses)),
                "evals_to_target_median": float(np.median(evals_to_target)),
                "seconds_to_target_median": float(np.median(secs_to_target)),
                "curve_evals": checkpoints.tolist(),
                "curve_seconds_median": np.median(np.asarray(curve_secs, dtype=np.float64), axis=0).tolist(),
                "curve_q25": np.percentile(curve_arr, 25, axis=0).tolist(),
                "curve_median": np.percentile(curve_arr, 50, axis=0).tolist(),
                "curve_q75": np.percentile(curve_arr, 75, axis=0).tolist(),
            }
            edge_flag: str = " (GRID EDGE)" if best_idx in (0, len(lr_grid) - 1) else ""
            print(f"  [{tag}] {regime:<9}{variant:<21} lr={best_lr:.2e}{edge_flag}  final gap median {fmt(q50)}")
        results["regimes"][regime] = regime_out
    return results


def experiment_ablation(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 1: least squares, default regime sizes."""
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    out = run_ablation(
        cfg, args, ls_factory(base), tag="exp1",
        regimes=dict(cfg["environment"]["noise_sigma_levels"]),
        epochs=args.epochs, batch_size=int(cfg["training"]["batch_size"]),
        tune_seeds=args.tune_seeds, eval_seeds=args.eval_seeds,
        seed_offset=0, lr_points=args.lr_points,
    )
    out["description"] = (f"Least squares, n={base.n_samples}, d={base.n_features}, batch "
                          f"{out['batch_size']}, snapshot every {out['snapshot_interval']} steps")
    return out


def experiment_hetero(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 6: badly scaled features, the adaptive scaling's home turf."""
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    out = run_ablation(
        cfg, args, scaled_ls_factory(base, args.het_decades), tag="exp6",
        regimes={"volatile": sigma},
        epochs=args.epochs, batch_size=int(cfg["training"]["batch_size"]),
        tune_seeds=args.tune_seeds, eval_seeds=args.eval_seeds,
        seed_offset=7000, lr_points=args.lr_points,
    )
    out["description"] = (f"Least squares with column scales spread over {args.het_decades:g} decades, "
                          f"n={base.n_samples}, d={base.n_features}, batch {out['batch_size']}, "
                          f"snapshot every {out['snapshot_interval']} steps")
    return out


def mlp_factory(base: RegimeConfig, hidden: int) -> ObjectiveFactory:
    """Nonlinear target tanh(2 * clean signal) plus Gaussian noise: a non-convex fit."""
    def make(seed: int, sigma: float) -> Any:
        ds = make_dataset(base, seed, sigma)
        clean: FloatArray = np.einsum("td,td->t", ds.features, ds.alpha_path)
        targets: FloatArray = np.tanh(2.0 * clean) + 0.5 * ds.gaussian_noise
        return MLPRegression(ds.features, targets, hidden, init_seed=seed + 31)
    return make


def experiment_nonconvex(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 7: non-convex one-hidden-layer network; no guarantees apply."""
    base: RegimeConfig = dataclasses.replace(
        RegimeConfig.from_dict(cfg), n_features=args.nc_inputs, alpha_drift_sigma=0.0, regime_switch_prob=0.0,
    )
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    out = run_ablation(
        cfg, args, mlp_factory(base, args.nc_hidden), tag="exp7",
        regimes={"volatile": sigma},
        epochs=args.nc_epochs, batch_size=int(cfg["training"]["batch_size"]),
        tune_seeds=args.nc_tune_seeds, eval_seeds=args.nc_eval_seeds,
        seed_offset=8000, lr_points=args.nc_lr_points, tune_on_loss=True,
    )
    out["description"] = (f"Non-convex: one-hidden-layer tanh network ({args.nc_inputs}-{args.nc_hidden}-1, "
                          f"{out['regimes']['volatile']['d']} parameters), n={base.n_samples}, batch {out['batch_size']}, "
                          f"snapshot every {out['snapshot_interval']} steps. Learning rate tuned on final TRAIN LOSS; "
                          "the 'gap' column is the squared full-gradient norm (no global optimum is known)")
    out["metric"] = "squared full-gradient norm"
    out["report_loss"] = True
    return out


def preconditioned_constants(features: FloatArray) -> Dict[str, float]:
    """Smoothness / strong-convexity constants of least squares, raw and Jacobi-preconditioned.

    Raw:        L = max_i ||x_i||^2,  gamma = lambda_min(H),  H = X^T X / n.
    Jacobi D = diag(H):  L_D = max_i ||D^{-1/2} x_i||^2,  gamma_D = lambda_min(D^{-1/2} H D^{-1/2}).
    The SVRG rate of Proposition 5 depends on kappa = L / gamma, so kappa_D versus kappa is
    the predicted benefit of a (frozen) diagonal preconditioner.
    """
    n: int = features.shape[0]
    hess: FloatArray = features.T @ features / float(n)
    l_raw: float = float(np.max(np.sum(features ** 2, axis=1)))
    gamma: float = float(np.linalg.eigvalsh(hess)[0])
    scaled: FloatArray = features / np.sqrt(np.diag(hess))
    hess_d: FloatArray = scaled.T @ scaled / float(n)
    l_d: float = float(np.max(np.sum(scaled ** 2, axis=1)))
    gamma_d: float = float(np.linalg.eigvalsh(hess_d)[0])
    return {
        "L": l_raw, "gamma": gamma, "kappa": l_raw / gamma,
        "L_jacobi": l_d, "gamma_jacobi": gamma_d, "kappa_jacobi": l_d / gamma_d,
        "kappa_ratio": (l_raw / gamma) / (l_d / gamma_d),
    }


@dataclass
class DiagSvrgResult:
    final_gap: float
    evals: List[float]          # cumulative sample-gradient evaluations at each epoch end
    gaps: List[float]           # gap of the snapshot at each epoch end (index 0 = initial point)
    diverged: bool


def diag_precond_svrg(
    obj: Any,
    diag: FloatArray,
    eta: float,
    inner_steps: int,
    batch_size: int,
    budget_evals: int,
    snapshot_rule: str,
    seed: int,
    sampler: Optional[DeterministicBatchSampler] = None,
    w0: Optional[FloatArray] = None,
) -> DiagSvrgResult:
    """SVRG with a FROZEN diagonal preconditioner: w <- w - eta * D^{-1} g_hat.

    ``snapshot_rule`` is ``"random"`` (Johnson-Zhang Option II: the next snapshot is a uniformly
    random inner iterate, the setting of their theorem) or ``"last"`` (Option I, used by the engine).
    With ``sampler=None`` batches are drawn uniformly with replacement (the theorem's setting).
    """
    n: int = obj.n
    rng = np.random.default_rng(seed)
    snap: FloatArray = np.zeros(obj.d) if w0 is None else np.array(w0, dtype=np.float64)
    evals_out: List[float] = [0.0]
    gaps_out: List[float] = [obj.gap(snap)]
    evals: int = 0
    step: int = 0
    diverged: bool = False
    while evals < budget_evals:
        mu: FloatArray = obj.full_grad(snap)
        evals += n
        w: FloatArray = snap.copy()
        iterates: List[FloatArray] = []
        for _ in range(inner_steps):
            rows = rng.integers(0, n, size=batch_size) if sampler is None else sampler.batch(step)
            g: FloatArray = obj.grad(w, rows) - obj.grad(snap, rows) + mu
            w = w - eta * g / diag
            step += 1
            evals += 2 * batch_size
            if snapshot_rule == "random":
                iterates.append(w)
        snap = iterates[int(rng.integers(inner_steps))] if snapshot_rule == "random" else w
        gap: float = obj.gap(snap)
        evals_out.append(float(evals))
        gaps_out.append(gap)
        if not math.isfinite(gap) or gap > 1e12:
            diverged = True
            break
    return DiagSvrgResult(float("inf") if diverged else gaps_out[-1], evals_out, gaps_out, diverged)


def gaps_on_grid(evals: Sequence[float], gaps: Sequence[float], grid: Sequence[float]) -> List[float]:
    """Gap of the last completed epoch at or before each grid point (step function)."""
    e: FloatArray = np.asarray(evals, dtype=np.float64)
    g: FloatArray = np.asarray(gaps, dtype=np.float64)
    idx: npt.NDArray[np.int64] = np.searchsorted(e, np.asarray(grid, dtype=np.float64), side="right") - 1
    return [float(g[max(i, 0)]) for i in idx]


def theory_constants(features: FloatArray) -> Dict[str, float]:
    """Constants for the Johnson-Zhang recipe on least squares with Jacobi D = diag(X^T X / n)."""
    n: int = features.shape[0]
    diag: FloatArray = np.mean(features ** 2, axis=0)
    scaled: FloatArray = features / np.sqrt(diag)
    l_d: float = float(np.max(np.sum(scaled ** 2, axis=1)))
    gamma_d: float = float(np.linalg.eigvalsh(scaled.T @ scaled / float(n))[0])
    eta: float = 0.1 / l_d
    inner: int = int(math.ceil(50.0 * l_d / gamma_d))
    alpha: float = 1.0 / (gamma_d * eta * (1.0 - 2.0 * l_d * eta) * inner) + 2.0 * l_d * eta / (1.0 - 2.0 * l_d * eta)
    return {"L_D": l_d, "gamma_D": gamma_d, "eta": eta, "inner_steps": float(inner), "alpha": alpha}


def experiment_jacobi(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 11: a method read off Proposition 5, with theory-prescribed hyper-parameters.

    Jacobi-SVRG uses the frozen preconditioner D = diag(X^T X / n). The *theory* variant takes every
    hyper-parameter from the Johnson-Zhang recipe (eta = 0.1 / L_D, m = 50 L_D / gamma_D, single-sample
    steps, random-iterate snapshots): zero tuning. The *tuned* variant uses mini-batches, last-iterate
    snapshots and a learning-rate sweep like every other method. Datasets and seeds are identical to
    Experiments 1 and 6, so paired comparisons against their stored results are valid.
    """
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    batch_size: int = int(cfg["training"]["batch_size"])
    interval: int = int(cfg["optimizer"]["snapshot_interval"])
    lr_grid: FloatArray = 10.0 ** np.linspace(args.lr_min_exp, args.lr_max_exp, args.lr_points)
    problems = (
        ("well-scaled features", ls_factory(base), int(cfg["seed"]), "ablation"),
        (f"badly scaled features ({args.het_decades:g} decades)",
         scaled_ls_factory(base, args.het_decades), int(cfg["seed"]) + 7000, "hetero"),
    )
    out: Dict[str, Any] = {"problems": {}}
    for name, factory, seed0, stored_key in problems:
        tune_sets = [factory(seed0 + 1000 + i, sigma) for i in range(args.tune_seeds)]
        eval_sets = [factory(seed0 + 2000 + i, sigma) for i in range(args.eval_seeds)]
        for o in tune_sets + eval_sets:
            o.prepare()
        n: int = tune_sets[0].n
        budget: int = int(args.epochs * n)
        grid: FloatArray = np.geomspace(4 * n, budget, args.curve_points)

        # ---- theory-prescribed: no tuning ----
        theory_gaps: List[float] = []
        theory_curves: List[List[float]] = []
        to_target: List[float] = []
        ratios: List[float] = []
        consts_list: List[Dict[str, float]] = []
        for i, o in enumerate(eval_sets):
            consts = theory_constants(o.features)
            consts_list.append(consts)
            diag = np.mean(o.features ** 2, axis=0)
            run = diag_precond_svrg(o, diag, consts["eta"], int(consts["inner_steps"]), 1, budget,
                                    "random", seed=seed0 + 2000 + i)
            theory_gaps.append(run.final_gap)
            theory_curves.append(gaps_on_grid(run.evals, run.gaps, grid))
            target = args.target_ratio * run.gaps[0]
            hit = [e for e, g in zip(run.evals, run.gaps) if g <= target]
            to_target.append(hit[0] if hit else float("inf"))
            ratios += [b / a for a, b in zip(run.gaps[:-1], run.gaps[1:]) if a > 1e-20]
        theory_out = {
            "final_gaps": theory_gaps, "final_gap_median": float(np.median(theory_gaps)),
            "evals_to_target_median": float(np.median(to_target)),
            "curve_median": np.percentile(np.asarray(theory_curves), 50, axis=0).tolist(),
            "curve_q25": np.percentile(np.asarray(theory_curves), 25, axis=0).tolist(),
            "curve_q75": np.percentile(np.asarray(theory_curves), 75, axis=0).tolist(),
            "alpha_predicted": float(np.median([c["alpha"] for c in consts_list])),
            "kappa_D": float(np.median([c["L_D"] / c["gamma_D"] for c in consts_list])),
            "eta": float(np.median([c["eta"] for c in consts_list])),
            "inner_steps": int(np.median([c["inner_steps"] for c in consts_list])),
            "mean_epoch_contraction": float(np.mean(ratios)) if ratios else float("nan"),
            "max_epoch_contraction": float(np.max(ratios)) if ratios else float("nan"),
            "epoch_transitions": len(ratios),
        }

        # ---- tuned: mini-batch, last iterate, lr sweep ----
        scores: List[float] = []
        for lr in lr_grid:
            gaps = []
            for j, o in enumerate(tune_sets):
                diag = np.mean(o.features ** 2, axis=0)
                gaps.append(diag_precond_svrg(
                    o, diag, float(lr), interval, batch_size, budget, "last", seed=seed0 + 1000 + j,
                    sampler=DeterministicBatchSampler(n, batch_size, seed0 + 1000 + j)).final_gap)
            scores.append(float(np.median(gaps)))
        best_idx = int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))
        best_lr = float(lr_grid[best_idx])
        tuned_gaps: List[float] = []
        tuned_curves: List[List[float]] = []
        to_target_t: List[float] = []
        for i, o in enumerate(eval_sets):
            diag = np.mean(o.features ** 2, axis=0)
            run = diag_precond_svrg(o, diag, best_lr, interval, batch_size, budget, "last",
                                    seed=seed0 + 2000 + i,
                                    sampler=DeterministicBatchSampler(n, batch_size, seed0 + 2000 + i))
            tuned_gaps.append(run.final_gap)
            tuned_curves.append(gaps_on_grid(run.evals, run.gaps, grid))
            target = args.target_ratio * run.gaps[0]
            hit = [e for e, g in zip(run.evals, run.gaps) if g <= target]
            to_target_t.append(hit[0] if hit else float("inf"))
        tuned_out = {
            "best_lr": best_lr, "best_lr_at_grid_edge": bool(best_idx in (0, len(lr_grid) - 1)),
            "final_gaps": tuned_gaps, "final_gap_median": float(np.median(tuned_gaps)),
            "evals_to_target_median": float(np.median(to_target_t)),
            "curve_median": np.percentile(np.asarray(tuned_curves), 50, axis=0).tolist(),
            "curve_q25": np.percentile(np.asarray(tuned_curves), 25, axis=0).tolist(),
            "curve_q75": np.percentile(np.asarray(tuned_curves), 75, axis=0).tolist(),
        }
        out["problems"][name] = {
            "stored_key": stored_key, "curve_evals": grid.tolist(),
            "theory": theory_out, "tuned": tuned_out,
        }
        print(f"  [exp11] {name:<38} theory gap {fmt(theory_out['final_gap_median'])} "
              f"(mean epoch contraction {theory_out['mean_epoch_contraction']:.3f} vs bound {theory_out['alpha_predicted']:.2f}); "
              f"tuned gap {fmt(tuned_out['final_gap_median'])}")
    out["description"] = ("Jacobi-preconditioned SVRG on least squares, volatile regime; datasets and seeds identical to "
                          "Experiments 1 and 6")
    return out


def experiment_conditioning(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 10: predicted benefit of diagonal preconditioning (Proposition 5)."""
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    seed0: int = int(cfg["seed"]) + 10000
    out: Dict[str, Any] = {"problems": {}}
    for name, factory in (
        ("well-scaled features", ls_factory(base)),
        (f"badly scaled features ({args.het_decades:g} decades)", scaled_ls_factory(base, args.het_decades)),
    ):
        per_seed: List[Dict[str, float]] = [
            preconditioned_constants(factory(seed0 + i, sigma).features) for i in range(args.cond_seeds)
        ]
        out["problems"][name] = {
            key: float(np.median([row[key] for row in per_seed])) for key in per_seed[0]
        }
        print(f"  [exp10] {name:<38} kappa {out['problems'][name]['kappa']:.3e} -> "
              f"{out['problems'][name]['kappa_jacobi']:.3e} (x{out['problems'][name]['kappa_ratio']:.1f})")
    out["seeds"] = args.cond_seeds
    return out


def experiment_snapshot_interval(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 9: sensitivity to the snapshot interval K, and Proposition 3 across K."""
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    batch_size: int = int(cfg["training"]["batch_size"])
    seed0: int = int(cfg["seed"]) + 9500
    lr_grid: FloatArray = 10.0 ** np.linspace(-4.0, -1.0, args.si_lr_points)
    tune_sets = [ls_factory(base)(seed0 + 1000 + i, sigma) for i in range(args.si_tune_seeds)]
    eval_sets = [ls_factory(base)(seed0 + 2000 + i, sigma) for i in range(args.si_eval_seeds)]
    for obj in tune_sets + eval_sets:
        obj.prepare()
    n: int = tune_sets[0].n
    budget: int = int(args.epochs * n)
    checkpoints: FloatArray = np.geomspace(4 * n, budget, 60)
    rows: List[Dict[str, Any]] = []
    for interval in args.si_intervals:
        cfg_k: Dict[str, Any] = {**opt_cfg, "snapshot_interval": interval}
        scores: List[float] = []
        for lr in lr_grid:
            gaps = [
                optimize(o, DeterministicBatchSampler(n, batch_size, seed0 + 1000 + j),
                         reference_config(cfg_k, "svrg_adam", float(lr)), budget).final_gap
                for j, o in enumerate(tune_sets)
            ]
            scores.append(float(np.median(gaps)))
        best_lr: float = float(lr_grid[int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))])
        finals: List[float] = []
        to_target: List[float] = []
        ratios: List[float] = []
        for i, o in enumerate(eval_sets):
            sampler = DeterministicBatchSampler(n, batch_size, seed0 + 2000 + i)
            run = optimize(o, sampler, reference_config(cfg_k, "svrg_adam", best_lr), budget,
                           checkpoints=checkpoints)
            finals.append(run.final_gap)
            target: float = args.target_ratio * o.gap(np.zeros(o.d))
            hit = [e for e, g in zip(run.curve_evals, run.curve_gaps) if g <= target]
            to_target.append(hit[0] if hit else float("inf"))
            probe = probe_variance_rows(
                o, DeterministicBatchSampler(n, batch_size, seed0 + 2000 + i),
                reference_config(cfg_k, "svrg_adam", best_lr),
                steps=max(4 * interval, 256), probe_batches=args.si_probe_batches,
                batch_size=batch_size, probe_seed=seed0 + 3000 + i,
            )
            ratios += [r["var_vr"] / r["prop3_bound"] for r in probe if r["prop3_bound"] > 0]
        row = {
            "interval": interval, "best_lr": best_lr,
            "final_gap_median": float(np.median(finals)),
            "evals_to_target_median": float(np.median(to_target)),
            "reached_fraction": float(np.mean([math.isfinite(t) for t in to_target])),
            "prop3_ratio_max": float(np.max(ratios)) if ratios else float("nan"),
            "prop3_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        }
        rows.append(row)
        print(f"  [exp9] K={interval:<5} lr={best_lr:.2e} final gap {fmt(row['final_gap_median'])}  "
              f"evals to target {row['evals_to_target_median']:.0f}  max var/Prop3 bound {row['prop3_ratio_max']:.2e}")
    return {"rows": rows, "target_ratio": args.target_ratio, "epochs": args.epochs,
            "description": f"Coordinate SVRG on least squares (n={n}, d={tune_sets[0].d}, batch {batch_size}), "
                           f"lr re-tuned for every interval K"}


def _test_metrics_logistic(w: FloatArray, features: FloatArray, labels: FloatArray) -> Tuple[float, float]:
    margin: FloatArray = labels * (features @ w)
    return float(np.mean(np.logaddexp(0.0, -margin))), float(np.mean(margin > 0.0))


def experiment_heldout(cfg: Dict[str, Any], args: argparse.Namespace, prior: Dict[str, Any]) -> Dict[str, Any]:
    """Experiment 8: does optimisation precision survive to held-out data? (random split)

    Uses each method's learning rate tuned on TRAIN loss in Experiments 4 and 7, and fresh datasets.
    The split is random (interleaved), so this tests over-optimisation / generalisation, not
    temporal drift (Experiment 3 covers that).
    """
    for key in ("logistic", "nonconvex"):
        if key not in prior:
            raise SystemExit(f"Experiment 8 needs the results of Experiment {4 if key == 'logistic' else 7}; run those first")
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    batch_size: int = int(cfg["training"]["batch_size"])
    seed0: int = int(cfg["seed"]) + 9000
    out: Dict[str, Any] = {"logistic": {}, "nonconvex": {}}

    # ---- logistic ----
    lg_base = dataclasses.replace(
        RegimeConfig.from_dict(cfg), n_samples=args.lg_samples + args.ho_test, n_features=args.lg_features,
    )
    lg_lrs = {v: prior["logistic"]["regimes"]["volatile"]["variants"][v]["best_lr"] for v in VARIANTS}
    results_lg: Dict[str, Dict[str, List[float]]] = {v: {"test_logloss": [], "test_acc": [], "train_loss": []} for v in VARIANTS}
    for i in range(args.ho_seeds):
        ds = make_dataset(lg_base, seed0 + i, sigma)
        labels = np.where(ds.targets >= 0.0, 1.0, -1.0)
        perm = np.random.default_rng(seed0 + 500 + i).permutation(ds.n_samples)
        tr, te = perm[: args.lg_samples], perm[args.lg_samples:]
        train_obj = LogisticRegression(ds.features[tr], labels[tr], args.lg_l2)
        train_obj.prepare()
        for v in VARIANTS:
            run = optimize(train_obj, DeterministicBatchSampler(train_obj.n, args.lg_batch, seed0 + i),
                           reference_config({**opt_cfg}, v, lg_lrs[v]), int(args.lg_epochs * train_obj.n),
                           lr_schedule=VARIANTS[v][2])
            ll, acc = _test_metrics_logistic(run.weights, ds.features[te], labels[te])
            results_lg[v]["test_logloss"].append(ll)
            results_lg[v]["test_acc"].append(acc)
            results_lg[v]["train_loss"].append(train_obj.loss(run.weights))
    out["logistic"] = results_lg
    print("  [exp8] logistic done:", {v: round(float(np.median(r['test_logloss'])), 5) for v, r in results_lg.items()})

    # ---- non-convex network ----
    nc_base = dataclasses.replace(
        RegimeConfig.from_dict(cfg), n_features=args.nc_inputs, alpha_drift_sigma=0.0, regime_switch_prob=0.0,
        n_samples=RegimeConfig.from_dict(cfg).n_samples + args.ho_test,
    )
    nc_lrs = {v: prior["nonconvex"]["regimes"]["volatile"]["variants"][v]["best_lr"] for v in VARIANTS}
    results_nc: Dict[str, Dict[str, List[float]]] = {v: {"test_loss": [], "train_loss": []} for v in VARIANTS}
    n_train: int = RegimeConfig.from_dict(cfg).n_samples
    for i in range(args.ho_seeds):
        ds = make_dataset(nc_base, seed0 + 100 + i, sigma)
        clean = np.einsum("td,td->t", ds.features, ds.alpha_path)
        y = np.tanh(2.0 * clean) + 0.5 * ds.gaussian_noise
        perm = np.random.default_rng(seed0 + 700 + i).permutation(ds.n_samples)
        tr, te = perm[:n_train], perm[n_train:]
        train_obj = MLPRegression(ds.features[tr], y[tr], args.nc_hidden, init_seed=seed0 + 31 + i)
        test_obj = MLPRegression(ds.features[te], y[te], args.nc_hidden, init_seed=seed0 + 31 + i)
        for v in VARIANTS:
            run = optimize(train_obj, DeterministicBatchSampler(train_obj.n, batch_size, seed0 + 100 + i),
                           reference_config({**opt_cfg}, v, nc_lrs[v]), int(args.nc_epochs * train_obj.n),
                           w0=train_obj.init_weights, lr_schedule=VARIANTS[v][2])
            results_nc[v]["test_loss"].append(test_obj.loss(run.weights))
            results_nc[v]["train_loss"].append(train_obj.loss(run.weights))
    out["nonconvex"] = results_nc
    print("  [exp8] non-convex done:", {v: round(float(np.median(r['test_loss'])), 5) for v, r in results_nc.items()})
    out["settings"] = {"seeds": args.ho_seeds, "test_rows": args.ho_test}
    return out


def experiment_logistic(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 4: L2-regularised logistic regression on the sign of the return."""
    base: RegimeConfig = dataclasses.replace(
        RegimeConfig.from_dict(cfg), n_samples=args.lg_samples, n_features=args.lg_features,
    )
    out = run_ablation(
        cfg, args, logistic_factory(base, args.lg_l2), tag="exp4",
        regimes=dict(cfg["environment"]["noise_sigma_levels"]),
        epochs=args.lg_epochs, batch_size=args.lg_batch,
        tune_seeds=args.lg_tune_seeds, eval_seeds=args.lg_eval_seeds,
        seed_offset=5000, lr_points=args.lg_lr_points,
    )
    out["description"] = (f"Logistic regression on sign(return), L2={args.lg_l2:g}, n={base.n_samples}, "
                          f"d={base.n_features}, batch {out['batch_size']}, snapshot every {out['snapshot_interval']} steps")
    return out


def experiment_highdim(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Experiment 5: larger least-squares problem, with wall-clock and direct-solve timing."""
    base: RegimeConfig = dataclasses.replace(
        RegimeConfig.from_dict(cfg), n_samples=args.hd_samples, n_features=args.hd_features,
        alpha_drift_sigma=0.0, regime_switch_prob=0.0,
    )
    sigma: float = float(cfg["environment"]["noise_sigma_levels"]["volatile"])
    out = run_ablation(
        cfg, args, ls_factory(base), tag="exp5",
        regimes={"volatile": sigma},
        epochs=args.hd_epochs, batch_size=args.hd_batch,
        tune_seeds=args.hd_tune_seeds, eval_seeds=args.hd_eval_seeds,
        seed_offset=6000, lr_points=args.hd_lr_points,
        snapshot_interval=args.hd_snapshot_interval, time_direct_solve=True,
    )
    out["description"] = (f"Least squares, n={base.n_samples}, d={base.n_features}, batch {out['batch_size']}, "
                          f"snapshot every {out['snapshot_interval']} steps, wall-clock on one CPU core (NumPy)")
    return out


# --------------------------------------------------------------------------- #
# Experiment 2: variance decay along the trajectory
# --------------------------------------------------------------------------- #
def probe_variance_rows(
    obj: LeastSquares,
    sampler: DeterministicBatchSampler,
    ref_cfg: ReferenceConfig,
    steps: int,
    probe_batches: int,
    batch_size: int,
    probe_seed: int,
) -> List[Dict[str, float]]:
    """Run reference SVRG and probe Var(g_hat), Var(g_sgd) at maximum snapshot staleness."""
    opt: ReferenceSVRG = ReferenceSVRG(ref_cfg, obj.d)
    probe_rng = np.random.default_rng(probe_seed)
    lbar2: float = obj.smoothness_second_moment()
    interval: int = ref_cfg.snapshot_interval
    w: FloatArray = np.zeros(obj.d)
    rows_out: List[Dict[str, float]] = []
    for step in range(steps):
        if opt.needs_snapshot:
            opt.refresh_snapshot(w, obj.full_grad(w))
        if opt.steps_since_snapshot == interval - 1:
            true_grad: FloatArray = obj.full_grad(w)
            delta_sq: float = float(np.sum((w - opt.snapshot) ** 2))  # type: ignore[operator]
            vr_sum = sgd_sum = 0.0
            for _ in range(probe_batches):
                rows = np.sort(probe_rng.choice(obj.n, size=batch_size, replace=False))
                g_live = obj.grad(w, rows)
                g_snap = obj.grad(opt.snapshot, rows)  # type: ignore[arg-type]
                g_hat = opt.variance_reduced_gradient(w, g_live, g_snap)
                vr_sum += float(np.sum((g_hat - true_grad) ** 2))
                sgd_sum += float(np.sum((g_live - true_grad) ** 2))
            rows_out.append({
                "step": float(step),
                "var_vr": vr_sum / probe_batches,
                "var_sgd": sgd_sum / probe_batches,
                "lemma2_bound": lbar2 * delta_sq / batch_size,
                "prop3_bound": lbar2 * obj.d * (ref_cfg.lr * ref_cfg.update_clip * interval) ** 2 / batch_size,
                "snapshot_distance": math.sqrt(delta_sq),
                "gap": obj.gap(w),
            })
        rows = sampler.batch(step)
        w = opt.step(w, obj.grad(w, rows), obj.grad(opt.snapshot, rows))  # type: ignore[arg-type]
    return rows_out


def experiment_variance(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    batch_size: int = int(cfg["training"]["batch_size"])
    seed0: int = int(cfg["seed"])
    ds = generate_market(dataclasses.replace(base, seed=seed0 + 3000))
    obj: LeastSquares = LeastSquares(ds.features, ds.targets)
    sampler = DeterministicBatchSampler(obj.n, batch_size, seed0 + 3000)
    lr: float = float(opt_cfg["lr"])
    ref_cfg: ReferenceConfig = reference_config(opt_cfg, "svrg_adam", lr)
    rows_out = probe_variance_rows(obj, sampler, ref_cfg, args.variance_steps, args.probe_batches,
                                   batch_size, seed0 + 3001)
    within_lemma: float = float(np.mean([r["var_vr"] <= r["lemma2_bound"] for r in rows_out]))
    within_prop: float = float(np.mean([r["var_vr"] <= r["prop3_bound"] for r in rows_out]))
    print(f"  [exp2] probes={len(rows_out)}  within Lemma 2: {within_lemma:.0%}  within Prop 3: {within_prop:.0%}")
    return {
        "rows": rows_out,
        "fraction_within_lemma2": within_lemma,
        "fraction_within_prop3": within_prop,
        "lr": lr,
        "snapshot_interval": ref_cfg.snapshot_interval,
    }


# --------------------------------------------------------------------------- #
# Experiment 3: walk-forward alpha tracking
# --------------------------------------------------------------------------- #
def _pearson(a: FloatArray, b: FloatArray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def walk_forward_once(
    dataset: MarketDataset,
    opt_cfg: Dict[str, Any],
    window: int,
    horizon: int,
    batch_size: int,
    budget_per_window: int,
    lr: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    x: FloatArray = dataset.features
    y: FloatArray = dataset.targets
    clean: FloatArray = np.einsum("td,td->t", x, dataset.alpha_path)
    n: int = dataset.n_samples
    methods: Dict[str, Optional[ReferenceConfig]] = {
        "ols": None,
        "adam": reference_config(opt_cfg, "adam", lr),
        "svrg_adam": reference_config(opt_cfg, "svrg_adam", lr),
    }
    weights: Dict[str, FloatArray] = {m: np.zeros(dataset.n_features) for m in methods}
    preds: Dict[str, List[FloatArray]] = {m: [] for m in methods}
    test_y: List[FloatArray] = []
    test_clean: List[FloatArray] = []

    for start in range(window, n - horizon + 1, horizon):
        tr = slice(start - window, start)
        te = slice(start, start + horizon)
        obj: LeastSquares = LeastSquares(x[tr], y[tr])
        sampler = DeterministicBatchSampler(window, batch_size, seed + start)
        for name, cfg_m in methods.items():
            if cfg_m is None:
                weights[name] = obj.optimum
            else:
                weights[name] = optimize(
                    obj, sampler, cfg_m, budget_per_window, w0=weights[name], track_gap=False
                ).weights
            preds[name].append(x[te] @ weights[name])
        test_y.append(y[te])
        test_clean.append(clean[te])

    y_oos: FloatArray = np.concatenate(test_y)
    clean_oos: FloatArray = np.concatenate(test_clean)
    out: Dict[str, Dict[str, float]] = {}
    for name in methods:
        p: FloatArray = np.concatenate(preds[name])
        pnl: FloatArray = np.sign(p) * y_oos
        out[name] = {
            "ic_clean": _pearson(p, clean_oos),
            "ic_target": _pearson(p, y_oos),
            "hit_rate": float(np.mean(np.sign(p) == np.sign(clean_oos))),
            "sharpe_per_tick": float(np.mean(pnl) / (np.std(pnl) + 1e-12)),
        }
    return out


def experiment_walk_forward(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    # A learnable non-stationary alpha: slower drift and rarer regime switches
    # than the optimiser-stress default, longer history for many windows.
    wf_base: RegimeConfig = dataclasses.replace(
        base, n_samples=args.wf_samples, alpha_drift_sigma=0.005, regime_switch_prob=5e-4,
    )
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    batch_size: int = int(cfg["training"]["batch_size"])
    seed0: int = int(cfg["seed"])
    datasets: List[MarketDataset] = [
        generate_market(dataclasses.replace(wf_base, seed=seed0 + 4000 + i))
        for i in range(args.wf_seeds)
    ]
    by_budget: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for budget in args.wf_budgets:
        per_seed: List[Dict[str, Dict[str, float]]] = []
        for i, ds in enumerate(datasets):
            per_seed.append(walk_forward_once(
                ds, opt_cfg, window=args.wf_window, horizon=args.wf_horizon,
                batch_size=batch_size, budget_per_window=budget,
                lr=float(opt_cfg["lr"]), seed=seed0 + 4000 + i,
            ))
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method in per_seed[0]:
            summary[method] = {}
            for metric in per_seed[0][method]:
                q25, q50, q75 = quartiles([s[method][metric] for s in per_seed])
                summary[method][metric] = {
                    "q25": q25, "median": q50, "q75": q75,
                    "values": [s[method][metric] for s in per_seed],
                }
        by_budget[str(budget)] = summary
        line: str = "  ".join(f"{m}={summary[m]['ic_clean']['median']:.3f}" for m in summary)
        print(f"  [exp3] budget {budget:>6}: IC(clean) median  {line}")
    return {
        "by_budget": by_budget,
        "settings": {
            "window": args.wf_window, "horizon": args.wf_horizon,
            "budgets": list(args.wf_budgets), "seeds": args.wf_seeds,
            "samples": args.wf_samples, "lr": float(opt_cfg["lr"]),
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def paired_bootstrap(diffs: Sequence[float], draws: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Median paired difference with a 95% bootstrap interval (resampling seeds)."""
    arr: FloatArray = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    meds: FloatArray = np.median(rng.choice(arr, size=(draws, len(arr)), replace=True), axis=1)
    return float(np.median(arr)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def significance_lines(results: Dict[str, Any]) -> List[str]:
    """Paired comparisons on held-out seeds, in decades of loss gap (log10)."""
    lines: List[str] = [
        "### Paired comparisons on held-out seeds",
        "",
        "Difference in `log10(final gap)`: **negative means the first method is better**. Median over seeds with a 95% "
        "paired bootstrap interval (resampling seeds); 'wins' counts seeds where the first method had the smaller gap. "
        "Baselines are the four non-SVRG methods; 'best baseline' is the one with the lowest median gap in that cell. "
        "Gaps at float64 round-off are floored at 1e-40.",
        "",
        "| Experiment / regime | Comparison | median diff (decades) | 95% CI | wins |",
        "|---|---|---|---|---|",
    ]
    names = {"ablation": "Exp 1 least squares", "logistic": "Exp 4 logistic", "highdim": "Exp 5 larger LS",
             "hetero": "Exp 6 bad scaling", "nonconvex": "Exp 7 non-convex"}
    baselines = ["sgd_momentum", "sgd_momentum_cosine", "adam", "adam_cosine"]
    for key, label in names.items():
        if key not in results:
            continue
        for regime, block in results[key]["regimes"].items():
            v = block["variants"]
            if "final_gaps" not in v["svrg_adam"]:
                continue
            best_base = min(baselines, key=lambda b: v[b]["final_gap_median"])
            logs = {m: np.log10(np.maximum(np.asarray(v[m]["final_gaps"], dtype=float), 1e-40)) for m in v}
            pairs = [("svrg_adam_cosine", best_base), ("svrg_adam", best_base), ("svrg_momentum", best_base),
                     ("svrg_adam", "svrg_momentum")]
            for a_name, b_name in pairs:
                if a_name not in logs or b_name not in logs:
                    continue
                diff = logs[a_name] - logs[b_name]
                med, lo, hi = paired_bootstrap(diff)
                wins = int(np.sum(diff < 0))
                lines.append(
                    f"| {label} / {regime} | {VARIANT_LABELS[a_name]} vs {VARIANT_LABELS[b_name]} | "
                    f"{med:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {wins}/{len(diff)} |"
                )
    if "jacobi" in results:
        for pname, pr in results["jacobi"]["problems"].items():
            stored = results.get(pr["stored_key"])
            if stored is None:
                continue
            v = stored["regimes"]["volatile"]["variants"]
            best_base = min(baselines, key=lambda b: v[b]["final_gap_median"])
            for mname, mkey in (("Jacobi SVRG (theory)", "theory"), ("Jacobi SVRG (tuned)", "tuned")):
                mine = np.log10(np.maximum(np.asarray(pr[mkey]["final_gaps"], dtype=float), 1e-40))
                for other in (best_base, "svrg_adam_cosine", "svrg_adam"):
                    theirs = np.log10(np.maximum(np.asarray(v[other]["final_gaps"], dtype=float), 1e-40))
                    if len(theirs) != len(mine):
                        continue
                    diff = mine - theirs
                    med, lo, hi = paired_bootstrap(diff)
                    lines.append(
                        f"| Exp 11 {pname} | {mname} vs {VARIANT_LABELS[other]} | {med:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                        f"{int(np.sum(diff < 0))}/{len(diff)} |")
    if "walk_forward" in results:
        lines += [
            "",
            "Walk-forward (Experiment 3): paired difference in out-of-sample IC versus OLS, median with 95% bootstrap interval over seeds. "
            "An interval containing 0 means no detectable difference.",
            "",
            "| Budget / window | Method vs OLS | median IC diff | 95% CI |",
            "|---|---|---|---|",
        ]
        for budget, summ in results["walk_forward"]["by_budget"].items():
            if "values" not in summ["ols"]["ic_clean"]:
                continue
            for method, label in (("adam", "Adam"), ("svrg_adam", "Coordinate SVRG")):
                diff = np.asarray(summ[method]["ic_clean"]["values"]) - np.asarray(summ["ols"]["ic_clean"]["values"])
                med, lo, hi = paired_bootstrap(diff)
                lines.append(f"| {int(budget):,} | {label} | {med:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines.append("")
    return lines


def write_tables(results: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    ablation_blocks = (
        ("ablation", "Experiment 1: tuned ablation on least squares"),
        ("logistic", "Experiment 4: tuned ablation on logistic regression (sign of the return)"),
        ("highdim", "Experiment 5: larger problem with wall-clock timing"),
        ("hetero", "Experiment 6: badly scaled features (adaptive scaling's home turf)"),
        ("nonconvex", "Experiment 7: non-convex network (no guarantees apply)"),
    )
    for key, title in ablation_blocks:
        if key not in results:
            continue
        ab = results[key]
        timed: bool = key == "highdim"
        lines += [
            f"### {title}",
            "",
            f"{ab.get('description', '')}. Budget: {ab['budget_epochs']} epochs of sample-gradient evaluations. "
            "Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; "
            f"'evals to target' = sample-gradient evaluations to reach {ab['target_ratio']:g} x the initial gap.",
            "",
        ]
        for regime, block in ab["regimes"].items():
            header = "| Method | tuned lr | final gap, median [IQR] | evals to target |"
            rule = "|---|---|---|---|"
            if timed:
                header += " seconds to target |"
                rule += "---|"
            if ab.get("report_loss"):
                header += " final train loss |"
                rule += "---|"
            lines += [f"**{regime}** (noise sigma = {block['sigma']})", "", header, rule]
            for variant, r in block["variants"].items():
                target = r["evals_to_target_median"]
                target_txt = "not reached" if math.isinf(target) else f"{target:,.0f}"
                edge = " †" if r.get("best_lr_at_grid_edge") else ""
                row = (
                    f"| {VARIANT_LABELS[variant]} | {r['best_lr']:.2e}{edge} | "
                    f"{fmt(r['final_gap_median'])} [{fmt(r['final_gap_q25'])}, {fmt(r['final_gap_q75'])}] | {target_txt} |"
                )
                if timed:
                    secs = r["seconds_to_target_median"]
                    row += " " + ("not reached" if math.isinf(secs) else f"{secs:.2f}") + " |"
                if ab.get("report_loss"):
                    row += f" {r['final_loss_median']:.4f} |"
                lines.append(row)
            lines.append("")
            if "direct_solve_seconds" in block:
                lines += [
                    f"Direct solves on the same data (one CPU core): `lstsq` {block['direct_solve_seconds']:.2f} s, "
                    f"normal equations {block['normal_equations_seconds']:.2f} s. "
                    "The iterative times above exclude the cost of the learning-rate sweep.",
                    "",
                ]
        lines += ["† best learning rate sat on the edge of the sweep grid (the true optimum may lie outside it).", ""]
    if "variance" in results:
        v = results["variance"]
        lines += [
            "### Experiment 2: variance along the trajectory",
            "",
            f"Probes taken at maximum snapshot staleness (every {v['snapshot_interval']} steps). "
            f"Measured variance fell within the Lemma 2 bound in {v['fraction_within_lemma2']:.0%} of probes "
            f"and within the Proposition 3 bound in {v['fraction_within_prop3']:.0%}.",
            "",
            "| step | Var(g_hat) | Var(g_sgd) | ratio sgd/vr | snapshot distance | gap |",
            "|---|---|---|---|---|---|",
        ]
        rows = v["rows"]
        picks = sorted(set([0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1]))
        for i in picks:
            r = rows[i]
            ratio = r["var_sgd"] / max(r["var_vr"], 1e-300)
            lines.append(
                f"| {int(r['step'])} | {fmt(r['var_vr'])} | {fmt(r['var_sgd'])} | {ratio:,.1f}x | "
                f"{fmt(r['snapshot_distance'])} | {fmt(r['gap'])} |"
            )
        lines.append("")
    if "walk_forward" in results:
        w = results["walk_forward"]
        st = w["settings"]
        lines += [
            "### Experiment 3: walk-forward alpha tracking under a compute budget",
            "",
            f"{st['samples']} ticks, window {st['window']}, re-fit every {st['horizon']} ticks, "
            f"{st['seeds']} seeds, iterative methods warm-started at lr {st['lr']:g} (untuned). "
            "Each row gives the sample-gradient evaluations allowed per re-fit. "
            "IC = out-of-sample Pearson correlation of the prediction with the *clean* hidden signal "
            "(observable only in simulation); median [IQR] over seeds. OLS is the closed-form reference "
            "and ignores the budget.",
            "",
            "| Budget / window | OLS | Adam | Coordinate SVRG |",
            "|---|---|---|---|",
        ]
        for budget, summ in w["by_budget"].items():
            def cell(method: str) -> str:
                m = summ[method]["ic_clean"]
                return f"{m['median']:.3f} [{m['q25']:.3f}, {m['q75']:.3f}]"
            lines.append(f"| {int(budget):,} | {cell('ols')} | {cell('adam')} | {cell('svrg_adam')} |")
        lines.append("")
        last = list(w["by_budget"].values())[-1]
        lines += [
            f"Secondary metrics at the largest budget ({list(w['by_budget'].keys())[-1]}):",
            "",
            "| Method | IC vs realised target | sign hit-rate | per-tick Sharpe |",
            "|---|---|---|---|",
        ]
        for method, label in (("ols", "OLS"), ("adam", "Adam"), ("svrg_adam", "Coordinate SVRG")):
            m = last[method]
            def c2(k: str) -> str:
                return f"{m[k]['median']:.3f} [{m[k]['q25']:.3f}, {m[k]['q75']:.3f}]"
            lines.append(f"| {label} | {c2('ic_target')} | {c2('hit_rate')} | {c2('sharpe_per_tick')} |")
        lines.append("")
    if "heldout" in results:
        h = results["heldout"]
        lines += [
            "### Experiment 8: held-out evaluation (random split)",
            "",
            f"{h['settings']['seeds']} fresh datasets per task, {h['settings']['test_rows']} held-out rows each, "
            "learning rates tuned on training loss in Experiments 4 and 7. Median [IQR] over seeds. The split is random, "
            "so this measures over-optimisation, not temporal drift.",
            "",
            "**Logistic regression (volatile)**: test log-loss (lower is better), test accuracy.",
            "",
            "| Method | test log-loss | test accuracy | train loss |",
            "|---|---|---|---|",
        ]
        for v, r in h["logistic"].items():
            def c(k: str, prec: int = 5) -> str:
                q25, q50, q75 = quartiles(r[k])
                return f"{q50:.{prec}f} [{q25:.{prec}f}, {q75:.{prec}f}]"
            lines.append(f"| {VARIANT_LABELS[v]} | {c('test_logloss')} | {c('test_acc', 4)} | {c('train_loss')} |")
        base_name = min(["sgd_momentum", "sgd_momentum_cosine", "adam", "adam_cosine"],
                        key=lambda b: float(np.median(h["logistic"][b]["test_logloss"])))
        lines += ["", "Paired test-log-loss difference versus the best baseline by median "
                  f"({VARIANT_LABELS[base_name]}; chosen on these same seeds, which is conservative for SVRG). "
                  "Negative favours the first method.", "",
                  "| Method | median diff | 95% CI |", "|---|---|---|"]
        for name in ("svrg_momentum", "svrg_adam", "svrg_adam_cosine"):
            d = np.asarray(h["logistic"][name]["test_logloss"]) - np.asarray(h["logistic"][base_name]["test_logloss"])
            med, lo, hi = paired_bootstrap(d)
            lines.append(f"| {VARIANT_LABELS[name]} | {med:+.2e} | [{lo:+.2e}, {hi:+.2e}] |")
        lines += ["", "**Non-convex network (volatile)**: test loss (0.5 x MSE, lower is better; the irreducible "
                  "noise floor is 0.125), train loss.", "",
                  "| Method | test loss | train loss |", "|---|---|---|"]
        for v, r in h["nonconvex"].items():
            def c2(k: str) -> str:
                q25, q50, q75 = quartiles(r[k])
                return f"{q50:.4f} [{q25:.4f}, {q75:.4f}]"
            lines.append(f"| {VARIANT_LABELS[v]} | {c2('test_loss')} | {c2('train_loss')} |")
        base_name = min(["sgd_momentum", "sgd_momentum_cosine", "adam", "adam_cosine"],
                        key=lambda b: float(np.median(h["nonconvex"][b]["test_loss"])))
        lines += ["", "Paired test-loss difference versus the best baseline by median "
                  f"({VARIANT_LABELS[base_name]}). Negative favours the first method.", "",
                  "| Method | median diff | 95% CI |", "|---|---|---|"]
        for name in ("svrg_momentum", "svrg_adam", "svrg_adam_cosine"):
            d = np.asarray(h["nonconvex"][name]["test_loss"]) - np.asarray(h["nonconvex"][base_name]["test_loss"])
            med, lo, hi = paired_bootstrap(d)
            lines.append(f"| {VARIANT_LABELS[name]} | {med:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        lines.append("")
    if "jacobi" in results:
        jc = results["jacobi"]
        lines += [
            "### Experiment 11: Jacobi-preconditioned SVRG with theory-prescribed hyper-parameters",
            "",
            f"{jc['description']}. **Theory** = every hyper-parameter from the Johnson-Zhang recipe "
            "(`eta = 0.1 / L_D`, `m = 50 L_D / gamma_D` single-sample steps, random-iterate snapshots): zero tuning. "
            "**Tuned** = mini-batches, last-iterate snapshots, learning rate swept like every other method. "
            "The theorem guarantees `E[gap_s] <= alpha * gap_{s-1}` per epoch; 'observed' is the measured mean "
            "epoch-to-epoch ratio (epochs with gap above 1e-20 only).",
            "",
            "| Problem | Method | median final gap | evals to target | kappa_D | predicted alpha | observed mean ratio (max) |",
            "|---|---|---|---|---|---|---|",
        ]
        for pname, pr in jc["problems"].items():
            th, tu = pr["theory"], pr["tuned"]
            e_th, e_tu = th["evals_to_target_median"], tu["evals_to_target_median"]
            lines.append(
                f"| {pname} | Jacobi SVRG, theory-prescribed (no tuning) | {fmt(th['final_gap_median'])} | "
                f"{'not reached' if math.isinf(e_th) else f'{e_th:,.0f}'} | {th['kappa_D']:.0f} | {th['alpha_predicted']:.2f} | "
                f"{th['mean_epoch_contraction']:.3f} ({th['max_epoch_contraction']:.3f}) |")
            lines.append(
                f"| {pname} | Jacobi SVRG, tuned lr {tu['best_lr']:.2e}{' †' if tu['best_lr_at_grid_edge'] else ''} | "
                f"{fmt(tu['final_gap_median'])} | {'not reached' if math.isinf(e_tu) else f'{e_tu:,.0f}'} | | | |")
        lines.append("")
    if "conditioning" in results:
        cd = results["conditioning"]
        lines += [
            "### Experiment 10: predicted benefit of diagonal preconditioning (Proposition 5)",
            "",
            f"Median over {cd['seeds']} datasets. `kappa = L / gamma` with `L = max_i ||x_i||^2` and `gamma = lambda_min(X^T X / n)`; "
            "the Jacobi-preconditioned constants use `D = diag(X^T X / n)`. Proposition 5 gives SVRG's linear rate in terms of this `kappa`.",
            "",
            "| Problem | kappa (raw) | kappa (Jacobi) | predicted reduction |",
            "|---|---|---|---|",
        ]
        for name, c in cd["problems"].items():
            lines.append(f"| {name} | {c['kappa']:.3e} | {c['kappa_jacobi']:.3e} | {c['kappa_ratio']:.1f}x |")
        lines.append("")
    if "snapshot_interval" in results:
        si = results["snapshot_interval"]
        lines += [
            "### Experiment 9: snapshot interval K",
            "",
            f"{si['description']}. 'Prop 3 ratio' = measured gradient variance at worst-case staleness divided by the "
            "Proposition 3 bound; the bound holds when it is at most 1.",
            "",
            "| K | tuned lr | lr x K | final gap, median | evals to target | seeds reaching target | max Prop 3 ratio | median Prop 3 ratio |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in si["rows"]:
            e = r["evals_to_target_median"]
            lines.append(
                f"| {r['interval']} | {r['best_lr']:.2e} | {r['best_lr'] * r['interval']:.2f} | {fmt(r['final_gap_median'])} | "
                f"{'not reached' if math.isinf(e) else f'{e:,.0f}'} | {r['reached_fraction']:.0%} | "
                f"{r['prop3_ratio_max']:.2e} | {r['prop3_ratio_median']:.2e} |"
            )
        lines.append("")
    lines += significance_lines(results)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(results: Dict[str, Any], out_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    colors: Dict[str, str] = {
        "sgd_momentum": "#8c8c8c", "sgd_momentum_cosine": "#5a5a5a",
        "adam": "#6baed6", "adam_cosine": "#1f77b4",
        "svrg_momentum": "#ff7f0e", "svrg_adam": "#d62728", "svrg_adam_cosine": "#7f0000",
    }
    for key, fname, title in (
        ("ablation", "ablation_curves.png", "Least squares: tuned ablation, median and IQR over held-out seeds"),
        ("logistic", "logistic_curves.png", "Logistic regression on sign(return): tuned ablation"),
        ("highdim", "highdim_curves.png", "Larger least-squares problem"),
        ("hetero", "hetero_curves.png", "Least squares with badly scaled features"),
        ("nonconvex", "nonconvex_curves.png", "Non-convex network: squared full-gradient norm"),
    ):
        if key not in results:
            continue
        regimes = results[key]["regimes"]
        wall: bool = key == "highdim"
        ncols: int = len(regimes) * (2 if wall else 1)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
        axes = np.atleast_1d(axes)
        axis_iter = iter(axes)
        for regime, block in regimes.items():
            for use_seconds in ((False, True) if wall else (False,)):
                ax = next(axis_iter)
                for variant, r in block["variants"].items():
                    xs = np.asarray(r["curve_seconds_median"] if use_seconds else r["curve_evals"])
                    med = np.maximum(np.asarray(r["curve_median"], dtype=float), 1e-18)
                    lo = np.maximum(np.asarray(r["curve_q25"], dtype=float), 1e-18)
                    hi = np.maximum(np.asarray(r["curve_q75"], dtype=float), 1e-18)
                    ax.plot(xs, med, label=VARIANT_LABELS[variant], color=colors[variant], lw=2,
                            ls="--" if "cosine" in variant else "-")
                    ax.fill_between(xs, lo, hi, color=colors[variant], alpha=0.15)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"{regime} (sigma={block['sigma']})")
                ax.set_xlabel("wall-clock seconds (one CPU core)" if use_seconds else "sample-gradient evaluations")
                ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("squared full-gradient norm" if key == "nonconvex" else "loss gap  f(w) - f(w*)")
        axes[0].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))
    if "variance" in results:
        rows = results["variance"]["rows"]
        steps = [r["step"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.semilogy(steps, [r["var_sgd"] for r in rows], label="mini-batch SGD  Var(g)", color="#8c8c8c", lw=2)
        ax.semilogy(steps, [r["var_vr"] for r in rows], label="SVRG  Var(g_hat)", color="#d62728", lw=2)
        ax.semilogy(steps, [r["lemma2_bound"] for r in rows], "--", label="Lemma 2 bound", color="#2ca02c")
        ax.set_xlabel("optimizer step"); ax.set_ylabel("E||g - grad f(w_t)||^2")
        ax.set_title("Variance Reduction Law along a real trajectory")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "variance_decay.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))
    if "walk_forward" in results:
        by_budget = results["walk_forward"]["by_budget"]
        budgets = [int(b) for b in by_budget]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for method, label, color in (("ols", "OLS (closed form)", "#8c8c8c"),
                                     ("adam", "Adam, warm start", "#1f77b4"),
                                     ("svrg_adam", "Coordinate SVRG, warm start", "#d62728")):
            med = np.array([by_budget[str(b)][method]["ic_clean"]["median"] for b in budgets])
            lo = np.array([by_budget[str(b)][method]["ic_clean"]["q25"] for b in budgets])
            hi = np.array([by_budget[str(b)][method]["ic_clean"]["q75"] for b in budgets])
            ax.plot(budgets, med, marker="o", label=label, color=color, lw=2)
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("sample-gradient evaluations per re-fit")
        ax.set_ylabel("out-of-sample IC vs clean signal")
        ax.set_title("Walk-forward alpha tracking under a compute budget")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "walk_forward_ic.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))
    if "snapshot_interval" in results:
        rows = results["snapshot_interval"]["rows"]
        ks = [r["interval"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ev = [r["evals_to_target_median"] if math.isfinite(r["evals_to_target_median"]) else np.nan for r in rows]
        axes[0].plot(ks, ev, marker="o", color="#d62728", lw=2)
        axes[0].set_xscale("log", base=2)
        if np.any(np.isfinite(ev)):
            axes[0].set_yscale("log")
        axes[0].set_xlabel("snapshot interval K (steps)"); axes[0].set_ylabel("evals to 1e-8 x initial gap")
        axes[0].set_title("Sensitivity to the snapshot interval"); axes[0].grid(alpha=0.3, which="both")
        axes[1].plot(ks, [r["prop3_ratio_max"] for r in rows], marker="o", label="max over probes", color="#2ca02c", lw=2)
        axes[1].plot(ks, [r["prop3_ratio_median"] for r in rows], marker="s", label="median", color="#1f77b4", lw=2)
        axes[1].axhline(1.0, ls="--", color="black", label="bound = 1")
        axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
        axes[1].set_xlabel("snapshot interval K (steps)"); axes[1].set_ylabel("measured variance / Proposition 3 bound")
        axes[1].set_title("Proposition 3 holds across K"); axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "snapshot_interval.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))
    if "jacobi" in results:
        probs = results["jacobi"]["problems"]
        fig, axes = plt.subplots(1, len(probs), figsize=(5.5 * len(probs), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (pname, pr) in zip(axes, probs.items()):
            xs = np.asarray(pr["curve_evals"])
            for label, key, color in (("Jacobi SVRG, theory-prescribed (no tuning)", "theory", "#2ca02c"),
                                      ("Jacobi SVRG, tuned", "tuned", "#006400")):
                med = np.maximum(np.asarray(pr[key]["curve_median"], dtype=float), 1e-18)
                lo = np.maximum(np.asarray(pr[key]["curve_q25"], dtype=float), 1e-18)
                hi = np.maximum(np.asarray(pr[key]["curve_q75"], dtype=float), 1e-18)
                ax.plot(xs, med, label=label, color=color, lw=2)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
            stored = results.get(pr["stored_key"])
            if stored is not None:
                for variant in ("adam_cosine", "svrg_adam_cosine"):
                    r = stored["regimes"]["volatile"]["variants"][variant]
                    ax.plot(r["curve_evals"], np.maximum(np.asarray(r["curve_median"], dtype=float), 1e-18),
                            ls="--", lw=1.5, color=colors[variant], label=VARIANT_LABELS[variant])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(pname); ax.set_xlabel("sample-gradient evaluations"); ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("loss gap  f(w) - f(w*)"); axes[0].legend(fontsize=7)
        fig.suptitle("Jacobi-preconditioned SVRG: prescribed by theory vs tuned baselines")
        fig.tight_layout()
        p = out_dir / "jacobi_curves.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))
    return written


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stochastic_regime.yaml"))
    parser.add_argument("--only", nargs="+", type=int, choices=list(range(1, 12)), default=list(range(1, 12)))
    parser.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    parser.add_argument("--tables-only", action="store_true",
                        help="regenerate tables.md and figures from the stored results.json, run nothing")
    parser.add_argument("--merge", action="store_true",
                        help="update results/experiments.json in place instead of overwriting it")
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "docs" / "figures"))
    # experiment 1
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-points", type=int, default=13)
    parser.add_argument("--lr-min-exp", type=float, default=-5.0)
    parser.add_argument("--lr-max-exp", type=float, default=-0.5)
    parser.add_argument("--tune-seeds", type=int, default=2)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--curve-points", type=int, default=40)
    parser.add_argument("--target-ratio", type=float, default=1e-8)
    # experiment 4 (logistic)
    parser.add_argument("--lg-samples", type=int, default=20000)
    parser.add_argument("--lg-features", type=int, default=64)
    parser.add_argument("--lg-l2", type=float, default=1e-2)
    parser.add_argument("--lg-epochs", type=int, default=40)
    parser.add_argument("--lg-batch", type=int, default=64)
    parser.add_argument("--lg-lr-points", type=int, default=9)
    parser.add_argument("--lg-tune-seeds", type=int, default=1)
    parser.add_argument("--lg-eval-seeds", type=int, default=12)
    # experiment 8 (held-out)
    parser.add_argument("--ho-seeds", type=int, default=8)
    parser.add_argument("--ho-test", type=int, default=5000)
    # experiment 10 (conditioning)
    parser.add_argument("--cond-seeds", type=int, default=5)
    # experiment 9 (snapshot interval)
    parser.add_argument("--si-intervals", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--si-lr-points", type=int, default=7)
    parser.add_argument("--si-tune-seeds", type=int, default=2)
    parser.add_argument("--si-eval-seeds", type=int, default=6)
    parser.add_argument("--si-probe-batches", type=int, default=32)
    # experiment 7 (non-convex)
    parser.add_argument("--nc-inputs", type=int, default=16)
    parser.add_argument("--nc-hidden", type=int, default=16)
    parser.add_argument("--nc-epochs", type=int, default=100)
    parser.add_argument("--nc-lr-points", type=int, default=9)
    parser.add_argument("--nc-tune-seeds", type=int, default=2)
    parser.add_argument("--nc-eval-seeds", type=int, default=8)
    # experiment 6 (heterogeneous feature scales)
    parser.add_argument("--het-decades", type=float, default=3.0)
    # experiment 5 (larger least squares)
    parser.add_argument("--hd-samples", type=int, default=40000)
    parser.add_argument("--hd-features", type=int, default=200)
    parser.add_argument("--hd-epochs", type=int, default=30)
    parser.add_argument("--hd-batch", type=int, default=256)
    parser.add_argument("--hd-snapshot-interval", type=int, default=128)
    parser.add_argument("--hd-lr-points", type=int, default=9)
    parser.add_argument("--hd-tune-seeds", type=int, default=1)
    parser.add_argument("--hd-eval-seeds", type=int, default=3)
    # experiment 2
    parser.add_argument("--variance-steps", type=int, default=1600)
    parser.add_argument("--probe-batches", type=int, default=64)
    # experiment 3
    parser.add_argument("--wf-samples", type=int, default=8192)
    parser.add_argument("--wf-window", type=int, default=512)
    parser.add_argument("--wf-horizon", type=int, default=64)
    parser.add_argument("--wf-budgets", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--wf-seeds", type=int, default=20)
    args = parser.parse_args()
    if args.quick:
        args.epochs, args.lr_points, args.tune_seeds, args.eval_seeds = 10, 5, 1, 2
        args.curve_points, args.variance_steps, args.probe_batches = 12, 320, 16
        args.wf_samples, args.wf_seeds, args.wf_budgets = 2048, 2, [2048, 8192]
        args.lg_samples, args.lg_features, args.lg_epochs, args.lg_lr_points = 2000, 8, 8, 4
        args.lg_eval_seeds = 2
        args.hd_samples, args.hd_features, args.hd_epochs, args.hd_lr_points = 4000, 20, 8, 4
        args.hd_eval_seeds = 2
        args.nc_epochs, args.nc_lr_points, args.nc_tune_seeds, args.nc_eval_seeds = 8, 4, 1, 2
        args.ho_seeds, args.ho_test = 2, 1000
        args.si_intervals, args.si_lr_points, args.si_tune_seeds, args.si_eval_seeds = [8, 32, 128], 3, 1, 2
        args.si_probe_batches = 8
        args.cond_seeds = 2

    cfg: Dict[str, Any] = load_config(args.config)
    if args.tables_only:
        stored: Dict[str, Any] = json.loads((Path(args.out) / "experiments.json").read_text(encoding="utf-8"))
        write_tables(stored, Path(args.out) / "tables.md")
        print(f"regenerated tables and {len(make_figures(stored, Path(args.figures)))} figure(s) from stored results")
        return
    results: Dict[str, Any] = {}
    started: float = time.time()
    if 1 in args.only:
        print("Experiment 1: tuned 2x2 ablation")
        results["ablation"] = experiment_ablation(cfg, args)
    if 2 in args.only:
        print("Experiment 2: variance decay")
        results["variance"] = experiment_variance(cfg, args)
    if 3 in args.only:
        print("Experiment 3: walk-forward alpha tracking")
        results["walk_forward"] = experiment_walk_forward(cfg, args)
    if 4 in args.only:
        print("Experiment 4: logistic regression on the sign of the return")
        results["logistic"] = experiment_logistic(cfg, args)
    if 5 in args.only:
        print("Experiment 5: larger problem with wall-clock timing")
        results["highdim"] = experiment_highdim(cfg, args)
    if 6 in args.only:
        print("Experiment 6: badly scaled features")
        results["hetero"] = experiment_hetero(cfg, args)
    if 7 in args.only:
        print("Experiment 7: non-convex network")
        results["nonconvex"] = experiment_nonconvex(cfg, args)
    prior: Dict[str, Any] = dict(results)
    existing: Path = Path(args.out) / "experiments.json"
    if existing.exists():
        for key, value in json.loads(existing.read_text(encoding="utf-8")).items():
            prior.setdefault(key, value)
    if 8 in args.only:
        print("Experiment 8: held-out evaluation")
        results["heldout"] = experiment_heldout(cfg, args, prior)
    if 9 in args.only:
        print("Experiment 9: snapshot interval")
        results["snapshot_interval"] = experiment_snapshot_interval(cfg, args)
    if 10 in args.only:
        print("Experiment 10: conditioning")
        results["conditioning"] = experiment_conditioning(cfg, args)
    if 11 in args.only:
        print("Experiment 11: Jacobi-preconditioned SVRG")
        results["jacobi"] = experiment_jacobi(cfg, args)

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path: Path = out_dir / "experiments.json"
    if args.merge and results_path.exists():
        merged: Dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
        merged.update(results)
        results = merged
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_tables(results, out_dir / "tables.md")
    figures: List[str] = make_figures(results, Path(args.figures))
    print(f"done in {time.time() - started:.0f}s -> {out_dir}/experiments.json, tables.md, {len(figures)} figure(s)")


if __name__ == "__main__":
    main()
