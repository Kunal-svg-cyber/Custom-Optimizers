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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
}
VARIANT_LABELS: Dict[str, str] = {
    "sgd_momentum": "SGD + momentum",
    "sgd_momentum_cosine": "SGD + momentum, cosine lr",
    "adam": "Adam (clamped)",
    "adam_cosine": "Adam (clamped), cosine lr",
    "svrg_momentum": "SVRG + momentum",
    "svrg_adam": "Coordinate SVRG (ours)",
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

    def smoothness_second_moment(self) -> float:
        """(1/n) sum_i L_i^2 with L_i = ||x_i||^2 (per-sample smoothness)."""
        l_i: FloatArray = np.sum(self.features ** 2, axis=1)
        return float(np.mean(l_i ** 2))


@dataclass
class RunResult:
    weights: FloatArray
    final_gap: float
    evals_used: int
    curve_evals: List[float]
    curve_gaps: List[float]
    diverged: bool


def optimize(
    objective: LeastSquares,
    sampler: DeterministicBatchSampler,
    config: ReferenceConfig,
    budget_evals: int,
    w0: Optional[FloatArray] = None,
    checkpoints: Optional[Sequence[float]] = None,
    track_gap: bool = True,
    lr_schedule: str = "const",
) -> RunResult:
    """Run one optimizer for a fixed budget of sample-gradient evaluations."""
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
    diverged: bool = False

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
            ci += 1
        if step % 32 == 0 and (not np.all(np.isfinite(w)) or float(np.linalg.norm(w)) > 1e6):
            diverged = True
            break

    if diverged:
        while ci < len(cps):
            curve_e.append(float(cps[ci]))
            curve_g.append(float("inf"))
            ci += 1
    final_gap: float = float("inf") if diverged else (objective.gap(w) if track_gap else float("nan"))
    return RunResult(w, final_gap, evals, curve_e, curve_g, diverged)


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
def experiment_ablation(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    base: RegimeConfig = RegimeConfig.from_dict(cfg)
    levels: Dict[str, float] = dict(cfg["environment"]["noise_sigma_levels"])
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    batch_size: int = int(cfg["training"]["batch_size"])
    lr_grid: FloatArray = 10.0 ** np.linspace(args.lr_min_exp, args.lr_max_exp, args.lr_points)
    results: Dict[str, Any] = {"lr_grid": lr_grid.tolist(), "regimes": {}}
    seed0: int = int(cfg["seed"])

    for regime, sigma in levels.items():
        tune_sets: List[LeastSquares] = []
        eval_sets: List[LeastSquares] = []
        eval_seeds: List[int] = []
        for i in range(args.tune_seeds):
            ds = make_dataset(base, seed0 + 1000 + i, sigma)
            tune_sets.append(LeastSquares(ds.features, ds.targets))
        for i in range(args.eval_seeds):
            ds = make_dataset(base, seed0 + 2000 + i, sigma)
            eval_sets.append(LeastSquares(ds.features, ds.targets))
            eval_seeds.append(seed0 + 2000 + i)
        n: int = tune_sets[0].n
        budget: int = int(args.epochs * n)
        checkpoints: FloatArray = np.geomspace(4 * n, budget, args.curve_points)
        regime_out: Dict[str, Any] = {"sigma": sigma, "variants": {}}

        for variant in VARIANTS:
            tune_scores: List[float] = []
            for lr in lr_grid:
                gaps: List[float] = []
                for j, obj in enumerate(tune_sets):
                    sampler = DeterministicBatchSampler(n, batch_size, seed0 + 1000 + j)
                    run = optimize(
                        obj, sampler, reference_config(opt_cfg, variant, float(lr)), budget,
                        lr_schedule=VARIANTS[variant][2],
                    )
                    gaps.append(run.final_gap)
                tune_scores.append(float(np.median(gaps)))
            best_idx: int = int(np.argmin(np.where(np.isfinite(tune_scores), tune_scores, np.inf)))
            best_lr: float = float(lr_grid[best_idx])

            final_gaps: List[float] = []
            curves: List[List[float]] = []
            evals_to_target: List[float] = []
            for obj, seed in zip(eval_sets, eval_seeds):
                sampler = DeterministicBatchSampler(n, batch_size, seed)
                initial_gap: float = obj.gap(np.zeros(obj.d))
                run = optimize(
                    obj, sampler, reference_config(opt_cfg, variant, best_lr), budget,
                    checkpoints=checkpoints, lr_schedule=VARIANTS[variant][2],
                )
                final_gaps.append(run.final_gap)
                curves.append(run.curve_gaps)
                target: float = args.target_ratio * initial_gap
                hit: List[float] = [e for e, g in zip(run.curve_evals, run.curve_gaps) if g <= target]
                evals_to_target.append(hit[0] if hit else float("inf"))
            q25, q50, q75 = quartiles(final_gaps)
            curve_arr: FloatArray = np.asarray(curves, dtype=np.float64)
            regime_out["variants"][variant] = {
                "best_lr": best_lr,
                "best_lr_at_grid_edge": bool(best_idx in (0, len(lr_grid) - 1)),
                "tune_scores": tune_scores,
                "final_gap_q25": q25,
                "final_gap_median": q50,
                "final_gap_q75": q75,
                "evals_to_target_median": float(np.median(evals_to_target)),
                "curve_evals": checkpoints.tolist(),
                "curve_q25": np.percentile(curve_arr, 25, axis=0).tolist(),
                "curve_median": np.percentile(curve_arr, 50, axis=0).tolist(),
                "curve_q75": np.percentile(curve_arr, 75, axis=0).tolist(),
            }
            edge_flag: str = " (GRID EDGE)" if best_idx in (0, len(lr_grid) - 1) else ""
            print(f"  [exp1] {regime:<9}{variant:<21} lr={best_lr:.2e}{edge_flag}  final gap median {fmt(q50)}")
        results["regimes"][regime] = regime_out
    results["budget_epochs"] = args.epochs
    results["target_ratio"] = args.target_ratio
    return results


# --------------------------------------------------------------------------- #
# Experiment 2: variance decay along the trajectory
# --------------------------------------------------------------------------- #
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
    opt: ReferenceSVRG = ReferenceSVRG(ref_cfg, obj.d)
    probe_rng = np.random.default_rng(seed0 + 3001)
    lbar2: float = obj.smoothness_second_moment()
    interval: int = ref_cfg.snapshot_interval

    w: FloatArray = np.zeros(obj.d)
    rows_out: List[Dict[str, float]] = []
    for step in range(args.variance_steps):
        if opt.needs_snapshot:
            opt.refresh_snapshot(w, obj.full_grad(w))
        # probe at the moment of maximum staleness inside each snapshot cycle
        if opt.steps_since_snapshot == interval - 1:
            true_grad: FloatArray = obj.full_grad(w)
            delta_sq: float = float(np.sum((w - opt.snapshot) ** 2))  # type: ignore[operator]
            vr_sum = sgd_sum = 0.0
            for _ in range(args.probe_batches):
                rows = np.sort(probe_rng.choice(obj.n, size=batch_size, replace=False))
                g_live = obj.grad(w, rows)
                g_snap = obj.grad(opt.snapshot, rows)  # type: ignore[arg-type]
                g_hat = opt.variance_reduced_gradient(w, g_live, g_snap)
                vr_sum += float(np.sum((g_hat - true_grad) ** 2))
                sgd_sum += float(np.sum((g_live - true_grad) ** 2))
            rows_out.append({
                "step": float(step),
                "var_vr": vr_sum / args.probe_batches,
                "var_sgd": sgd_sum / args.probe_batches,
                "lemma2_bound": lbar2 * delta_sq / batch_size,
                "prop3_bound": lbar2 * obj.d * (lr * ref_cfg.update_clip * interval) ** 2 / batch_size,
                "snapshot_distance": math.sqrt(delta_sq),
                "gap": obj.gap(w),
            })
        rows = sampler.batch(step)
        w = opt.step(w, obj.grad(w, rows), obj.grad(opt.snapshot, rows))  # type: ignore[arg-type]

    within_lemma: float = float(np.mean([r["var_vr"] <= r["lemma2_bound"] for r in rows_out]))
    within_prop: float = float(np.mean([r["var_vr"] <= r["prop3_bound"] for r in rows_out]))
    print(f"  [exp2] probes={len(rows_out)}  within Lemma 2: {within_lemma:.0%}  within Prop 3: {within_prop:.0%}")
    return {
        "rows": rows_out,
        "fraction_within_lemma2": within_lemma,
        "fraction_within_prop3": within_prop,
        "lr": lr,
        "snapshot_interval": interval,
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
                summary[method][metric] = {"q25": q25, "median": q50, "q75": q75}
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
def write_tables(results: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    if "ablation" in results:
        ab = results["ablation"]
        lines += [
            "### Experiment 1: tuned 2x2 ablation",
            "",
            f"Budget: {ab['budget_epochs']} epochs of sample-gradient evaluations. "
            "Final loss gap `f(w) - f(w*)`, median [IQR] over held-out evaluation seeds; "
            f"'evals to target' = sample-gradient evaluations to reach {ab['target_ratio']:g} x the initial gap.",
            "",
        ]
        for regime, block in ab["regimes"].items():
            lines += [
                f"**{regime}** (noise sigma = {block['sigma']})", "",
                "| Method | tuned lr | final gap, median [IQR] | evals to target |",
                "|---|---|---|---|",
            ]
            for variant, r in block["variants"].items():
                target = r["evals_to_target_median"]
                target_txt = "not reached" if math.isinf(target) else f"{target:,.0f}"
                edge = " †" if r.get("best_lr_at_grid_edge") else ""
                lines.append(
                    f"| {VARIANT_LABELS[variant]} | {r['best_lr']:.2e}{edge} | "
                    f"{fmt(r['final_gap_median'])} [{fmt(r['final_gap_q25'])}, {fmt(r['final_gap_q75'])}] | {target_txt} |"
                )
            lines.append("")
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
        "svrg_momentum": "#ff7f0e", "svrg_adam": "#d62728",
    }
    if "ablation" in results:
        regimes = results["ablation"]["regimes"]
        fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (regime, block) in zip(axes, regimes.items()):
            for variant, r in block["variants"].items():
                xs = np.asarray(r["curve_evals"])
                med = np.maximum(np.asarray(r["curve_median"], dtype=float), 1e-18)
                lo = np.maximum(np.asarray(r["curve_q25"], dtype=float), 1e-18)
                hi = np.maximum(np.asarray(r["curve_q75"], dtype=float), 1e-18)
                ax.plot(xs, med, label=VARIANT_LABELS[variant], color=colors[variant], lw=2,
                        ls="--" if "cosine" in variant else "-")
                ax.fill_between(xs, lo, hi, color=colors[variant], alpha=0.15)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{regime} (sigma={block['sigma']})")
            ax.set_xlabel("sample-gradient evaluations")
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("loss gap  f(w) - f(w*)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Tuned 2x2 ablation: median and IQR over held-out seeds")
        fig.tight_layout()
        p = out_dir / "ablation_curves.png"
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
    return written


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stochastic_regime.yaml"))
    parser.add_argument("--only", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "docs" / "figures"))
    # experiment 1
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-points", type=int, default=13)
    parser.add_argument("--lr-min-exp", type=float, default=-5.0)
    parser.add_argument("--lr-max-exp", type=float, default=-0.5)
    parser.add_argument("--tune-seeds", type=int, default=2)
    parser.add_argument("--eval-seeds", type=int, default=7)
    parser.add_argument("--curve-points", type=int, default=40)
    parser.add_argument("--target-ratio", type=float, default=1e-8)
    # experiment 2
    parser.add_argument("--variance-steps", type=int, default=1600)
    parser.add_argument("--probe-batches", type=int, default=64)
    # experiment 3
    parser.add_argument("--wf-samples", type=int, default=8192)
    parser.add_argument("--wf-window", type=int, default=512)
    parser.add_argument("--wf-horizon", type=int, default=64)
    parser.add_argument("--wf-budgets", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--wf-seeds", type=int, default=8)
    args = parser.parse_args()
    if args.quick:
        args.epochs, args.lr_points, args.tune_seeds, args.eval_seeds = 10, 5, 1, 2
        args.curve_points, args.variance_steps, args.probe_batches = 12, 320, 16
        args.wf_samples, args.wf_seeds, args.wf_budgets = 2048, 2, [2048, 8192]

    cfg: Dict[str, Any] = load_config(args.config)
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

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiments.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_tables(results, out_dir / "tables.md")
    figures: List[str] = make_figures(results, Path(args.figures))
    print(f"done in {time.time() - started:.0f}s -> {out_dir}/experiments.json, tables.md, {len(figures)} figure(s)")


if __name__ == "__main__":
    main()
