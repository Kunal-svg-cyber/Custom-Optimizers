"""Mathematical invariant validation for the coordinate-wise SVRG engine.

Layout:
    * environment / sampler determinism   (NumPy only, always runs)
    * NumPy reference oracle + theory     (NumPy only, always runs)
    * PyTorch backend invariants          (skipped if torch is missing)
    * JAX backend invariants              (skipped if jax is missing)
    * cross-backend parity                (skipped unless both are present)

The four invariants under test:
    1. Variance Reduction Law        -> test_*_variance_reduction_*
    2. Deterministic Sample Alignment -> test_*_alignment_*, test_*_step_counter_*
    3. Hardware Contiguity            -> test_torch_flat_contiguity_*
    4. Coordinate Boundary Safeguards -> test_*_zero_*, test_*_tiny_*, test_*_nonfinite_*
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

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
from src.preconditioning import (  # noqa: E402
    jacobi_diag_least_squares,
    jacobi_diag_logistic,
    johnson_zhang_recipe,
)
from src.reference_numpy import ReferenceConfig, ReferenceSVRG  # noqa: E402

try:
    import torch

    from src.torch_optimizer import CoordinateSVRG

    HAS_TORCH: bool = True
except ImportError:  # pragma: no cover - depends on the machine
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    from src.jax_optimizer import (
        SVRGConfig,
        SVRGState,
        compute_full_gradient,
        init_state,
        make_svrg_step,
        refresh_snapshot,
        variance_reduced_gradient,
    )

    HAS_JAX: bool = True
except ImportError:  # pragma: no cover - depends on the machine
    HAS_JAX = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
requires_both = pytest.mark.skipif(
    not (HAS_TORCH and HAS_JAX), reason="needs both torch and jax"
)


# --------------------------------------------------------------------------- #
# Shared problem builders
# --------------------------------------------------------------------------- #
def ls_config(seed: int = 7, n: int = 512, d: int = 8) -> RegimeConfig:
    """Clean least-squares regime: fixed alpha, Gaussian noise only."""
    return RegimeConfig(
        seed=seed,
        n_samples=n,
        n_features=d,
        feature_autocorr=0.2,
        signal_scale=1.0,
        alpha_drift_sigma=0.0,
        regime_switch_prob=0.0,
        noise_sigma=1.0,
        toxic_flow_prob=0.0,
        toxic_flow_scale=0.0,
        toxic_flow_df=3.0,
        jump_prob=0.0,
        jump_scale=0.0,
    )


def ls_problem(
    batch_size: int = 16,
) -> Tuple[MarketDataset, DeterministicBatchSampler, List[np.ndarray]]:
    config: RegimeConfig = ls_config()
    dataset: MarketDataset = generate_market(config)
    sampler = DeterministicBatchSampler(dataset.n_samples, batch_size, config.seed)
    chunks: List[np.ndarray] = sampler.full_pass_chunks(8)
    return dataset, sampler, chunks


def sgd_batch_gradient(dataset: MarketDataset, rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb: np.ndarray = dataset.features[rows]
    yb: np.ndarray = dataset.targets[rows]
    return xb.T @ (xb @ w - yb) / float(len(rows))


# --------------------------------------------------------------------------- #
# Environment and sampler (NumPy only)
# --------------------------------------------------------------------------- #
def test_environment_is_seed_deterministic() -> None:
    config: RegimeConfig = ls_config(seed=11)
    first: MarketDataset = generate_market(config)
    second: MarketDataset = generate_market(config)
    other: MarketDataset = generate_market(dataclasses.replace(config, seed=12))
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.targets, second.targets)
    assert not np.array_equal(first.targets, other.targets)
    assert np.isfinite(first.targets).all()


def test_noise_knobs_do_not_reshuffle_features_or_alpha() -> None:
    config: RegimeConfig = ls_config(seed=3)
    base: MarketDataset = generate_market(config)
    noisier: MarketDataset = generate_market(dataclasses.replace(config, noise_sigma=3.0))
    assert np.array_equal(base.features, noisier.features)
    assert np.array_equal(base.alpha_path, noisier.alpha_path)
    assert not np.array_equal(base.targets, noisier.targets)


def test_snr_falls_as_noise_sigma_rises() -> None:
    config: RegimeConfig = dataclasses.replace(ls_config(n=2048), noise_sigma=0.25)
    calm: float = generate_market(config).empirical_snr_db
    toxic: float = generate_market(dataclasses.replace(config, noise_sigma=3.0)).empirical_snr_db
    assert calm > toxic


def test_default_yaml_regime_is_low_snr() -> None:
    raw: Dict[str, Any] = load_config(REPO_ROOT / "configs" / "stochastic_regime.yaml")
    dataset: MarketDataset = generate_market(RegimeConfig.from_dict(raw))
    assert dataset.empirical_snr_db < 0.0
    assert np.isfinite(dataset.targets).all()


def test_least_squares_optimum_has_zero_gradient() -> None:
    dataset, _, _ = ls_problem()
    grad: np.ndarray = dataset.full_gradient(dataset.optimal_weights)
    assert np.max(np.abs(grad)) < 1e-10


def test_sampler_is_deterministic_and_covers_each_epoch_once() -> None:
    dataset, sampler, _ = ls_problem(batch_size=16)
    assert np.array_equal(sampler.batch(5), sampler.batch(5))
    epoch_rows: np.ndarray = np.concatenate(
        [sampler.batch(i) for i in range(sampler.batches_per_epoch)]
    )
    assert len(np.unique(epoch_rows)) == len(epoch_rows) == dataset.n_samples
    next_epoch_first: np.ndarray = sampler.batch(sampler.batches_per_epoch)
    assert not np.array_equal(next_epoch_first, sampler.batch(0))


def test_full_pass_chunks_partition_the_data() -> None:
    dataset, sampler, chunks = ls_problem()
    joined: np.ndarray = np.concatenate(chunks)
    assert np.array_equal(joined, np.arange(dataset.n_samples))
    with pytest.raises(ValueError):
        sampler.full_pass_chunks(7)


# --------------------------------------------------------------------------- #
# NumPy reference oracle and the analytical results in docs/THEORY.md
# --------------------------------------------------------------------------- #
def test_reference_first_step_is_sign_step() -> None:
    """Bias-corrected first step: m_hat = g, v_hat = g^2, so update = sign(g)."""
    config = ReferenceConfig(lr=0.1, snapshot_interval=10)
    ref = ReferenceSVRG(config, dim=3)
    w0 = np.zeros(3)
    ref.refresh_snapshot(w0, np.array([0.5, -2.0, 0.0]))
    g_live = np.array([0.5, -2.0, 0.0])
    w1 = ref.step(w0, g_live, grad_snapshot=np.zeros(3))
    # g_hat = g_live - 0 + mu = [1.0, -4.0, 0.0] -> update [+1, -1, 0]
    assert np.allclose(w1, w0 - 0.1 * np.array([1.0, -1.0, 0.0]))


def test_reference_ablation_flags_change_behaviour() -> None:
    plain = ReferenceSVRG(ReferenceConfig(variance_reduction=False), dim=2)
    assert not plain.needs_snapshot                       # Adam / SGD need no snapshot
    vr = ReferenceSVRG(ReferenceConfig(variance_reduction=True), dim=2)
    assert vr.needs_snapshot
    with pytest.raises(RuntimeError):
        vr.step(np.zeros(2), np.ones(2), np.ones(2))      # VR without a snapshot
    momentum_only = ReferenceSVRG(
        ReferenceConfig(lr=0.5, beta1=0.0, variance_reduction=False, adaptive=False), dim=2
    )
    w1 = momentum_only.step(np.zeros(2), np.array([2.0, -4.0]))
    assert np.allclose(w1, [-1.0, 2.0])                   # plain SGD step: w - lr * g


def test_reference_zero_and_nonfinite_gradients_are_safe() -> None:
    ref = ReferenceSVRG(ReferenceConfig(lr=0.1), dim=4)
    ref.refresh_snapshot(np.ones(4), np.zeros(4))
    w = np.ones(4)
    for _ in range(5):
        w = ref.step(w, np.zeros(4), np.zeros(4))
    assert np.array_equal(w, np.ones(4))
    w = ref.step(w, np.full(4, np.nan), np.full(4, np.nan))
    assert np.isfinite(w).all()
    assert np.isfinite(ref.exp_avg).all() and np.isfinite(ref.exp_avg_sq).all()
    w = ref.step(w, np.full(4, np.inf), np.zeros(4))
    assert np.isfinite(w).all()


def test_lemma1_variance_reduced_gradient_is_unbiased() -> None:
    """Averaging g_hat over ALL equal-size batches of a partition gives grad F exactly."""
    dataset, _, chunks = ls_problem()
    w_snap = np.linspace(-0.3, 0.3, dataset.n_features)
    w = np.linspace(0.4, -0.2, dataset.n_features)
    mu = dataset.full_gradient(w_snap)
    mean_hat = np.mean(
        [sgd_batch_gradient(dataset, c, w) - sgd_batch_gradient(dataset, c, w_snap) + mu for c in chunks],
        axis=0,
    )
    assert np.allclose(mean_hat, dataset.full_gradient(w), atol=1e-12)


def test_lemma2_bound_and_least_squares_scaling_identity() -> None:
    """Var(g_hat) <= Lbar2/b ||w - w~||^2, and scales exactly with displacement^2."""
    dataset, sampler, chunks = ls_problem(batch_size=16)
    w_star = dataset.optimal_weights
    mu = dataset.full_gradient(w_star)
    direction = np.linspace(-1.0, 1.0, dataset.n_features)
    direction /= np.linalg.norm(direction)
    l_i = np.sum(dataset.features ** 2, axis=1)
    lbar2 = float(np.mean(l_i ** 2))

    def variance(displacement: float) -> float:
        w = w_star + displacement * direction
        true_grad = dataset.full_gradient(w)
        total = 0.0
        for batch_id in range(200):
            rows = sampler.batch(batch_id)
            g_hat = (
                sgd_batch_gradient(dataset, rows, w)
                - sgd_batch_gradient(dataset, rows, w_star)
                + mu
            )
            total += float(np.sum((g_hat - true_grad) ** 2))
        return total / 200.0

    far, near, zero = variance(0.5), variance(0.05), variance(0.0)
    assert far <= lbar2 / 16.0 * 0.5 ** 2
    assert near <= lbar2 / 16.0 * 0.05 ** 2
    assert abs(near / far - 0.01) < 1e-9        # exact quadratic scaling for least squares
    assert zero < 1e-25


def test_proposition3_staleness_bound_holds_for_the_adaptive_engine() -> None:
    """Run the reference engine; at max staleness Var(g_hat) must respect Proposition 3."""
    dataset, sampler, _ = ls_problem(batch_size=16)
    interval = 8
    config = ReferenceConfig(lr=0.05, snapshot_interval=interval, update_clip=10.0)
    ref = ReferenceSVRG(config, dataset.n_features)
    l_i = np.sum(dataset.features ** 2, axis=1)
    lbar2 = float(np.mean(l_i ** 2))
    bound = lbar2 * dataset.n_features * (config.lr * config.update_clip * interval) ** 2 / 16.0
    probe_rng = np.random.default_rng(0)
    w = np.zeros(dataset.n_features)
    checked = 0
    for step in range(96):
        if ref.needs_snapshot:
            ref.refresh_snapshot(w, dataset.full_gradient(w))
        if ref.steps_since_snapshot == interval - 1:
            true_grad = dataset.full_gradient(w)
            total = 0.0
            for _ in range(32):
                rows = np.sort(probe_rng.choice(dataset.n_samples, size=16, replace=False))
                g_hat = ref.variance_reduced_gradient(
                    w, sgd_batch_gradient(dataset, rows, w),
                    sgd_batch_gradient(dataset, rows, ref.snapshot),
                )
                total += float(np.sum((g_hat - true_grad) ** 2))
            assert total / 32.0 <= bound
            checked += 1
        rows = sampler.batch(step)
        w = ref.step(
            w, sgd_batch_gradient(dataset, rows, w),
            sgd_batch_gradient(dataset, rows, ref.snapshot),
        )
    assert checked >= 5


def test_reference_svrg_beats_its_own_noise_floor_ablation() -> None:
    """Same lr: VR keeps converging where the VR-off ablation stalls at a noise floor."""
    dataset, sampler, chunks = ls_problem(batch_size=16)
    w_star = dataset.optimal_weights

    def run(variance_reduction: bool) -> float:
        cfg = ReferenceConfig(lr=0.02, snapshot_interval=64, variance_reduction=variance_reduction)
        ref = ReferenceSVRG(cfg, dataset.n_features)
        w = np.zeros(dataset.n_features)
        for step in range(800):
            if ref.needs_snapshot:
                ref.refresh_snapshot(w, dataset.full_gradient(w))
            rows = sampler.batch(step)
            g_snap = sgd_batch_gradient(dataset, rows, ref.snapshot) if variance_reduction else None
            w = ref.step(w, sgd_batch_gradient(dataset, rows, w), g_snap)
        return float(np.linalg.norm(w - w_star))

    with_vr, without_vr = run(True), run(False)
    assert with_vr < 1e-3
    assert without_vr > 10.0 * with_vr


# Adversarial gradient values: zeros, denormals, tiny, huge, infinities and NaN.
FUZZ_VALUES: List[float] = [
    0.0, 1e-45, 1e-38, 1e-30, 1e-15, 1e-8, 1.0, 1e8, 1e18, 1e30, 3e38,
    -1e-30, -1.0, -1e30, -3e38, float("inf"), float("-inf"), float("nan"),
]


def _fuzz_gradient(rng: np.random.Generator, size: int) -> np.ndarray:
    picks = rng.integers(0, len(FUZZ_VALUES), size=size)
    return np.array([FUZZ_VALUES[i] for i in picks], dtype=np.float64)


def test_reference_fuzz_never_produces_nonfinite_values_or_oversized_steps() -> None:
    """Proposition 4 under adversarial inputs, float32 and float64, extreme betas."""
    rng = np.random.default_rng(2024)
    for dtype in (np.float32, np.float64):
        for trial in range(120):
            cfg = ReferenceConfig(
                lr=float(10 ** rng.uniform(-4, 0)),
                beta1=float(rng.choice([0.0, 0.5, 0.9, 0.99, 0.9999])),
                beta2=float(rng.choice([0.0, 0.9, 0.999, 0.9999])),
                update_clip=float(rng.choice([1.0, 10.0, 100.0])),
                snapshot_interval=int(rng.integers(1, 6)),
            )
            ref = ReferenceSVRG(cfg, dim=6, dtype=dtype)
            w = rng.standard_normal(6).astype(dtype)
            with np.errstate(all="ignore"):
                ref.refresh_snapshot(w, _fuzz_gradient(rng, 6).astype(dtype))
                for _ in range(15):
                    g_live = _fuzz_gradient(rng, 6).astype(dtype)
                    g_snap = _fuzz_gradient(rng, 6).astype(dtype)
                    new_w = ref.step(w, g_live, g_snap)
                    assert np.isfinite(new_w).all(), (dtype, trial)
                    assert np.isfinite(ref.exp_avg).all() and np.isfinite(ref.exp_avg_sq).all()
                    step_size = np.abs(new_w.astype(np.float64) - w.astype(np.float64))
                    assert np.all(step_size <= cfg.lr * cfg.update_clip * 1.001 + 1e-5)
                    w = new_w


# --------------------------------------------------------------------------- #
# Objectives used by the experiment suite
# --------------------------------------------------------------------------- #
def test_logistic_objective_gradient_optimum_and_gap() -> None:
    from experiments.run_experiments import LogisticRegression

    dataset, _, _ = ls_problem()
    labels = np.where(dataset.targets >= 0.0, 1.0, -1.0)
    obj = LogisticRegression(dataset.features, labels, l2=1e-2)
    rng = np.random.default_rng(0)
    w = rng.standard_normal(obj.d) * 0.3
    # analytic gradient vs central finite differences
    numeric = np.array([
        (obj.loss(w + 1e-6 * e) - obj.loss(w - 1e-6 * e)) / 2e-6 for e in np.eye(obj.d)
    ])
    assert np.allclose(obj.full_grad(w), numeric, atol=1e-7)
    # mini-batch gradient over the whole data equals the full gradient
    assert np.allclose(obj.grad(w, np.arange(obj.n)), obj.full_grad(w), atol=1e-12)
    # Newton reaches a stationary point; the gap is zero there and positive elsewhere
    obj.prepare()
    assert np.linalg.norm(obj.full_grad(obj.optimum)) < 1e-10
    assert obj.gap(obj.optimum) == 0.0
    assert obj.gap(w) > 0.0
    # direct difference and local quadratic form agree across the switch-over
    delta = 5e-3 * rng.standard_normal(obj.d)
    near = obj.optimum + delta
    direct = obj.loss(near) - obj.loss(obj.optimum)
    assert abs(obj.gap(near) - direct) / direct < 1e-2


def test_mlp_objective_gradient_matches_finite_differences() -> None:
    from experiments.run_experiments import MLPRegression

    rng = np.random.default_rng(0)
    features = rng.standard_normal((40, 5))
    targets = rng.standard_normal(40)
    obj = MLPRegression(features, targets, hidden=4, init_seed=1)
    w = obj.init_weights + 0.1 * rng.standard_normal(obj.d)
    numeric = np.array([
        (obj.loss(w + 1e-6 * e) - obj.loss(w - 1e-6 * e)) / 2e-6 for e in np.eye(obj.d)
    ])
    assert np.allclose(obj.full_grad(w), numeric, atol=1e-7)
    rows = np.arange(10)
    # mini-batch gradient on a subset equals the full-gradient formula on that subset
    sub = MLPRegression(features[rows], targets[rows], hidden=4, init_seed=1)
    assert np.allclose(obj.grad(w, rows), sub.full_grad(w), atol=1e-12)
    assert abs(obj.gap(w) - float(np.sum(obj.full_grad(w) ** 2))) < 1e-12


def test_paired_bootstrap_interval_covers_the_true_shift() -> None:
    from experiments.run_experiments import paired_bootstrap

    rng = np.random.default_rng(1)
    diffs = 0.5 + 0.1 * rng.standard_normal(30)
    median, low, high = paired_bootstrap(diffs.tolist())
    assert low < 0.5 < high
    assert low <= median <= high
    zero_median, zlow, zhigh = paired_bootstrap((0.1 * rng.standard_normal(30)).tolist())
    assert zlow < 0.0 < zhigh


def test_preconditioned_svrg_is_plain_svrg_in_rescaled_coordinates() -> None:
    """Proposition 5's change of variables u = D^{1/2} w, checked on actual iterates."""
    dataset, sampler, _ = ls_problem()
    d = dataset.n_features
    diag = np.exp(np.random.default_rng(3).uniform(-1.5, 1.5, size=d))
    root = np.sqrt(diag)
    eta, interval, steps = 0.01, 7, 40

    def grad_u(u: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return sgd_batch_gradient(dataset, rows, u / root) / root

    def full_u(u: np.ndarray) -> np.ndarray:
        return dataset.full_gradient(u / root) / root

    w = np.zeros(d)
    u = np.zeros(d)
    w_snap, u_snap = w.copy(), u.copy()
    mu_w, mu_u = dataset.full_gradient(w_snap), full_u(u_snap)
    for k in range(steps):
        if k % interval == 0:
            w_snap, u_snap = w.copy(), u.copy()
            mu_w, mu_u = dataset.full_gradient(w_snap), full_u(u_snap)
        rows = sampler.batch(k)
        g_w = sgd_batch_gradient(dataset, rows, w) - sgd_batch_gradient(dataset, rows, w_snap) + mu_w
        g_u = grad_u(u, rows) - grad_u(u_snap, rows) + mu_u
        w = w - eta * g_w / diag            # preconditioned step in w
        u = u - eta * g_u                   # plain step in u
        assert np.allclose(root * w, u, atol=1e-10), f"coordinates diverged at step {k}"


def test_jacobi_conditioning_is_invariant_to_column_scaling() -> None:
    from experiments.run_experiments import preconditioned_constants

    dataset, _, _ = ls_problem()
    base = preconditioned_constants(dataset.features)
    scales = 10.0 ** np.linspace(-2.0, 2.0, dataset.n_features)
    scaled = preconditioned_constants(dataset.features * scales)
    assert abs(scaled["kappa_jacobi"] - base["kappa_jacobi"]) / base["kappa_jacobi"] < 1e-6
    assert scaled["kappa"] > 1e3 * base["kappa"]            # raw conditioning explodes
    assert scaled["kappa_ratio"] > 1e3


def test_johnson_zhang_recipe_contracts_each_epoch_in_expectation() -> None:
    """Proposition 5 / Johnson-Zhang: E[gap_s] <= alpha * gap_{s-1} with zero tuning."""
    from experiments.run_experiments import LeastSquares, diag_precond_svrg, theory_constants

    dataset, _, _ = ls_problem()
    obj = LeastSquares(dataset.features, dataset.targets)
    obj.prepare()
    consts = theory_constants(dataset.features)
    assert consts["alpha"] <= 0.5 + 1e-12                  # the recipe is built to give alpha = 1/2
    diag = np.mean(dataset.features ** 2, axis=0)
    ratios = []
    for seed in range(4):
        run = diag_precond_svrg(
            obj, diag, consts["eta"], int(consts["inner_steps"]), 1,
            budget_evals=int(7 * (obj.n + 2 * consts["inner_steps"])), snapshot_rule="random", seed=seed,
        )
        assert not run.diverged
        ratios += [b / a for a, b in zip(run.gaps[:-1], run.gaps[1:]) if a > 1e-20]
    assert len(ratios) >= 20
    assert float(np.mean(ratios)) <= consts["alpha"]       # the guarantee is in expectation
    assert run.gaps[-1] < 1e-3 * run.gaps[0]                # and the iteration really converges


def test_diag_precond_svrg_with_unit_diagonal_is_plain_svrg() -> None:
    from experiments.run_experiments import LeastSquares, diag_precond_svrg

    dataset, sampler, _ = ls_problem(batch_size=16)
    obj = LeastSquares(dataset.features, dataset.targets)
    obj.prepare()
    run = diag_precond_svrg(
        obj, np.ones(obj.d), 0.05, 32, 16, budget_evals=obj.n * 40, snapshot_rule="last", seed=0,
        sampler=sampler,
    )
    assert not run.diverged and run.gaps[-1] < 1e-9 * run.gaps[0]


def test_real_datasets_have_badly_scaled_raw_features() -> None:
    pytest.importorskip("sklearn")
    from experiments.run_experiments import REAL_TASKS, load_real_task

    for name in REAL_TASKS:
        obj = load_real_task(name, 1e-3)
        std = obj.features[:, :-1].std(axis=0)
        assert np.log10(std.max() / std.min()) > 1.5          # genuinely badly scaled
        assert np.all(obj.features[:, -1] == 1.0)              # intercept column
        obj.prepare()
        assert np.isfinite(obj.gap(np.zeros(obj.d)))


def test_hessian_conditioning_jacobi_is_invariant_to_diagonal_rescaling() -> None:
    from experiments.run_experiments import hessian_conditioning

    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 6))
    hess = a.T @ a + 0.1 * np.eye(6)
    scale = 10.0 ** np.linspace(-2, 2, 6)
    hess_scaled = hess * np.outer(scale, scale)
    base = hessian_conditioning(hess, np.diag(hess))
    scaled = hessian_conditioning(hess_scaled, np.diag(hess_scaled))
    assert abs(scaled["cond_jacobi"] - base["cond_jacobi"]) / base["cond_jacobi"] < 1e-6
    assert scaled["cond_raw"] > 1e3 * base["cond_raw"]


def test_misaligned_batches_are_worse_than_plain_sgd_and_aligned_batches_win() -> None:
    dataset, sampler, _ = ls_problem(batch_size=16)
    w_star = dataset.optimal_weights
    mu = dataset.full_gradient(w_star)
    direction = np.linspace(-1.0, 1.0, dataset.n_features)
    direction /= np.linalg.norm(direction)
    w = w_star + 0.05 * direction
    true_grad = dataset.full_gradient(w)
    rng = np.random.default_rng(1)
    err = {"aligned": 0.0, "misaligned": 0.0, "sgd": 0.0}
    for _ in range(300):
        rows = np.sort(rng.choice(dataset.n_samples, size=16, replace=False))
        other = np.sort(rng.choice(dataset.n_samples, size=16, replace=False))
        g = sgd_batch_gradient(dataset, rows, w)
        err["aligned"] += float(np.sum((g - sgd_batch_gradient(dataset, rows, w_star) + mu - true_grad) ** 2))
        err["misaligned"] += float(np.sum((g - sgd_batch_gradient(dataset, other, w_star) + mu - true_grad) ** 2))
        err["sgd"] += float(np.sum((g - true_grad) ** 2))
    assert err["aligned"] < 0.05 * err["sgd"]            # alignment is what delivers the reduction
    assert err["misaligned"] > err["sgd"]                # misalignment is worse than no control variate


def test_naive_adam_fails_where_the_reference_engine_does_not() -> None:
    from experiments.run_experiments import INVARIANT_FUZZ_VALUES, _naive_adam_steps

    benign = [1.0, -1.0, 0.5]
    assert _naive_adam_steps("naive", np.random.default_rng(0), 10, 0.9, 0.999, 0.01, benign)
    failures = sum(
        not _naive_adam_steps("naive", np.random.default_rng(t), 10, 0.9, 0.999, 0.1, INVARIANT_FUZZ_VALUES)
        for t in range(40)
    )
    assert failures > 20                                   # the textbook update breaks on hostile inputs


def test_market_features_are_causal() -> None:
    """Features on day t must not change if any later price or volume changes."""
    from experiments.real_market_study import build_features, synthetic_panel

    close, volume = synthetic_panel("signal", 6, 400, 0)
    feats, _ = build_features(close, volume)
    t = 250
    close2, volume2 = close.copy(), volume.copy()
    rng = np.random.default_rng(9)
    close2[t + 1:] *= rng.uniform(0.5, 2.0, size=close2[t + 1:].shape)
    volume2[t + 1:] *= rng.uniform(0.1, 10.0, size=volume2[t + 1:].shape)
    feats2, _ = build_features(close2, volume2)
    assert np.allclose(feats[: t + 1], feats2[: t + 1], equal_nan=True)
    truncated, _ = build_features(close[: t + 1], volume[: t + 1])
    assert np.allclose(truncated, feats[: t + 1], equal_nan=True)


def test_walk_forward_predictions_have_no_lookahead() -> None:
    """Predictions for blocks inside a short history must equal those made with extra future data."""
    from experiments.real_market_study import (
        METHODS, build_features, synthetic_panel, walk_forward_predictions,
    )

    close, volume = synthetic_panel("signal", 8, 700, 1)
    feats_full, target_full = build_features(close, volume)
    short = 520
    feats_short, target_short = build_features(close[:short], volume[:short])
    kwargs = dict(window=252, horizon=21, budget_epochs=2.0, batch=64, seed=0)
    full = walk_forward_predictions(feats_full, target_full, **kwargs)
    part = walk_forward_predictions(feats_short, target_short, **kwargs)
    limit = short - 1 - 21
    compared = 0
    for name in METHODS:
        a, b = full[name][:limit], part[name][:limit]
        assert np.allclose(a, b, equal_nan=True), name
        compared += int(np.isfinite(b).sum())
    assert compared > 0                                     # the comparison covered real predictions


def test_signal_is_detected_and_placebo_is_not() -> None:
    """With a clearly detectable synthetic signal the pipeline finds it on every seed; shifted targets do not."""
    from experiments.real_market_study import (
        build_features, placebo_target, summarise, synthetic_panel, walk_forward_predictions,
    )

    real, null = [], []
    for seed in (3, 4, 5):
        close, volume = synthetic_panel("signal", 10, 1800, seed, strength=3.0)
        feats, target = build_features(close, volume)
        preds = walk_forward_predictions(feats, target, window=504, horizon=21, budget_epochs=2.0, batch=64, seed=0)
        real.append(summarise(preds, target, cost_bps=5.0, seed=0)["table"]["ols"]["mean_daily_ic"])
        null.append(summarise(preds, placebo_target(target, 7), cost_bps=5.0, seed=0)["table"]["ols"]["mean_daily_ic"])
    assert min(real) > 0.05                                 # detected on every seed
    assert max(abs(x) for x in null) < 0.03                 # circularly shifted targets: no signal


def test_placebo_target_preserves_marginals_and_bootstrap_pairing() -> None:
    from experiments.real_market_study import paired_block_bootstrap, placebo_target

    rng = np.random.default_rng(0)
    target = rng.standard_normal((300, 4))
    shifted = placebo_target(target, 1)
    assert np.allclose(np.sort(target, axis=0), np.sort(shifted, axis=0))
    assert not np.allclose(target, shifted)
    series = {"a": rng.standard_normal(400), "b": None}
    series["b"] = series["a"].copy()
    est, lo, hi = paired_block_bootstrap(series, "a", "b", lambda v: float(np.mean(v)), draws=200)
    assert est == 0.0 and lo == 0.0 and hi == 0.0           # identical series: zero difference, zero width
    series["b"] = series["a"] - 0.5
    est, lo, hi = paired_block_bootstrap(series, "a", "b", lambda v: float(np.mean(v)), draws=200)
    assert abs(est - 0.5) < 1e-12 and lo > 0.45 and hi < 0.55


def test_csv_panel_loader_aligns_on_common_dates() -> None:
    import csv
    import tempfile

    from experiments.real_market_study import load_csv_panel, synthetic_panel

    close, volume = synthetic_panel("null", 6, 120, 0)
    with tempfile.TemporaryDirectory() as directory:
        for j in range(6):
            first = 5 if j == 0 else 0                       # asset 0 starts later: common dates shrink
            with open(Path(directory) / f"A{j}.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Date", "Close", "Adj Close", "Volume"])
                for t in range(first, 120):
                    writer.writerow([f"2020-{1 + t // 28:02d}-{1 + t % 28:02d}", 0.0, close[t, j], volume[t, j]])
        names, loaded_close, loaded_volume = load_csv_panel(Path(directory))
    assert names == [f"A{j}" for j in range(6)]
    assert loaded_close.shape == (115, 6) and loaded_volume.shape == (115, 6)
    assert np.allclose(loaded_close[:, 1], close[5:, 1])      # Adj Close is preferred over Close


def test_report_numbers_are_in_sync_with_stored_results_and_macros_are_defined() -> None:
    """The report's numbers are generated from results/experiments.json; neither may drift or dangle."""
    import re
    import subprocess

    check = subprocess.run(
        [sys.executable, str(REPO_ROOT / "paper" / "make_numbers.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert check.returncode == 0, check.stdout + check.stderr
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (REPO_ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")))
    used = set(re.findall(r"\\(Exp[A-Za-z]+)", (REPO_ROOT / "paper" / "technical_report.tex").read_text(encoding="utf-8")))
    assert used, "report uses no generated macros"
    assert used <= defined, f"undefined macros in the report: {sorted(used - defined)}"


def test_least_squares_gap_matches_loss_difference() -> None:
    from experiments.run_experiments import LeastSquares

    dataset, _, _ = ls_problem()
    obj = LeastSquares(dataset.features, dataset.targets)
    w = np.linspace(-1.0, 1.0, obj.d)
    assert abs(obj.gap(w) - (obj.loss(w) - obj.optimal_loss)) < 1e-12


def test_reference_frozen_preconditioner_step_and_validation() -> None:
    cfg = ReferenceConfig(lr=0.5, beta1=0.0, variance_reduction=False, adaptive=True)
    ref = ReferenceSVRG(cfg, dim=2)
    ref.set_preconditioner(np.array([2.0, 4.0]))            # takes precedence over `adaptive`
    w1 = ref.step(np.zeros(2), np.array([1.0, -2.0]))
    assert np.allclose(w1, [-0.5 * 1.0 / 2.0, 0.5 * 2.0 / 4.0])
    with pytest.raises(ValueError):
        ref.set_preconditioner(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        ref.set_preconditioner(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        ref.set_preconditioner(np.array([1.0, float("nan")]))


def test_preconditioning_helpers_match_definitions() -> None:
    from experiments.run_experiments import LogisticRegression

    dataset, _, _ = ls_problem()
    x = dataset.features
    assert np.allclose(jacobi_diag_least_squares(x), np.diag(x.T @ x / x.shape[0]))
    labels = np.where(dataset.targets >= 0.0, 1.0, -1.0)
    obj = LogisticRegression(x, labels, l2=1e-2)
    bound = jacobi_diag_logistic(x, 1e-2)
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.standard_normal(x.shape[1])
        assert np.all(np.diag(obj._hessian(w)) <= bound + 1e-12)    # a valid curvature upper bound
    recipe = johnson_zhang_recipe(x, jacobi_diag_least_squares(x))
    assert recipe["alpha"] <= 0.5 + 1e-12
    with pytest.raises(ValueError):
        jacobi_diag_logistic(x, -1.0)


# --------------------------------------------------------------------------- #
# PyTorch backend
# --------------------------------------------------------------------------- #
def _torch_ls_setup(
    dtype: "torch.dtype" = None,  # type: ignore[assignment]
) -> Tuple[Any, ...]:
    dtype = torch.float64 if dtype is None else dtype
    dataset, sampler, chunks = ls_problem()
    features = torch.as_tensor(dataset.features, dtype=dtype)
    targets = torch.as_tensor(dataset.targets, dtype=dtype)
    w = torch.nn.Parameter(torch.zeros(dataset.n_features, dtype=dtype))

    def loss_on(rows: np.ndarray) -> "torch.Tensor":
        idx = torch.as_tensor(rows)
        residual = features[idx] @ w - targets[idx]
        return 0.5 * (residual * residual).mean()

    def batch_closure(batch_id: int) -> "torch.Tensor":
        return loss_on(sampler.batch(batch_id))

    def chunk_closure(chunk_id: int) -> "torch.Tensor":
        return loss_on(chunks[chunk_id])

    return dataset, sampler, chunks, w, batch_closure, chunk_closure


@requires_torch
def test_torch_zero_gradients_are_numerically_safe() -> None:
    w = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32))
    opt = CoordinateSVRG([w], lr=0.1, eps=1e-15)

    def zero_loss(_: int) -> "torch.Tensor":
        return (w * 0.0).sum()

    opt.refresh_snapshot(zero_loss, num_chunks=2)
    for _ in range(5):
        opt.step(zero_loss)
    assert torch.isfinite(w).all()
    assert torch.equal(w.detach(), torch.zeros(5))
    assert opt.last_stats["min_denominator"] >= 0.99e-15
    assert opt.last_stats["nonfinite_count"] == 0.0


@requires_torch
def test_torch_tiny_gradients_do_not_overflow() -> None:
    w = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = CoordinateSVRG([w], lr=0.1)

    def tiny_loss(_: int) -> "torch.Tensor":
        return (w * 1e-20).sum()

    opt.refresh_snapshot(tiny_loss)
    for _ in range(10):
        opt.step(tiny_loss)
    assert torch.isfinite(w).all()
    assert torch.isfinite(torch.stack(list(opt.flat_state().values()))).all()


@requires_torch
def test_torch_nonfinite_gradients_are_sanitised() -> None:
    w = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = CoordinateSVRG([w], lr=0.1)

    def good_loss(_: int) -> "torch.Tensor":
        return (w * 2.0).sum()

    def nan_loss(_: int) -> "torch.Tensor":
        return (w * float("nan")).sum()

    opt.refresh_snapshot(good_loss)
    opt.step(nan_loss)
    assert torch.isfinite(w).all()
    assert torch.isfinite(torch.stack(list(opt.flat_state().values()))).all()
    assert opt.last_stats["nonfinite_count"] > 0.0


@requires_torch
def test_torch_step_requires_snapshot() -> None:
    w = torch.nn.Parameter(torch.zeros(3))
    opt = CoordinateSVRG([w])
    assert opt.needs_snapshot
    with pytest.raises(RuntimeError):
        opt.step(lambda _: (w * 1.0).sum())


@requires_torch
def test_torch_alignment_and_step_counter_across_batch_shifts() -> None:
    w = torch.nn.Parameter(torch.zeros(3))
    opt = CoordinateSVRG([w], lr=0.01, snapshot_interval=100)
    seen: List[int] = []

    def recording_closure(batch_id: int) -> "torch.Tensor":
        seen.append(batch_id)
        return ((w - 1.0) ** 2).sum()

    opt.refresh_snapshot(lambda _: ((w - 1.0) ** 2).sum())
    assert opt.step_count == 0
    for _ in range(3):
        opt.step(recording_closure)
    # Each step evaluates live then snapshot on the SAME id; ids shift per step.
    assert seen == [0, 0, 1, 1, 2, 2]
    assert opt.step_count == 3
    assert opt.last_batch_id == 2

    seen.clear()
    opt.step(recording_closure, batch_id=7)
    assert seen == [7, 7]
    assert opt.step_count == 4
    assert opt.param_groups[0]["steps_since_snapshot"] == 4


@requires_torch
def test_torch_snapshot_refresh_resets_counter_and_flags_staleness() -> None:
    w = torch.nn.Parameter(torch.zeros(3))
    opt = CoordinateSVRG([w], lr=0.01, snapshot_interval=2)

    def loss(_: int) -> "torch.Tensor":
        return ((w - 1.0) ** 2).sum()

    opt.refresh_snapshot(loss)
    assert not opt.needs_snapshot
    opt.step(loss)
    opt.step(loss)
    assert opt.needs_snapshot
    opt.refresh_snapshot(loss)
    assert not opt.needs_snapshot
    assert opt.param_groups[0]["steps_since_snapshot"] == 0


@requires_torch
def test_torch_flat_contiguity_of_params_and_state() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 3)
    before = model(inputs).detach().clone()
    opt = CoordinateSVRG(model.parameters(), lr=0.01)

    def loss(_: int) -> "torch.Tensor":
        return torch.nn.functional.mse_loss(model(inputs), targets)

    opt.refresh_snapshot(loss)
    flat = opt.flat_parameters()
    assert flat.is_contiguous()
    base_storage: int = flat.untyped_storage().data_ptr()
    itemsize: int = flat.element_size()
    expected_offset: int = 0
    for p in model.parameters():
        assert p.untyped_storage().data_ptr() == base_storage
        assert p.data_ptr() - flat.data_ptr() == expected_offset * itemsize
        assert p.is_contiguous()
        expected_offset += p.numel()
    assert expected_offset == flat.numel()
    # Flattening must not change the function the model computes.
    assert torch.allclose(model(inputs).detach(), before)

    state_rows: Dict[str, "torch.Tensor"] = opt.flat_state()
    state_storage: int = state_rows["snapshot"].untyped_storage().data_ptr()
    for row in state_rows.values():
        assert row.is_contiguous()
        assert row.untyped_storage().data_ptr() == state_storage
    for p in model.parameters():
        for key in ("snapshot", "mu", "exp_avg", "exp_avg_sq"):
            assert opt.state[p][key].untyped_storage().data_ptr() == state_storage

    snapshot_before = state_rows["snapshot"].clone()
    opt.step(loss)
    # The snapshot is frozen: stepping must not move it, but must move the params.
    assert torch.equal(opt.flat_state()["snapshot"], snapshot_before)
    assert not torch.equal(opt.flat_parameters(), snapshot_before)


@requires_torch
def test_torch_gradient_pair_leaves_live_weights_untouched() -> None:
    _, _, _, w, batch_closure, chunk_closure = _torch_ls_setup()
    opt = CoordinateSVRG([w], lr=0.01)
    opt.refresh_snapshot(chunk_closure, num_chunks=8)
    with torch.no_grad():
        w.copy_(torch.linspace(-1.0, 1.0, w.numel(), dtype=w.dtype))
    before = w.detach().clone()
    opt.variance_reduced_gradient(batch_closure, 3)
    assert torch.equal(w.detach(), before)


@requires_torch
def test_torch_variance_reduction_law() -> None:
    """E||g_hat - grad f(w_t)||^2 scales like ||w_t - w~||^2 and vanishes at w~.

    For least squares g_hat - grad f(w_t) = (H_B - H)(w_t - w~), so scaling the
    displacement by 1/10 must shrink the variance by exactly 1/100, while plain
    mini-batch SGD keeps a noise floor at the optimum.
    """
    dataset, sampler, _, w, batch_closure, chunk_closure = _torch_ls_setup()
    opt = CoordinateSVRG([w], lr=0.01, snapshot_interval=10**9)
    w_star = torch.as_tensor(dataset.optimal_weights, dtype=w.dtype)
    with torch.no_grad():
        w.copy_(w_star)
    opt.refresh_snapshot(chunk_closure, num_chunks=8)

    direction = torch.linspace(-1.0, 1.0, w.numel(), dtype=w.dtype)
    direction = direction / torch.linalg.vector_norm(direction)

    def variances(displacement: float) -> Tuple[float, float]:
        with torch.no_grad():
            w.copy_(w_star + displacement * direction)
        w_np: np.ndarray = w.detach().numpy().copy()
        true_grad: np.ndarray = dataset.full_gradient(w_np)
        vr_total: float = 0.0
        sgd_total: float = 0.0
        num_batches: int = 200
        for batch_id in range(num_batches):
            g_hat = opt.variance_reduced_gradient(batch_closure, batch_id)[0].numpy()
            g_sgd = sgd_batch_gradient(dataset, sampler.batch(batch_id), w_np)
            vr_total += float(np.sum((g_hat - true_grad) ** 2))
            sgd_total += float(np.sum((g_sgd - true_grad) ** 2))
        return vr_total / num_batches, sgd_total / num_batches

    vr_far, _ = variances(0.5)
    vr_near, sgd_near = variances(0.05)
    assert 0.005 < vr_near / vr_far < 0.02          # ~ (0.05 / 0.5)^2 = 0.01
    assert vr_near < 0.05 * sgd_near                # SGD noise floor does not vanish

    vr_at_snapshot, _ = variances(0.0)
    assert vr_at_snapshot < 1e-20                   # exactly zero at w = w~


@requires_torch
def test_torch_converges_past_the_sgd_noise_floor() -> None:
    dataset, sampler, chunks, w, batch_closure, chunk_closure = _torch_ls_setup()
    opt = CoordinateSVRG([w], lr=0.02, snapshot_interval=64)
    w_star = torch.as_tensor(dataset.optimal_weights, dtype=w.dtype)
    initial_distance: float = float(torch.linalg.vector_norm(w.detach() - w_star))
    for _ in range(800):
        if opt.needs_snapshot:
            opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
        opt.step(batch_closure)
    final_distance: float = float(torch.linalg.vector_norm(w.detach() - w_star))
    assert final_distance < 0.02 * initial_distance
    assert opt.last_stats["snapshot_distance"] < 0.5


@requires_torch
def test_torch_checkpoint_round_trip_preserves_trajectory() -> None:
    dataset, _, chunks, w, batch_closure, chunk_closure = _torch_ls_setup()
    opt = CoordinateSVRG([w], lr=0.02, snapshot_interval=1000)
    opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
    for _ in range(5):
        opt.step(batch_closure)
    saved: Dict[str, Any] = copy.deepcopy(opt.state_dict())
    saved_params = w.detach().clone()

    dataset2, sampler2, chunks2, w2, batch_closure2, _ = _torch_ls_setup()
    with torch.no_grad():
        w2.copy_(saved_params)
    restored = CoordinateSVRG([w2], lr=0.02, snapshot_interval=1000)
    restored.load_state_dict(saved)
    assert not restored.needs_snapshot
    assert restored.step_count == opt.step_count

    for _ in range(4):
        opt.step(batch_closure)
        restored.step(batch_closure2)
    assert torch.allclose(w.detach(), w2.detach(), atol=1e-12)


# --------------------------------------------------------------------------- #
# JAX backend
# --------------------------------------------------------------------------- #
def _jax_ls_setup() -> Tuple[Any, ...]:
    dataset, sampler, chunks = ls_problem()
    features = jnp.asarray(dataset.features, dtype=jnp.float32)
    targets = jnp.asarray(dataset.targets, dtype=jnp.float32)

    def loss_fn(params: Any, batch: Any) -> Any:
        x, y = batch
        residual = x @ params - y
        return 0.5 * jnp.mean(residual * residual)

    def batch_at(batch_id: int) -> Tuple[Any, Any]:
        rows = sampler.batch(batch_id)
        return features[rows], targets[rows]

    chunk_stack = (
        jnp.stack([features[c] for c in chunks]),
        jnp.stack([targets[c] for c in chunks]),
    )
    return dataset, sampler, loss_fn, batch_at, chunk_stack


@requires_jax
def test_jax_state_is_a_registered_pytree() -> None:
    params = {"w": jnp.ones((3,)), "b": jnp.zeros(())}
    state = init_state(params)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, SVRGState)
    assert len(leaves) == 3 + 4 * 2  # 3 scalars + 4 pytrees of 2 leaves each
    doubled = jax.jit(lambda s: jax.tree_util.tree_map(lambda x: x, s))(state)
    assert isinstance(doubled, SVRGState)


@requires_jax
def test_jax_zero_gradients_are_numerically_safe() -> None:
    params = {"w": jnp.zeros((5,), dtype=jnp.float32)}

    def zero_loss(p: Any, batch: Any) -> Any:
        return jnp.sum(p["w"] * 0.0)

    config = SVRGConfig(lr=0.1)
    step = make_svrg_step(zero_loss, config)
    state = refresh_snapshot(params, init_state(params), {"w": jnp.zeros((5,))})
    for _ in range(5):
        params, state, stats = step(params, state, None)
    assert bool(jnp.all(jnp.isfinite(params["w"])))
    assert bool(jnp.all(params["w"] == 0.0))
    assert float(stats["min_denominator"]) >= 0.99e-15
    assert float(stats["nonfinite_count"]) == 0.0


@requires_jax
def test_jax_tiny_and_nonfinite_gradients_are_safe() -> None:
    params = {"w": jnp.ones((4,), dtype=jnp.float32)}
    config = SVRGConfig(lr=0.1)
    tiny_step = make_svrg_step(lambda p, b: jnp.sum(p["w"] * 1e-20), config)
    state = refresh_snapshot(params, init_state(params), {"w": jnp.full((4,), 1e-20)})
    p, s = params, state
    for _ in range(10):
        p, s, _ = tiny_step(p, s, None)
    assert bool(jnp.all(jnp.isfinite(p["w"])))

    nan_step = make_svrg_step(lambda p, b: jnp.sum(p["w"] * jnp.nan), config)
    p2, s2, stats = nan_step(params, state, None)
    assert bool(jnp.all(jnp.isfinite(p2["w"])))
    assert bool(jnp.all(jnp.isfinite(s2.exp_avg["w"])))
    assert bool(jnp.all(jnp.isfinite(s2.exp_avg_sq["w"])))
    assert float(stats["nonfinite_count"]) > 0.0


@requires_jax
def test_jax_step_counters_and_snapshot_reset() -> None:
    dataset, _, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
    config = SVRGConfig(lr=0.02, snapshot_interval=2)
    step = make_svrg_step(loss_fn, config)
    params = jnp.zeros((dataset.n_features,), dtype=jnp.float32)
    state = init_state(params)
    assert int(state.step) == 0
    for batch_id in range(3):
        params, state, _ = step(params, state, batch_at(batch_id))
    assert int(state.step) == 3
    assert int(state.steps_since_snapshot) == 3

    full_grad = compute_full_gradient(loss_fn, params, chunk_stack)
    state = refresh_snapshot(params, state, full_grad)
    assert int(state.step) == 3                      # global counter untouched
    assert int(state.steps_since_snapshot) == 0
    assert bool(state.snapshot_ready)


@requires_jax
def test_jax_alignment_both_gradients_see_identical_batch() -> None:
    dataset, _, loss_fn, batch_at, _ = _jax_ls_setup()
    checksums: List[float] = []

    def recording_loss(p: Any, batch: Any) -> Any:
        checksums.append(float(jnp.sum(batch[0])))
        return loss_fn(p, batch)

    config = SVRGConfig(lr=0.02)
    step = make_svrg_step(recording_loss, config)
    params = jnp.zeros((dataset.n_features,), dtype=jnp.float32)
    state = refresh_snapshot(params, init_state(params), jnp.zeros_like(params))
    with jax.disable_jit():
        for batch_id in range(3):
            params, state, _ = step(params, state, batch_at(batch_id))
    # Two evaluations per step (live + snapshot) share one batch checksum.
    assert len(checksums) == 6
    for i in range(3):
        assert checksums[2 * i] == checksums[2 * i + 1]
    assert checksums[0] != checksums[2]


@requires_jax
def test_jax_jit_matches_eager_execution() -> None:
    dataset, _, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
    config = SVRGConfig(lr=0.02)
    step = make_svrg_step(loss_fn, config)
    params0 = jnp.zeros((dataset.n_features,), dtype=jnp.float32)
    state0 = refresh_snapshot(
        params0, init_state(params0), compute_full_gradient(loss_fn, params0, chunk_stack)
    )
    p_jit, s_jit, _ = step(params0, state0, batch_at(1))
    with jax.disable_jit():
        p_eager, s_eager, _ = step(params0, state0, batch_at(1))
    assert np.allclose(np.asarray(p_jit), np.asarray(p_eager), atol=1e-6)
    assert np.allclose(np.asarray(s_jit.exp_avg), np.asarray(s_eager.exp_avg), atol=1e-6)


@requires_jax
def test_jax_variance_reduction_law() -> None:
    dataset, sampler, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
    w_star = jnp.asarray(dataset.optimal_weights, dtype=jnp.float32)
    state = refresh_snapshot(
        w_star, init_state(w_star), compute_full_gradient(loss_fn, w_star, chunk_stack)
    )
    direction = np.linspace(-1.0, 1.0, dataset.n_features)
    direction = direction / np.linalg.norm(direction)
    grad_fn = jax.jit(lambda p, b: variance_reduced_gradient(loss_fn, p, state, b))

    def variances(displacement: float) -> Tuple[float, float]:
        w = w_star + jnp.asarray(displacement * direction, dtype=jnp.float32)
        w_np = np.asarray(w, dtype=np.float64)
        true_grad = dataset.full_gradient(w_np)
        vr_total, sgd_total, num_batches = 0.0, 0.0, 200
        for batch_id in range(num_batches):
            g_hat = np.asarray(grad_fn(w, batch_at(batch_id)), dtype=np.float64)
            g_sgd = sgd_batch_gradient(dataset, sampler.batch(batch_id), w_np)
            vr_total += float(np.sum((g_hat - true_grad) ** 2))
            sgd_total += float(np.sum((g_sgd - true_grad) ** 2))
        return vr_total / num_batches, sgd_total / num_batches

    vr_far, _ = variances(0.5)
    vr_near, sgd_near = variances(0.05)
    assert 0.005 < vr_near / vr_far < 0.02
    assert vr_near < 0.05 * sgd_near
    vr_at_snapshot, _ = variances(0.0)
    assert vr_at_snapshot < 1e-9                    # float32 round-off only


@requires_jax
def test_jax_converges_past_the_sgd_noise_floor() -> None:
    dataset, _, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
    config = SVRGConfig(lr=0.02, snapshot_interval=64)
    step = make_svrg_step(loss_fn, config)
    params = jnp.zeros((dataset.n_features,), dtype=jnp.float32)
    state = init_state(params)
    w_star = np.asarray(dataset.optimal_weights)
    initial = float(np.linalg.norm(np.asarray(params) - w_star))
    steps_since = config.snapshot_interval
    for batch_id in range(800):
        if steps_since >= config.snapshot_interval:
            state = refresh_snapshot(
                params, state, compute_full_gradient(loss_fn, params, chunk_stack)
            )
            steps_since = 0
        params, state, _ = step(params, state, batch_at(batch_id))
        steps_since += 1
    final = float(np.linalg.norm(np.asarray(params) - w_star))
    assert final < 0.02 * initial


# --------------------------------------------------------------------------- #
# Cross-backend parity
# --------------------------------------------------------------------------- #
@requires_both
def test_torch_and_jax_agree_step_for_step() -> None:
    dataset, sampler, chunks = ls_problem()
    d: int = dataset.n_features
    w1: np.ndarray = np.linspace(-0.5, 0.5, d).astype(np.float32)
    w0: np.ndarray = np.zeros(d, dtype=np.float32)

    # --- torch (float32) ---
    features_t = torch.as_tensor(dataset.features, dtype=torch.float32)
    targets_t = torch.as_tensor(dataset.targets, dtype=torch.float32)
    w_t = torch.nn.Parameter(torch.as_tensor(w0.copy()))

    def torch_loss(rows: np.ndarray) -> "torch.Tensor":
        idx = torch.as_tensor(rows)
        residual = features_t[idx] @ w_t - targets_t[idx]
        return 0.5 * (residual * residual).mean()

    opt = CoordinateSVRG([w_t], lr=0.02)
    opt.refresh_snapshot(lambda c: torch_loss(chunks[c]), num_chunks=len(chunks))
    with torch.no_grad():
        w_t.copy_(torch.as_tensor(w1))

    # --- jax (float32) ---
    _, _, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
    step = make_svrg_step(loss_fn, SVRGConfig(lr=0.02))
    params = jnp.asarray(w0)
    state = refresh_snapshot(
        params, init_state(params), compute_full_gradient(loss_fn, params, chunk_stack)
    )
    params = jnp.asarray(w1)

    for batch_id in range(3):
        opt.step(lambda b: torch_loss(sampler.batch(b)), batch_id=batch_id)
        params, state, _ = step(params, state, batch_at(batch_id))
        assert np.allclose(w_t.detach().numpy(), np.asarray(params), atol=1e-4)


# --------------------------------------------------------------------------- #
# Differential tests: engines vs the independent NumPy oracle
# --------------------------------------------------------------------------- #
@requires_torch
def test_torch_matches_numpy_reference_trajectory() -> None:
    dataset, sampler, chunks, w, batch_closure, chunk_closure = _torch_ls_setup()
    interval, steps = 5, 14
    opt = CoordinateSVRG([w], lr=0.02, snapshot_interval=interval)
    ref = ReferenceSVRG(ReferenceConfig(lr=0.02, snapshot_interval=interval), dataset.n_features)
    w_ref = np.zeros(dataset.n_features)
    for k in range(steps):
        if opt.needs_snapshot:
            opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
        opt.step(batch_closure)
        if ref.needs_snapshot:
            ref.refresh_snapshot(w_ref, dataset.full_gradient(w_ref))
        rows = sampler.batch(k)
        w_ref = ref.step(
            w_ref,
            sgd_batch_gradient(dataset, rows, w_ref),
            sgd_batch_gradient(dataset, rows, ref.snapshot),
        )
        assert np.allclose(w.detach().numpy(), w_ref, atol=1e-8), f"diverged from oracle at step {k}"
    assert opt.step_count == ref.step_count == steps


@requires_jax
def test_jax_matches_numpy_reference_trajectory() -> None:
    dataset, sampler, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
    interval, steps = 5, 10
    config = SVRGConfig(lr=0.02, snapshot_interval=interval)
    step_fn = make_svrg_step(loss_fn, config)
    params = jnp.zeros((dataset.n_features,), dtype=jnp.float32)
    state = init_state(params)
    ref = ReferenceSVRG(ReferenceConfig(lr=0.02, snapshot_interval=interval), dataset.n_features)
    w_ref = np.zeros(dataset.n_features)
    since = interval
    for k in range(steps):
        if since >= interval:
            state = refresh_snapshot(params, state, compute_full_gradient(loss_fn, params, chunk_stack))
            since = 0
        params, state, _ = step_fn(params, state, batch_at(k))
        since += 1
        if ref.needs_snapshot:
            ref.refresh_snapshot(w_ref, dataset.full_gradient(w_ref))
        rows = sampler.batch(k)
        w_ref = ref.step(
            w_ref,
            sgd_batch_gradient(dataset, rows, w_ref),
            sgd_batch_gradient(dataset, rows, ref.snapshot),
        )
        assert np.allclose(np.asarray(params), w_ref, atol=1e-4), f"diverged from oracle at step {k}"


# --------------------------------------------------------------------------- #
# Fuzz tests for the engines (same adversarial inputs as the oracle)
# --------------------------------------------------------------------------- #
@requires_torch
def test_torch_fuzz_never_produces_nonfinite_values() -> None:
    rng = np.random.default_rng(7)
    for dtype in (torch.float32, torch.float64):
        for trial in range(40):
            w = torch.nn.Parameter(torch.randn(6, dtype=dtype))
            opt = CoordinateSVRG(
                [w], lr=float(10 ** rng.uniform(-4, 0)),
                betas=(float(rng.choice([0.0, 0.9, 0.9999])), float(rng.choice([0.0, 0.999, 0.9999]))),
                snapshot_interval=3,
            )
            for step in range(10):
                scale = torch.as_tensor(_fuzz_gradient(rng, 6), dtype=dtype)
                snap_scale = torch.as_tensor(_fuzz_gradient(rng, 6), dtype=dtype)
                holder = {"scale": scale}

                def closure(batch_id: int) -> "torch.Tensor":
                    return (w * holder["scale"]).sum()

                if opt.needs_snapshot:
                    holder["scale"] = snap_scale
                    opt.refresh_snapshot(closure)
                    holder["scale"] = scale
                opt.step(closure)
                assert torch.isfinite(w).all(), (dtype, trial, step)
                # Proposition 4 covers the update path: weights and both moments. The snapshot
                # rows ('snapshot', 'mu') store their inputs verbatim, so a non-finite full
                # gradient fed to refresh_snapshot is kept as given (the oracle and the JAX
                # engine behave identically), and the update path then sanitises it.
                moments = opt.flat_state()
                assert torch.isfinite(moments["exp_avg"]).all(), (dtype, trial, step)
                assert torch.isfinite(moments["exp_avg_sq"]).all(), (dtype, trial, step)


@requires_jax
def test_jax_fuzz_never_produces_nonfinite_values() -> None:
    rng = np.random.default_rng(11)
    for trial in range(30):
        config = SVRGConfig(
            lr=float(10 ** rng.uniform(-4, 0)),
            beta1=float(rng.choice([0.0, 0.9, 0.9999])), beta2=float(rng.choice([0.0, 0.999, 0.9999])),
        )
        params = jnp.asarray(rng.standard_normal(6), dtype=jnp.float32)
        state = init_state(params)
        for step in range(10):
            scale = jnp.asarray(_fuzz_gradient(rng, 6), dtype=jnp.float32)
            snap_scale = jnp.asarray(_fuzz_gradient(rng, 6), dtype=jnp.float32)
            step_fn = make_svrg_step(lambda p, b, s=scale: jnp.sum(p * s), config)
            if step % 3 == 0:
                state = refresh_snapshot(params, state, snap_scale)
            params, state, _ = step_fn(params, state, None)
            assert bool(jnp.all(jnp.isfinite(params))), (trial, step)
            assert bool(jnp.all(jnp.isfinite(state.exp_avg)))
            assert bool(jnp.all(jnp.isfinite(state.exp_avg_sq)))


# --------------------------------------------------------------------------- #
# Frozen diagonal preconditioner in the engines (Proposition 5)
# --------------------------------------------------------------------------- #
def _frozen_diag(dim: int) -> np.ndarray:
    return np.exp(np.random.default_rng(5).uniform(-1.0, 1.0, size=dim))


@requires_torch
def test_torch_frozen_preconditioner_matches_numpy_reference_trajectory() -> None:
    for beta1 in (0.0, 0.9):
        dataset, sampler, chunks, w, batch_closure, chunk_closure = _torch_ls_setup()
        diag = _frozen_diag(dataset.n_features)
        interval, steps = 5, 14
        opt = CoordinateSVRG([w], lr=0.02, betas=(beta1, 0.999), snapshot_interval=interval)
        opt.set_frozen_preconditioner(torch.as_tensor(diag))
        ref = ReferenceSVRG(
            ReferenceConfig(lr=0.02, beta1=beta1, snapshot_interval=interval), dataset.n_features
        )
        ref.set_preconditioner(diag)
        w_ref = np.zeros(dataset.n_features)
        for k in range(steps):
            if opt.needs_snapshot:
                opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
            opt.step(batch_closure)
            if ref.needs_snapshot:
                ref.refresh_snapshot(w_ref, dataset.full_gradient(w_ref))
            rows = sampler.batch(k)
            w_ref = ref.step(
                w_ref,
                sgd_batch_gradient(dataset, rows, w_ref),
                sgd_batch_gradient(dataset, rows, ref.snapshot),
            )
            assert np.allclose(w.detach().numpy(), w_ref, atol=1e-8), (beta1, k)


@requires_torch
def test_torch_frozen_preconditioner_validation_and_clear() -> None:
    w_a = torch.nn.Parameter(torch.ones(4))
    w_b = torch.nn.Parameter(torch.ones(4))
    opt_a = CoordinateSVRG([w_a], lr=0.1)
    opt_b = CoordinateSVRG([w_b], lr=0.1)
    with pytest.raises(ValueError):
        opt_b.set_frozen_preconditioner(torch.ones(3))                      # wrong size
    with pytest.raises(ValueError):
        opt_b.set_frozen_preconditioner(torch.tensor([1.0, 1.0, 0.0, 1.0]))  # not positive
    opt_b.set_frozen_preconditioner([torch.full((4,), 2.0)])                # list of tensors accepted
    opt_b.clear_frozen_preconditioner()

    def make_closure(w: "torch.nn.Parameter") -> Callable[[int], "torch.Tensor"]:
        return lambda _: ((w - 3.0) ** 2).sum()

    for opt, w in ((opt_a, w_a), (opt_b, w_b)):
        closure = make_closure(w)
        opt.refresh_snapshot(closure)
        opt.step(closure)
    assert torch.equal(w_a.detach(), w_b.detach())         # cleared == never set (adaptive path)


@requires_torch
def test_torch_frozen_preconditioner_survives_load_state_dict_on_same_object() -> None:
    dataset, _, chunks, w_a, closure_a, chunk_a = _torch_ls_setup()
    _, _, _, w_b, closure_b, chunk_b = _torch_ls_setup()
    diag = torch.as_tensor(_frozen_diag(dataset.n_features))
    opt_a = CoordinateSVRG([w_a], lr=0.02, snapshot_interval=1000)
    opt_b = CoordinateSVRG([w_b], lr=0.02, snapshot_interval=1000)
    for opt, chunk_closure in ((opt_a, chunk_a), (opt_b, chunk_b)):
        opt.set_frozen_preconditioner(diag)
        opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
    opt_a.step(closure_a)
    opt_b.step(closure_b)
    opt_a.load_state_dict(copy.deepcopy(opt_a.state_dict()))     # rebuilds the flat blocks
    opt_a.step(closure_a)                                         # must still use the frozen D
    opt_b.step(closure_b)
    assert torch.allclose(w_a.detach(), w_b.detach(), atol=1e-12)


@requires_jax
def test_jax_frozen_preconditioner_matches_numpy_reference_trajectory() -> None:
    for beta1 in (0.0, 0.9):
        dataset, sampler, loss_fn, batch_at, chunk_stack = _jax_ls_setup()
        diag = _frozen_diag(dataset.n_features)
        interval, steps = 5, 10
        config = SVRGConfig(lr=0.02, beta1=beta1, snapshot_interval=interval)
        step_fn = make_svrg_step(loss_fn, config, preconditioner=jnp.asarray(diag, dtype=jnp.float32))
        params = jnp.zeros((dataset.n_features,), dtype=jnp.float32)
        state = init_state(params)
        ref = ReferenceSVRG(
            ReferenceConfig(lr=0.02, beta1=beta1, snapshot_interval=interval), dataset.n_features
        )
        ref.set_preconditioner(diag)
        w_ref = np.zeros(dataset.n_features)
        since = interval
        for k in range(steps):
            if since >= interval:
                state = refresh_snapshot(params, state, compute_full_gradient(loss_fn, params, chunk_stack))
                since = 0
            params, state, _ = step_fn(params, state, batch_at(k))
            since += 1
            if ref.needs_snapshot:
                ref.refresh_snapshot(w_ref, dataset.full_gradient(w_ref))
            rows = sampler.batch(k)
            w_ref = ref.step(
                w_ref,
                sgd_batch_gradient(dataset, rows, w_ref),
                sgd_batch_gradient(dataset, rows, ref.snapshot),
            )
            assert np.allclose(np.asarray(params), w_ref, atol=1e-4), (beta1, k)


@requires_jax
def test_jax_frozen_preconditioner_rejects_invalid_values() -> None:
    config = SVRGConfig(lr=0.1)
    with pytest.raises(ValueError):
        make_svrg_step(lambda p, b: jnp.sum(p), config, preconditioner=jnp.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        make_svrg_step(lambda p, b: jnp.sum(p), config, preconditioner=jnp.array([1.0, jnp.nan]))
