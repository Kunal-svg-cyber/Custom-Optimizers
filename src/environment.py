"""Toxic market simulator: a hidden, shifting alpha buried under hostile noise.

This module only *generates data*. It contains no optimizer math and no test
assertions (strict decoupling). Everything is NumPy-only so both the PyTorch and
JAX backends can consume it without importing either framework here.

The simulated tick stream is::

    y_t = x_t . alpha_t + gaussian_t + toxic_flow_t + jump_t

* ``x_t``       AR(1) microstructure features.
* ``alpha_t``   hidden signal: slow random-walk drift plus rare full regime
                switches.
* ``gaussian``  ordinary high-frequency noise, scaled by the chosen sigma level.
* ``toxic``     rare, heavy-tailed (Student-t) institutional order-flow shocks.
* ``jump``      rare, large, signed arbitrage jumps.

Optimizers see a finite-sum least-squares landscape over the whole history, so
SVRG's snapshot full-gradient is well defined. ``DeterministicBatchSampler``
maps a global step index to a fixed set of row indices, which is what makes
sample alignment between the live and snapshot evaluations checkable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import yaml

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class RegimeConfig:
    """Immutable description of one stochastic market regime."""

    seed: int
    n_samples: int
    n_features: int
    feature_autocorr: float
    signal_scale: float
    alpha_drift_sigma: float
    regime_switch_prob: float
    noise_sigma: float
    toxic_flow_prob: float
    toxic_flow_scale: float
    toxic_flow_df: float
    jump_prob: float
    jump_scale: float

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if self.n_features < 1:
            raise ValueError("n_features must be at least 1")
        if not -1.0 < self.feature_autocorr < 1.0:
            raise ValueError("feature_autocorr must lie strictly in (-1, 1)")
        if self.signal_scale < 0.0 or self.alpha_drift_sigma < 0.0:
            raise ValueError("signal_scale and alpha_drift_sigma must be >= 0")
        if self.noise_sigma < 0.0:
            raise ValueError("noise_sigma must be >= 0")
        for name in ("regime_switch_prob", "toxic_flow_prob", "jump_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.toxic_flow_df <= 0.0:
            raise ValueError("toxic_flow_df must be positive")
        if self.toxic_flow_scale < 0.0 or self.jump_scale < 0.0:
            raise ValueError("toxic_flow_scale and jump_scale must be >= 0")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RegimeConfig":
        """Build a config from the parsed YAML mapping (top-level dict)."""
        env: Dict[str, Any] = dict(config["environment"])
        levels: Dict[str, float] = dict(env["noise_sigma_levels"])
        regime_name: str = str(env["noise_regime"])
        if regime_name not in levels:
            raise ValueError(
                f"noise_regime '{regime_name}' not in noise_sigma_levels {sorted(levels)}"
            )
        return cls(
            seed=int(config["seed"]),
            n_samples=int(env["n_samples"]),
            n_features=int(env["n_features"]),
            feature_autocorr=float(env["feature_autocorr"]),
            signal_scale=float(env["signal_scale"]),
            alpha_drift_sigma=float(env["alpha_drift_sigma"]),
            regime_switch_prob=float(env["regime_switch_prob"]),
            noise_sigma=float(levels[regime_name]),
            toxic_flow_prob=float(env["toxic_flow_prob"]),
            toxic_flow_scale=float(env["toxic_flow_scale"]),
            toxic_flow_df=float(env["toxic_flow_df"]),
            jump_prob=float(env["jump_prob"]),
            jump_scale=float(env["jump_scale"]),
        )


@dataclass(frozen=True, eq=False)
class MarketDataset:
    """A generated tick history plus the ground truth used to produce it."""

    features: FloatArray        # (n_samples, n_features)
    targets: FloatArray         # (n_samples,)
    alpha_path: FloatArray      # (n_samples, n_features) hidden alpha per tick
    gaussian_noise: FloatArray  # (n_samples,)
    toxic_flow: FloatArray      # (n_samples,)
    jumps: FloatArray           # (n_samples,)
    optimal_weights: FloatArray  # (n_features,) least-squares minimiser
    config: RegimeConfig

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def total_noise(self) -> FloatArray:
        """Sum of every corruption term added on top of the hidden signal."""
        return self.gaussian_noise + self.toxic_flow + self.jumps

    @property
    def empirical_snr_db(self) -> float:
        """Empirical signal-to-noise ratio of the targets, in decibels."""
        signal: FloatArray = np.einsum("td,td->t", self.features, self.alpha_path)
        noise_power: float = float(np.var(self.total_noise))
        signal_power: float = float(np.var(signal))
        if noise_power <= 0.0:
            return math.inf
        if signal_power <= 0.0:
            return -math.inf
        return 10.0 * math.log10(signal_power / noise_power)

    def full_gradient(self, weights: FloatArray) -> FloatArray:
        """Exact gradient of f(w) = 1/(2n) ||Xw - y||^2 (reference for tests)."""
        residual: FloatArray = self.features @ weights - self.targets
        return (self.features.T @ residual) / float(self.n_samples)

    def loss(self, weights: FloatArray) -> float:
        """Exact value of f(w) = 1/(2n) ||Xw - y||^2."""
        residual: FloatArray = self.features @ weights - self.targets
        return float(0.5 * np.mean(residual * residual))


def load_config(path: PathLike) -> Dict[str, Any]:
    """Parse a YAML regime file into a plain dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        parsed: Any = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return parsed


def generate_market(config: RegimeConfig) -> MarketDataset:
    """Generate a reproducible toxic-market dataset from ``config``."""
    n: int = config.n_samples
    d: int = config.n_features

    # Independent child streams: changing one noise knob never reshuffles the
    # features or the hidden alpha.
    feature_seq, alpha_seq, noise_seq = np.random.SeedSequence(config.seed).spawn(3)
    rng_x = np.random.default_rng(feature_seq)
    rng_a = np.random.default_rng(alpha_seq)
    rng_n = np.random.default_rng(noise_seq)

    # --- AR(1) microstructure features -------------------------------------
    innovations: FloatArray = rng_x.standard_normal((n, d))
    rho: float = config.feature_autocorr
    scale: float = math.sqrt(1.0 - rho * rho)
    features: FloatArray = np.empty((n, d), dtype=np.float64)
    features[0] = innovations[0]
    for t in range(1, n):
        features[t] = rho * features[t - 1] + scale * innovations[t]

    # --- Hidden alpha: slow drift with rare regime switches ----------------
    switches: npt.NDArray[np.bool_] = rng_a.random(n) < config.regime_switch_prob
    drift: FloatArray = config.alpha_drift_sigma * rng_a.standard_normal((n, d))
    redraws: FloatArray = (
        config.signal_scale * rng_a.standard_normal((n, d)) / math.sqrt(d)
    )
    alpha_path: FloatArray = np.empty((n, d), dtype=np.float64)
    current: FloatArray = redraws[0].copy()
    for t in range(n):
        if switches[t]:
            current = redraws[t].copy()
        else:
            current = current + drift[t]
        alpha_path[t] = current

    signal: FloatArray = np.einsum("td,td->t", features, alpha_path)

    # --- Corruption: HF noise, toxic order flow, arbitrage jumps -----------
    gaussian_noise: FloatArray = config.noise_sigma * rng_n.standard_normal(n)

    toxic_mask: FloatArray = (rng_n.random(n) < config.toxic_flow_prob).astype(np.float64)
    toxic_shocks: FloatArray = config.toxic_flow_scale * rng_n.standard_t(
        config.toxic_flow_df, size=n
    )
    toxic_flow: FloatArray = toxic_mask * toxic_shocks

    jump_mask: FloatArray = (rng_n.random(n) < config.jump_prob).astype(np.float64)
    jump_sign: FloatArray = rng_n.choice(np.array([-1.0, 1.0]), size=n)
    jump_size: FloatArray = config.jump_scale * (1.0 + rng_n.exponential(1.0, size=n))
    jumps: FloatArray = jump_mask * jump_sign * jump_size

    targets: FloatArray = signal + gaussian_noise + toxic_flow + jumps
    optimal_weights: FloatArray = np.linalg.lstsq(features, targets, rcond=None)[0]

    return MarketDataset(
        features=features,
        targets=targets,
        alpha_path=alpha_path,
        gaussian_noise=gaussian_noise,
        toxic_flow=toxic_flow,
        jumps=jumps,
        optimal_weights=optimal_weights,
        config=config,
    )


def market_from_yaml(path: PathLike) -> MarketDataset:
    """Convenience wrapper: YAML file -> RegimeConfig -> MarketDataset."""
    return generate_market(RegimeConfig.from_dict(load_config(path)))


class DeterministicBatchSampler:
    """Maps a global step index to a fixed mini-batch of row indices.

    The same ``step`` always yields the same indices, on every call and every
    process. Optimizers pass that one step/batch id to both the live evaluation
    ``g_t(w_t)`` and the snapshot evaluation ``g_t(w_snap)``, so the two
    gradients are computed on identical samples (Deterministic Sample
    Alignment). Each epoch is an independent permutation derived from
    ``(seed, epoch)``; the trailing remainder of an epoch is dropped so every
    batch has exactly ``batch_size`` rows.
    """

    def __init__(self, n_samples: int, batch_size: int, seed: int) -> None:
        if batch_size < 1 or batch_size > n_samples:
            raise ValueError("batch_size must satisfy 1 <= batch_size <= n_samples")
        self.n_samples: int = int(n_samples)
        self.batch_size: int = int(batch_size)
        self.seed: int = int(seed)
        self.batches_per_epoch: int = self.n_samples // self.batch_size
        self._cached_epoch: int = -1
        self._cached_permutation: Optional[IntArray] = None

    def epoch_permutation(self, epoch: int) -> IntArray:
        """Return the (cached) row permutation used during ``epoch``."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch != self._cached_epoch or self._cached_permutation is None:
            rng = np.random.default_rng([self.seed, epoch])
            self._cached_permutation = rng.permutation(self.n_samples).astype(np.int64)
            self._cached_epoch = epoch
        return self._cached_permutation

    def batch(self, step: int) -> IntArray:
        """Return the row indices for global step ``step`` (a fresh copy)."""
        if step < 0:
            raise ValueError("step must be non-negative")
        epoch, position = divmod(int(step), self.batches_per_epoch)
        permutation: IntArray = self.epoch_permutation(epoch)
        start: int = position * self.batch_size
        return permutation[start:start + self.batch_size].copy()

    def full_pass_chunks(self, num_chunks: int) -> List[IntArray]:
        """Split all rows into ``num_chunks`` equal, ordered chunks.

        Equal sizes make the mean of per-chunk mean-gradients exactly the
        full-data gradient, which is what the SVRG snapshot needs.
        """
        if num_chunks < 1 or self.n_samples % num_chunks != 0:
            raise ValueError("num_chunks must be >= 1 and divide n_samples evenly")
        all_rows: IntArray = np.arange(self.n_samples, dtype=np.int64)
        return [chunk.copy() for chunk in np.split(all_rows, num_chunks)]
